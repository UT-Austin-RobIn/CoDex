"""Primitive-aware functional grasp synthesis for the LEAP hand.

A grasp is only useful to CoDex if the fixed grasp-move-actuate primitive can
execute it: the hand slides in along its palm normal with every joint at half
of the grasp configuration, the support fingers close (plus a 0.2 rad PIP/DIP
squeeze) while the actuation finger stays at half curl, the object is lifted,
and only then does the actuation finger close onto the functional point.
This module therefore optimises every kinematic state of that primitive, not
only the final static pose:

  grasp state     actuation pad a few mm past the keypoint, aligned with the
                  press direction; support fingers commanded slightly inside
                  the object so the squeeze produces grip force; no other
                  hand/object or hand/table penetration; fingers neither
                  collide nor cross over each other
  approach        the half-curled hand swept along the approach and descent
                  path stays clear of object and table
  hold / press    the parked actuation finger is clear of the object and its
                  closing path moves the pad into the keypoint from outside

Candidates are seeded by anchoring the actuation pad on the keypoint (the
base pose is solved from the actuation finger's forward kinematics, so the
functional constraint holds from the first iteration) and optimised jointly
over base pose and all 16 joints. Geometric feasibility is only a proposal
filter: candidates are selected by simulating the primitive itself
(`sim_screen.py`), because static certificates do not predict whether a grasp
survives it.

Geometry is the simulator's collision geometry: the object distance field is
the minimum over the CoACD convex parts (exact outside the object), the hand
is sampled from the collision meshes of `leap_sim.urdf`.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d
import pytorch_kinematics as pk
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial.transform import Rotation

_MODELS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "models", "leap_rh"))

FINGERS = ("if", "mf", "rf", "th")
JOINT_ORDER = [f"leap_rh_{f}_{j}" for f in FINGERS[:3] for j in ("axl", "mcp", "pip", "dip")] + \
              [f"leap_rh_th_{j}" for j in ("cmc", "axl", "mcp", "ipl")]
# Distal pad centres in the distal-link frame; the pad faces -X of that frame.
# Identical to the simulator's LEAP_PAD_OFFSETS_RAW (act_dist is measured here).
PAD_OFFSETS = ((-0.010, -0.032, 0.015),) * 3 + ((-0.009, -0.035, -0.011),)
# Primitive constants (primitive_env.py).
PREGRASP_SCALE = 0.5      # zero-action pregrasp = 0.5 * grasp joints
SQUEEZE = 0.2             # support PIP/DIP closing past the grasp joints
APPROACH_OFFSET = 0.10    # pre-contact standoff along the palm normal
ACT_PRELOAD = 0.006       # actuation pad target this far past the keypoint
# Position control grips only where the commanded pose lies inside the object.
# Grasps that survive the primitive (zero-action lifts of the policy-selected
# grasps) command their support links 5-8 mm into the object and 20-28 mm
# after the squeeze; that interpenetration is what absorbs ~2 cm of hand pose
# error. Exact-contact targets close on air under the same error.
SUPPORT_DEPTH = 0.006     # support contact depth at the grasp configuration
SUPPORT_PRELOAD = 0.018   # squeezed support fingers must reach this deep
SUPPORT_MAX_DEPTH = 0.012 # closing links may be commanded this far inside
# Index, middle and ring keep their side-by-side order: the spread joints
# (+-60 deg) otherwise let a finger cross over its neighbour at a different
# depth, which the self-collision spheres do not see. Measured along the
# knuckle line at the middle phalanx and the pad.
FINGER_GAP = 0.015
DESCENT_HEIGHT = 0.25     # pre-pre-contact pose above the pre-contact pose
W_FC = 10.0               # weight of the normal-force balance surrogate
W_APERTURE = 50.0         # weight of the grasp-aperture prior

_LOWER = np.array([-0.314, -1.047, -0.506, -0.366] * 3 + [-0.349, -0.47, -1.2, -1.34], np.float32)
_UPPER = np.array([2.23, 1.047, 1.885, 2.042] * 3 + [2.094, 2.443, 1.9, 1.88], np.float32)


# ---------------------------------------------------------------------------
# Object distance field
# ---------------------------------------------------------------------------

class PartUnionSDF:
    """Signed distance to the union of convex parts on a regular grid.

    min_k sdf_k is the exact Euclidean distance to the union for points outside
    every part (the region hand geometry must stay in) and negative inside.
    """

    def __init__(self, parts, padding=0.12, resolution=160, device="cuda"):
        verts = np.vstack([p.vertices for p in parts])
        self.lo = verts.min(0) - padding
        self.hi = verts.max(0) + padding
        axes = [np.linspace(self.lo[i], self.hi[i], resolution) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
        sdf = np.full(len(grid), np.inf, np.float32)
        for p in parts:
            scene = o3d.t.geometry.RaycastingScene()
            m = o3d.t.geometry.TriangleMesh()
            m.vertex.positions = o3d.core.Tensor(np.asarray(p.vertices, np.float32))
            m.triangle.indices = o3d.core.Tensor(np.asarray(p.faces, np.int32))
            scene.add_triangles(m)
            sdf = np.minimum(sdf, scene.compute_signed_distance(o3d.core.Tensor(grid)).numpy())
        self.grid = torch.tensor(sdf.reshape((resolution,) * 3), device=device)[None, None]
        self.lo_t = torch.tensor(self.lo, dtype=torch.float32, device=device)
        self.range_t = torch.tensor(self.hi - self.lo, dtype=torch.float32, device=device)

    def __call__(self, pts):
        """pts [B, N, 3] -> signed distance [B, N] (differentiable)."""
        B, N, _ = pts.shape
        u = 2.0 * (pts - self.lo_t) / self.range_t - 1.0
        out = F.grid_sample(self.grid.expand(B, -1, -1, -1, -1), u[..., [2, 1, 0]].view(B, N, 1, 1, 3),
                            align_corners=True, padding_mode="border")
        return out.view(B, N)

    def normal(self, pts, eps=2e-3):
        """Outward unit normal by central differences (differentiable in pts)."""
        g = torch.stack([self(pts + e) - self(pts - e) for e in
                         torch.eye(3, device=pts.device) * eps], -1)
        return F.normalize(g, dim=-1)


# ---------------------------------------------------------------------------
# Hand model
# ---------------------------------------------------------------------------

def _origin(elem):
    o = elem.find("origin")
    T = np.eye(4)
    if o is not None:
        T[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
        T[:3, :3] = Rotation.from_euler("xyz", [float(x) for x in o.get("rpy", "0 0 0").split()]).as_matrix()
    return T


class LeapHand:
    """Differentiable LEAP kinematics with collision surface samples."""

    POINTS = {"palm": 400, "bs": 24, "px": 40, "md": 40, "ds": 80, "mp": 24}

    def __init__(self, device="cuda", seed=0):
        self.device = device
        with open(os.path.join(_MODELS_DIR, "leap_solver.urdf")) as f:
            self.chain = pk.build_chain_from_urdf(f.read()).to(device=device)
        if list(self.chain.get_joint_parameter_names()) != JOINT_ORDER:
            raise RuntimeError("LEAP chain joint order differs from the grasp-file convention")
        self.lo = torch.tensor(_LOWER, device=device)
        self.hi = torch.tensor(_UPPER, device=device)

        rng = np.random.default_rng(seed)
        root = ET.parse(os.path.join(_MODELS_DIR, "leap_sim.urdf")).getroot()
        self.links, self.pts, self.nrm, self.finger_of = [], [], [], []
        for link in root.findall("link"):
            name = link.get("name")
            if not name.startswith("leap_rh_"):
                continue
            meshes = []
            for col in link.findall("collision"):
                mesh = col.find("geometry/mesh")
                if mesh is None:
                    continue
                m = trimesh.load(os.path.join(_MODELS_DIR, mesh.get("filename")), force="mesh")
                m.apply_transform(_origin(col))
                meshes.append(m)
            if not meshes:
                continue
            m = trimesh.util.concatenate(meshes)
            n = self.POINTS[name.split("_")[-1]]
            p, fi = trimesh.sample.sample_surface_even(m, n * 3, seed=int(rng.integers(1 << 31)))
            keep = rng.choice(len(p), min(n, len(p)), replace=False)
            self.links.append(name)
            self.pts.append(torch.tensor(np.c_[p[keep], np.ones(len(keep))], dtype=torch.float32, device=device))
            self.nrm.append(torch.tensor(m.face_normals[fi[keep]], dtype=torch.float32, device=device))
            part = name.split("_")[2]
            self.finger_of.append(FINGERS.index(part) if part in FINGERS else -1)
        self.tips = [f"leap_rh_{f}_ds" for f in FINGERS]
        self.pad = torch.tensor([list(o) + [1.0] for o in PAD_OFFSETS], device=device)
        # Pad-side samples of the distal links: contact there is expected and
        # the squeeze presses the soft pad in, so a small overlap is allowed.
        pad_side = []
        for name, p, n in zip(self.links, self.pts, self.nrm):
            flag = torch.zeros(p.shape[0], dtype=torch.bool, device=device)
            if name in self.tips:
                c = self.pad[self.tips.index(name), :3]
                flag = (n[:, 0] < -0.3) & ((p[:, :3] - c).norm(dim=-1) < 0.022)
            pad_side.append(flag)
        self.pad_side = torch.cat(pad_side)
        self.point_finger = torch.cat([torch.full((p.shape[0],), f, device=device)
                                       for p, f in zip(self.pts, self.finger_of)])
        self.point_is_tip = torch.cat([torch.full((p.shape[0],), n in self.tips, device=device)
                                       for p, n in zip(self.pts, self.links)])
        # Links moved by the squeezed joints (PIP/DIP; thumb MCP/IP): only
        # contacts there are pressed into the object when the hand closes.
        closing = {f: ("md", "ds") for f in range(3)}
        closing[3] = ("px", "ds")
        self.contact_mask = [torch.cat([torch.full((p.shape[0],), fi == f and n.split("_")[-1] in closing[f],
                                                   device=device)
                                        for p, n, fi in zip(self.pts, self.links, self.finger_of)])
                             for f in range(4)]

        # Palm contact face: palm samples whose normal faces the grasp side
        # (+X of the hand base frame).
        q0 = torch.zeros(1, 16, device=device)
        T_palm = self.chain.forward_kinematics(q0)["leap_rh_palm"].get_matrix()[0]
        pi = self.links.index("leap_rh_palm")
        n_base = self.nrm[pi] @ T_palm[:3, :3].T
        face = n_base[:, 0] > 0.9
        self.palm_face = self.pts[pi][face]
        pf = (T_palm @ self.palm_face.T).T[:, :3]
        # Grasp-aperture reference: the palm-face centre, displaced along the
        # palm normal (+X) by a per-candidate depth when used.
        self.palm_center = pf.mean(0)
        self.self_spheres = self._spheres()
        knuckles = [self.chain.forward_kinematics(q0)[f"leap_rh_{f}_px"].get_matrix()[0, :3, 3] for f in FINGERS[:3]]
        self.knuckle_line = F.normalize(knuckles[2] - knuckles[0], dim=0)

    def _spheres(self):
        """Sphere approximation of the finger boxes (self-collision proxy)."""
        root = ET.parse(os.path.join(_MODELS_DIR, "leap_solver.urdf")).getroot()
        spheres = []  # (link, centre_h [4], radius, finger)
        for link in root.findall("link"):
            name = link.get("name")
            part = name.split("_")[2] if name.startswith("leap_rh_") else ""
            if part not in FINGERS or name.endswith("_bs") or name.endswith("_mp"):
                continue
            for col in link.findall("collision"):
                box = col.find("geometry/box")
                if box is None:
                    continue
                size = np.array([float(x) for x in box.get("size").split()])
                T = _origin(col)
                ax = int(np.argmax(size))
                r = float(np.sort(size)[1] / 2)
                n = max(1, int(np.ceil(size[ax] / (2 * r))))
                for t in np.linspace(-0.5, 0.5, n + 2)[1:-1] if n > 1 else [0.0]:
                    c = np.zeros(3)
                    c[ax] = t * (size[ax] - 2 * r) if n > 1 else 0.0
                    spheres.append((name, T[:3, :3] @ c + T[:3, 3], r, FINGERS.index(part)))
        return spheres

    def fk(self, q):
        """q [B,16] -> {link: [B,4,4]} in the hand base frame."""
        ret = self.chain.forward_kinematics(q)
        return {k: v.get_matrix() for k, v in ret.items()}

    def points(self, fk, T, links=None):
        """World collision samples of `links` (all by default): [B, N, 3]."""
        out = []
        for name, p in zip(self.links, self.pts):
            if links is not None and name not in links:
                continue
            out.append(((T @ fk[name]) @ p.T).transpose(1, 2)[..., :3])
        return torch.cat(out, 1)

    def pads(self, fk, T):
        """Pad centres [B,4,3] and press axes (pad normals, -X of distal) [B,4,3]."""
        pos, ax = [], []
        for i, tip in enumerate(self.tips):
            M = T @ fk[tip]
            pos.append((M @ self.pad[i])[:, :3])
            ax.append(-M[:, :3, 0])
        return torch.stack(pos, 1), torch.stack(ax, 1)

    def palm_points(self, fk, T):
        return ((T @ fk["leap_rh_palm"]) @ self.palm_face.T).transpose(1, 2)[..., :3]

    def finger_gaps(self, fk):
        """Side-by-side gaps of index-middle and middle-ring along the knuckle
        line, at the middle-phalanx origin and at the pad [B, 4] (hand frame).
        Negative when a finger has crossed over its neighbour."""
        md = torch.stack([fk[f"leap_rh_{f}_md"][:, :3, 3] for f in FINGERS[:3]], 1)
        pad = torch.stack([(fk[self.tips[i]] @ self.pad[i])[:, :3] for i in range(3)], 1)
        lat = torch.cat([md, pad], 1) @ self.knuckle_line          # [B, 6]
        return torch.cat([lat[:, 1:3] - lat[:, 0:2], lat[:, 4:6] - lat[:, 3:5]], 1)

    def self_collision(self, fk, T, margin=0.002):
        """Penetration depth of the worst sphere pair on different fingers [B]."""
        cs, rs, fs = [], [], []
        for name, c, r, f in self.self_spheres:
            M = T @ fk[name]
            cs.append(M[:, :3, :3] @ torch.tensor(c, dtype=torch.float32, device=M.device) + M[:, :3, 3])
            rs.append(r)
            fs.append(f)
        C = torch.stack(cs, 1)
        r = torch.tensor(rs, device=C.device)
        f = torch.tensor(fs, device=C.device)
        d = torch.cdist(C, C) - (r[:, None] + r[None, :])
        other = (f[:, None] != f[None, :])
        d = torch.where(other, d, torch.full_like(d, 1.0))
        return F.relu(margin - d).amax((1, 2))


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rot6d_to_matrix(r):
    a, b = F.normalize(r[:, :3], dim=-1), r[:, 3:]
    b = F.normalize(b - (a * b).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


def axis_angle(axis, angle):
    """Rodrigues: axis [B,3] unit, angle [B] -> [B,3,3]."""
    K = torch.zeros(axis.shape[0], 3, 3, device=axis.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K = K - K.transpose(1, 2)
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return torch.eye(3, device=axis.device) + s * K + (1 - c) * K @ K


def align(u, v):
    """Rotation taking unit u onto unit v (batched)."""
    axis = torch.cross(u, v, dim=-1)
    s = axis.norm(dim=-1)
    c = (u * v).sum(-1)
    fallback = torch.cross(u, torch.tensor([1.0, 0, 0], device=u.device).expand_as(u), dim=-1)
    fallback = torch.where(fallback.norm(dim=-1, keepdim=True) < 1e-3,
                           torch.cross(u, torch.tensor([0, 1.0, 0], device=u.device).expand_as(u), dim=-1),
                           fallback)
    axis = torch.where(s[:, None] < 1e-6, fallback, axis)
    return axis_angle(F.normalize(axis, dim=-1), torch.atan2(s, c))


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

# Pressing-posture sampling ranges per joint of the ACTUATION finger. The
# press stroke runs from 0.5*q to q, so a flexed q gives the pad a real
# stroke; support fingers start open and are pulled onto the surface.
_ACT_RANGE = {False: [(0.3, 1.5), (-0.25, 0.25), (0.1, 1.3), (0.1, 1.2)],     # IF/MF/RF
              True: [(0.3, 1.8), (0.0, 1.4), (-0.4, 1.0), (-0.2, 1.0)]}      # TH
_SUP_RANGE = {False: [(0.0, 1.2), (-0.3, 0.3), (0.0, 1.0), (0.0, 0.8)],
              True: [(0.2, 1.6), (0.0, 1.2), (-0.4, 0.8), (-0.2, 0.8)]}


class PrimitiveGraspSynthesizer:

    def __init__(self, parts, act_pos, act_dir, device="cuda", seed=0):
        self.device = device
        self.sdf = PartUnionSDF(parts, device=device)
        self.hand = LeapHand(device=device, seed=seed)
        self.a = torch.tensor(act_pos, dtype=torch.float32, device=device)
        self.d = F.normalize(torch.tensor(act_dir, dtype=torch.float32, device=device), dim=0)
        verts = np.vstack([p.vertices for p in parts])
        self.center = torch.tensor(verts.mean(0), dtype=torch.float32, device=device)
        self.scale = float(np.linalg.norm(verts.max(0) - verts.min(0)) / 2)
        self.gen = torch.Generator(device=device).manual_seed(seed)

    # -- sampling ------------------------------------------------------------
    def _uniform(self, B, ranges):
        lo = torch.tensor([r[0] for r in ranges], device=self.device)
        hi = torch.tensor([r[1] for r in ranges], device=self.device)
        return lo + (hi - lo) * torch.rand(B, len(ranges), device=self.device, generator=self.gen)

    def sample(self, B, act_fingers=(0, 1, 2, 3), max_tilt=0.35):
        """Anchored seeds: actuation pad on the keypoint, pad axis within
        `max_tilt` of the press direction, uniform roll about it."""
        dev = self.device
        fa = torch.tensor(act_fingers, device=dev)[
            torch.randint(len(act_fingers), (B,), device=dev, generator=self.gen)]
        q = torch.zeros(B, 16, device=dev)
        for f in range(4):
            th = f == 3
            m = fa == f
            for g in range(4):
                rng = _ACT_RANGE[th] if g == f else _SUP_RANGE[th]
                q[:, 4 * g:4 * g + 4] = torch.where(m[:, None], self._uniform(B, rng),
                                                    q[:, 4 * g:4 * g + 4])
        q = q.clamp(self.hand.lo, self.hand.hi)

        fk = self.hand.fk(q)
        I = torch.eye(4, device=dev).expand(B, 4, 4)
        pad, ax = self.hand.pads(fk, I)
        rows = torch.arange(B, device=dev)
        pad_b, ax_b = pad[rows, fa], ax[rows, fa]

        # Tilted press axis around d, then roll about it.
        rnd = torch.randn(B, 3, device=dev, generator=self.gen)
        perp = F.normalize(rnd - (rnd @ self.d)[:, None] * self.d, dim=-1)
        tilt = max_tilt * torch.sqrt(torch.rand(B, device=dev, generator=self.gen))
        target_ax = F.normalize(torch.cos(tilt)[:, None] * self.d + torch.sin(tilt)[:, None] * perp, dim=-1)
        roll = 2 * np.pi * torch.rand(B, device=dev, generator=self.gen)
        Rw = axis_angle(target_ax, roll) @ align(ax_b, target_ax)
        pos = self.a + ACT_PRELOAD * self.d - (Rw @ pad_b[:, :, None])[..., 0]
        return pos, Rw, q, fa

    # -- energy --------------------------------------------------------------
    def energy(self, pos, rot6d, q, fa, palm_w, gc_depth, detail=False):
        """Primitive-aware grasp energy per candidate (lower is better).

        fa: actuation finger per candidate; palm_w: 1 where the palm is pulled
        onto the object (power-grasp prior); gc_depth: distance of the grasp
        aperture centre in front of the palm face.
        """
        dev = self.device
        B = q.shape[0]
        rows = torch.arange(B, device=dev)
        H = self.hand
        T = torch.eye(4, device=dev).repeat(B, 1, 1)
        T[:, :3, :3] = rot6d_to_matrix(rot6d)
        T[:, :3, 3] = pos
        x_hat = T[:, :3, 0]  # palm normal / approach direction

        act_mask = F.one_hot(fa, 4).bool()             # [B,4]
        E = {}

        # ---- grasp state
        fk = H.fk(q)
        pad, ax = H.pads(fk, T)
        # Position control only pushes when the target lies inside the
        # surface: aim the actuation pad slightly past the keypoint.
        d_act = (pad[rows, fa] - (self.a + ACT_PRELOAD * self.d)).norm(dim=-1)
        cos_act = (ax[rows, fa] * self.d).sum(-1)
        E["act"] = 1500 * (d_act / 0.01) ** 2 + 20 * F.relu(0.95 - cos_act)

        sup = ~act_mask

        pts = H.points(fk, T)
        s = self.sdf(pts)
        is_act_tip = H.point_is_tip[None] & (H.point_finger[None] == fa[:, None])
        # Support pads may compress slightly; the actuation pad's target is
        # deliberately ACT_PRELOAD past the surface.
        closing_sup = torch.zeros_like(s, dtype=torch.bool)
        for g in range(4):
            closing_sup |= H.contact_mask[g][None] & sup[:, g:g + 1]
        tol = torch.where(closing_sup, SUPPORT_MAX_DEPTH, 0.0)
        tol = torch.where(is_act_tip, 0.002, tol)
        tol = torch.where(is_act_tip & H.pad_side[None], ACT_PRELOAD + 0.003, tol)
        pen = F.relu(-s - tol)
        E["pen"] = 2000 * (pen / 0.01).pow(2).sum(-1)
        E["table"] = 2000 * (F.relu(0.004 - pts[..., 2]) / 0.01).pow(2).sum(-1)

        # Support contact: each support finger's closest point (proximal,
        # middle or distal link -- wraps load the phalanges, not only pads)
        # is pulled onto the surface. Soft-min keeps the gradient smooth.
        tau = 0.002
        cp, cs = [], []
        for g in range(4):
            m = H.contact_mask[g]
            sg = s[:, m]
            w = torch.softmax(-sg.detach() / tau, -1)
            cs.append(-tau * torch.logsumexp(-sg / tau, -1))
            cp.append((w[..., None] * pts[:, m]).sum(1))
        c_sdf, c_pos = torch.stack(cs, 1), torch.stack(cp, 1)   # [B,4], [B,4,3]
        contact = (c_sdf + SUPPORT_DEPTH).abs() / 0.01
        E["contact"] = (torch.where(contact < 1, contact ** 2, 2 * contact - 1) * sup).sum(-1) * 20

        # Preload: the primitive closes support PIP/DIP SQUEEZE rad past the
        # grasp; each support finger's squeezed target must lie inside the
        # object, otherwise closing produces no grip force.
        sq = torch.zeros_like(q)
        for g in range(4):
            sq[:, 4 * g + 2:4 * g + 4] = SQUEEZE * sup[:, g:g + 1].float()
        s_sq = self.sdf(H.points(H.fk(q + sq), T))
        depth = torch.stack([tau * torch.logsumexp(-s_sq[:, H.contact_mask[g]] / tau, -1)
                             for g in range(4)], 1)          # ~ max penetration
        E["preload"] = 50 * ((F.relu(SUPPORT_PRELOAD - depth) / 0.01) ** 2 * sup).sum(-1)

        # Opposition of the support contacts (+ palm when present): unit
        # normal forces should balance (DexGraspNet-style FC surrogate).
        palm_pts = H.palm_points(fk, T)
        palm_sdf = self.sdf(palm_pts)
        palm_gap, palm_idx = palm_sdf.min(-1)
        palm_p = palm_pts[rows, palm_idx]
        palm_n = self.sdf.normal(palm_p[:, None])[:, 0]
        w_palm = torch.sigmoid((0.01 - palm_gap.detach()) / 0.003)
        n_in = -self.sdf.normal(c_pos)
        fsum = (n_in * sup[..., None]).sum(1) + w_palm[:, None] * -palm_n
        tsum = (torch.cross(c_pos - self.center, n_in, dim=-1) * sup[..., None]).sum(1) + \
            w_palm[:, None] * torch.cross(palm_p - self.center, -palm_n, dim=-1)
        E["fc"] = W_FC * (fsum.pow(2).sum(-1) + (tsum / self.scale).pow(2).sum(-1))
        # Aperture: the hand's grasp centre (in front of the palm, between
        # thumb and fingers) must lie inside the object, i.e. the object sits
        # in the hand rather than beside it. Without this the support fingers
        # settle on whichever face is nearest, usually the same face.
        gc_b = H.palm_center + torch.stack([gc_depth, torch.zeros_like(gc_depth),
                                            torch.zeros_like(gc_depth)], -1)
        gc = (T[:, :3, :3] @ gc_b[..., None])[..., 0] + T[:, :3, 3]
        gc_sdf = self.sdf(gc[:, None])[:, 0]
        E["aperture"] = W_APERTURE * (F.relu(gc_sdf + 0.004) / 0.01) ** 2
        # Optional power-grasp prior: pull the palm face onto the object.
        E["palm"] = palm_w * 20 * (F.relu(palm_gap) / 0.01) ** 2
        E["self"] = 500 * (H.self_collision(fk, T) / 0.01) ** 2
        gaps = H.finger_gaps(fk)
        E["order"] = 500 * (F.relu(FINGER_GAP - gaps) / 0.01).pow(2).sum(-1)

        # ---- approach: half-curled hand swept along the approach/descent
        q_pre = PREGRASP_SCALE * q
        fk_pre = H.fk(q_pre)
        pts_pre = H.points(fk_pre, T)  # [B,N,3] at the grasp base pose
        ap = 0.0
        offsets = [(-s_, 0.0) for s_ in (0.0, 0.025, 0.05, 0.075, 0.10)] + \
                  [(-APPROACH_OFFSET, h) for h in (0.05, 0.12, DESCENT_HEIGHT)]
        z = torch.tensor([0, 0, 1.0], device=dev)
        for along, up in offsets:
            shift = along * x_hat + up * z
            pp = pts_pre + shift[:, None]
            ap = ap + (F.relu(-0.003 - self.sdf(pp)) / 0.01).pow(2).sum(-1)
            ap = ap + (F.relu(0.004 - pp[..., 2]) / 0.01).pow(2).sum(-1)
        E["approach"] = 100 * ap

        # ---- press stroke: the parked (half-curl) pad must sit outside the
        # keypoint along the press direction, and the stroke must be clear.
        pad_pre, _ = H.pads(fk_pre, T)
        stroke = ((self.a - pad_pre[rows, fa]) * self.d).sum(-1)
        E["stroke"] = 200 * (F.relu(0.015 - stroke) / 0.01) ** 2
        q_mid = torch.where(act_mask.repeat_interleave(4, 1), 0.75 * q, q_pre)
        pm = H.points(H.fk(q_mid), T)
        act_pts = H.point_finger[None, :] == fa[:, None]
        E["press_path"] = 1000 * ((F.relu(0.002 - self.sdf(pm)) / 0.01).pow(2) * act_pts).sum(-1)

        total = sum(E.values())
        if detail:
            info = dict(d_act=d_act, cos_act=cos_act, c_sdf=c_sdf, depth=depth, stroke=stroke,
                        pen_nonact=torch.where(is_act_tip | closing_sup, torch.ones_like(s), s).amin(-1),
                        table=pts[..., 2].amin(-1),
                        self_pen=H.self_collision(fk, T, margin=0.0), finger_gap=gaps.amin(-1))
            return total, E, info
        return total, E

    # -- optimisation ----------------------------------------------------------
    def optimize(self, B=2048, oversample=8, steps=300, act_fingers=(0, 1, 2, 3),
                 palm_frac=0.5, log_every=100, gc_range=(0.02, 0.06)):
        """Sample anchored seeds, keep the best B, optimise them jointly.

        Returns the final state (pos, rot, q, fa, energy, terms, info)."""
        dev = self.device
        # Oversample anchored seeds; keep those with the object in the hand's
        # aperture and the least collision along the primitive.
        seeds = []
        for _ in range(oversample):
            pos, Rw, q, fa = self.sample(B, act_fingers)
            r6 = torch.cat([Rw[:, :, 0], Rw[:, :, 1]], -1)
            gcd = gc_range[0] + (gc_range[1] - gc_range[0]) * torch.rand(B, device=dev, generator=self.gen)
            with torch.no_grad():
                e, E = self.energy(pos, r6, q, fa, torch.zeros(B, device=dev), gcd)
                pre = E["pen"] + E["table"] + E["approach"] + E["aperture"]
            seeds.append((pos, r6, q, fa, gcd, pre))
        pos, r6, q, fa, gcd, pre = [torch.cat(x) for x in zip(*seeds)]
        # Keep the best seeds per actuation finger so no finger is crowded out.
        keep = []
        for f in act_fingers:
            idx = torch.nonzero(fa == f)[:, 0]
            keep.append(idx[pre[idx].argsort()[:B // len(act_fingers)]])
        keep = torch.cat(keep)
        if len(keep) < B:  # top up with the best remaining seeds
            rest = torch.ones_like(pre, dtype=torch.bool)
            rest[keep] = False
            idx = torch.nonzero(rest)[:, 0]
            keep = torch.cat([keep, idx[pre[idx].argsort()[:B - len(keep)]]])
        pos, r6, q, fa, gcd = pos[keep].clone(), r6[keep].clone(), q[keep].clone(), fa[keep], gcd[keep]
        palm_w = (torch.rand(B, device=dev, generator=self.gen) < palm_frac).float()

        pos.requires_grad_(True)
        r6.requires_grad_(True)
        q.requires_grad_(True)
        opt = torch.optim.Adam([{"params": [pos], "lr": 1e-3},
                                {"params": [r6], "lr": 3e-3},
                                {"params": [q], "lr": 1e-2}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
        for it in range(steps):
            opt.zero_grad()
            e, E = self.energy(pos, r6, q, fa, palm_w, gcd)
            e.sum().backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                q.data = torch.max(torch.min(q.data, self.hand.hi), self.hand.lo)
            if log_every and (it % log_every == 0 or it == steps - 1):
                parts = " ".join(f"{k}={v.median().item():.2f}" for k, v in E.items())
                print(f"  step {it:4d} E_med={e.median().item():.2f} | {parts}", flush=True)
        with torch.no_grad():
            e, E, info = self.energy(pos, r6, q, fa, palm_w, gcd, detail=True)
        return dict(pos=pos.detach(), rot=rot6d_to_matrix(r6.detach()), q=q.detach(), fa=fa,
                    palm_w=palm_w, energy=e, terms={k: v for k, v in E.items()}, info=info)

    @staticmethod
    def gates(res):
        """Geometric proposal filter (sim decides the rest)."""
        i = res["info"]
        sup = ~F.one_hot(res["fa"], 4).bool()
        n_contact = ((i["c_sdf"] < 0.002) & sup).sum(-1)
        g = dict(
            act=(i["d_act"] < 0.005) & (i["cos_act"] > 0.8),
            pen=i["pen_nonact"] > -0.003,
            table=i["table"] > 0.0,
            approach=res["terms"]["approach"] < 2000,
            stroke=i["stroke"] > 0.008,
            self=i["self_pen"] < 0.002,
            order=i["finger_gap"] > 0.005,
            contacts=n_contact >= 2,
            preload=((i["depth"] > 0.012) & sup).sum(-1) >= 2,
        )
        ok = torch.ones_like(res["energy"], dtype=torch.bool)
        for v in g.values():
            ok &= v
        return ok, g


def to_grasp_vectors(res, idx):
    """Solver state -> 23-vectors (quat wxyz, pos, 16 joints names order)."""
    out = []
    for i in idx:
        R = res["rot"][i].cpu().numpy().astype(np.float64)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        xyzw = Rotation.from_matrix(R).as_quat()
        out.append(np.r_[xyzw[[3, 0, 1, 2]], res["pos"][i].cpu().numpy(), res["q"][i].cpu().numpy()])
    return out


def load_scene(mesh_dir, keypoints_path):
    """CoACD parts and the actuation keypoint with the functional object standing on the table
    (z = 0), as the primitive spawns it; this is also the grasp-file frame.

    mesh_dir is the scene's CoACD directory (mesh.glb + metadata.json)."""
    with open(os.path.join(mesh_dir, "metadata.json")) as f:
        lift = json.load(f)["bbox"][2] / 2.0         # canonical mesh is centered
    sc = trimesh.load(os.path.join(mesh_dir, "mesh.glb"))
    parts = sc.dump() if isinstance(sc, trimesh.Scene) else [sc]
    for p in parts:
        p.apply_translation([0.0, 0.0, lift])
    with open(keypoints_path) as f:
        ac = json.load(f)["act_contact"]
    return parts, np.asarray(ac["pos"]) + [0.0, 0.0, lift], np.asarray(ac["dir"])
