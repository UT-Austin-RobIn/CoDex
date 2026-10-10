"""
Batched differentiable grasp optimizer for dexterous hands using PyTorch.

This is CoDex's functional-grasp synthesis backend. It follows the FRoGGeR
formulation (Li et al., IROS 2023) but is a from-scratch GPU reimplementation
(thousands of candidate grasps optimised in parallel), extended with
actuation-finger constraints for functional grasping:
  - Optimises the min-weight metric l* for force-closure quality.
  - Surface contact equality constraints (fingertips on object surface).
  - Collision avoidance inequality constraints.
  - Contact normals from differentiable SDF gradients.
  - Grasp map G and wrench matrix W computed in PyTorch on GPU.
  - Min-weight LP solved per candidate; analytic gradient ∇l*.
  - One finger is reserved to reach the object's actuation point (trigger,
    button, ...); the remaining fingers + palm provide force closure.

Debug env vars: CODEX_GRASP_MULTI=1 enables multi-assignment actuation IK
(every candidate tries all four fingers and keeps the closest reach — grows
the pool but collapses it toward the thumb), CODEX_GRASP_NO_BASE=1 freezes
the base pose during the main optimisation, and CODEX_GRASP_DEBUG=1 dumps
per-stage grasp snapshots to out/grasp_debug/.
Defaults (all off) are the validated configuration.
"""

import torch
import torch.nn.functional as F
import pytorch_kinematics as pk
import numpy as np
import os
import trimesh
import open3d as o3d
from typing import Optional, List, Tuple
from scipy.optimize import linprog as scipy_linprog
import time

# LEAP finger link names: leap_rh_<prefix>_<suffix> (index, middle, ring, thumb).
FINGER_PREFIXES = ['if', 'mf', 'rf', 'th']
FINGER_LINK_SUFFIXES = [['bs', 'px', 'md', 'ds']] * 3 + [['mp', 'bs', 'px', 'ds']]

# Per-candidate fixed actuation-finger assignment (amap = candidate % 4) is
# the default: letting every candidate try all four fingers and keep the
# closest reach ("multi-assign", CODEX_GRASP_MULTI=1) collapses pools to the
# thumb — it wins the reach contest from almost any base pose — losing the
# index/middle-press grasps that suit top-button objects.
_DISABLE_MULTI_ASSIGN = not bool(int(os.environ.get("CODEX_GRASP_MULTI", "0")))
_DISABLE_BASE_UNFREEZE = bool(int(os.environ.get("CODEX_GRASP_NO_BASE", "0")))
_STAGE_DEBUG = bool(int(os.environ.get("CODEX_GRASP_DEBUG", "0")))

# Hand model assets (URDF + link meshes) live in the repo-level assets dir.
_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "models")
)


# ---------------------------------------------------------------------------
# SDF grid
# ---------------------------------------------------------------------------


class BatchedSDF:
    """Pre-computed signed-distance field on a regular 3-D grid."""

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        X_WO_matrix: np.ndarray,
        bounds_padding: float = 0.15,
        resolution: int = 128,
        device: str = "cuda",
    ):
        self.device = device
        self.resolution = resolution

        verts_O = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        R_WO = X_WO_matrix[:3, :3].astype(np.float64)
        t_WO = X_WO_matrix[:3, 3].astype(np.float64)
        verts_W = (R_WO @ verts_O.astype(np.float64).T).T + t_WO
        self._verts_W = verts_W  # store for OBB computation
        self._faces = np.asarray(mesh.faces)  # store for surface sampling

        self.bbox_min = np.min(verts_W, axis=0) - bounds_padding
        self.bbox_max = np.max(verts_W, axis=0) + bounds_padding

        print(f"  Computing {resolution}^3 SDF grid ...")
        lin = [
            np.linspace(self.bbox_min[i], self.bbox_max[i], resolution)
            for i in range(3)
        ]
        gx, gy, gz = np.meshgrid(*lin, indexing="ij")
        pts_W = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

        # Object-frame query via Open3D
        R_OW = R_WO.T
        pts_O = ((pts_W - t_WO) @ R_OW.T).astype(np.float32)
        mesh_o3d = o3d.t.geometry.TriangleMesh()
        mesh_o3d.vertex.positions = o3d.core.Tensor(verts_O)
        mesh_o3d.triangle.indices = o3d.core.Tensor(faces)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_o3d)
        sdf_vals = scene.compute_signed_distance(
            o3d.core.Tensor(pts_O, dtype=o3d.core.float32)
        ).numpy()
        print(f"  SDF range [{sdf_vals.min():.4f}, {sdf_vals.max():.4f}]")

        self.sdf_tensor = (
            torch.tensor(
                sdf_vals.reshape(resolution, resolution, resolution),
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )  # [1,1,D,H,W]
        # Keep the object-only field for physical contact tests. add_floor()
        # replaces sdf_tensor with a table-augmented field, but a palm near
        # the table is not necessarily touching the object.
        self.object_sdf_tensor = self.sdf_tensor
        self.bbox_min_t = torch.tensor(
            self.bbox_min, dtype=torch.float32, device=device
        )
        self.bbox_max_t = torch.tensor(
            self.bbox_max, dtype=torch.float32, device=device
        )
        self.range_t = self.bbox_max_t - self.bbox_min_t

        # Tight object bounding box (world frame, no padding)
        verts_f32 = verts_W.astype(np.float32)
        self.obj_bbox_min = torch.tensor(
            verts_f32.min(axis=0), dtype=torch.float32, device=device
        )
        self.obj_bbox_max = torch.tensor(
            verts_f32.max(axis=0), dtype=torch.float32, device=device
        )
        self.obj_center = (self.obj_bbox_min + self.obj_bbox_max) / 2

    def add_clearance_volume(self, center, direction, radius=0.015, height=0.05):
        """Store cylindrical no-go zone as SEPARATE SDF (for support fingers only).

        Actuation finger must be able to enter this region to reach the trigger.
        Support fingers should treat it as solid. Store center/direction/geometry
        so query_with_clearance() can add the clearance SDF on demand.

        `direction` is the PRESS direction (into the surface); the clearance
        cylinder extends the opposite way, outward from the surface, where the
        actuation finger approaches from.
        """
        self._clearance_center = torch.tensor(center, dtype=torch.float32, device=self.sdf_tensor.device)
        d = -np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        self._clearance_dir = torch.tensor(d, dtype=torch.float32, device=self.sdf_tensor.device)
        self._clearance_radius = float(radius)
        self._clearance_height = float(height)
        # Report voxel count for sanity (not actually modifying SDF)
        res = self.sdf_tensor.shape[2]
        lin = [np.linspace(self.bbox_min[i], self.bbox_max[i], res) for i in range(3)]
        gx, gy, gz = np.meshgrid(*lin, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
        delta = pts - np.asarray(center, dtype=np.float64)
        proj = delta @ d
        perp = np.linalg.norm(delta - np.outer(proj, d).reshape(-1, 3), axis=-1)
        inside = (proj >= 0) & (proj <= height) & (perp <= radius)
        print(f"  Actuation clearance (separate, support-only): "
              f"{inside.sum()} voxels (r={radius*1000:.0f}mm h={height*1000:.0f}mm)")

    def _clearance_sdf(self, points: torch.Tensor) -> torch.Tensor:
        """Signed distance to the clearance cylinder (positive outside, negative
        inside).  points [B,N,3] -> [B,N]. Returns +inf if no clearance defined.
        """
        if not hasattr(self, '_clearance_center'):
            return torch.full(points.shape[:-1], float('inf'), device=points.device)
        delta = points - self._clearance_center
        proj = (delta * self._clearance_dir).sum(-1)
        perp_vec = delta - proj.unsqueeze(-1) * self._clearance_dir
        perp = perp_vec.norm(dim=-1)
        # Finite cylinder SDF approximation.
        return torch.maximum(perp - self._clearance_radius,
                             torch.maximum(-proj, proj - self._clearance_height))

    def add_floor(self, z_min):
        """Set SDF negative below z_min (table surface). Baked into SDF grid directly
        so all collision queries see the floor as solid. Also stored separately for
        viser's floor plane drawing."""
        self._floor_z = float(z_min)
        res = self.sdf_tensor.shape[2]
        lin_z = np.linspace(self.bbox_min[2], self.bbox_max[2], res)
        sdf_np = self.sdf_tensor[0, 0].cpu().numpy().copy()
        for zi, z in enumerate(lin_z):
            if z < z_min:
                sdf_np[:, :, zi] = np.minimum(sdf_np[:, :, zi], -0.01)
        self.sdf_tensor = torch.tensor(
            sdf_np, dtype=torch.float32, device=self.sdf_tensor.device
        ).unsqueeze(0).unsqueeze(0)
        print(f"  Floor at z={z_min*1000:.0f}mm (baked into SDF)")

    def _floor_sdf(self, points: torch.Tensor) -> torch.Tensor:
        """Signed distance to floor (z=floor_z). Positive above, negative below.
        Used only for separated reporting (e.g., viser). Not used in query()."""
        if not hasattr(self, '_floor_z'):
            return torch.full(points.shape[:-1], float('inf'), device=points.device)
        return points[..., 2] - self._floor_z

    def query(self, points: torch.Tensor, include_clearance: bool = True,
              include_floor: bool = True) -> torch.Tensor:
        """Differentiable SDF look-up.  points [B,N,3] -> [B,N].

        include_clearance=True (default): merges the actuation clearance zone
        into the SDF so support fingers treat it as solid.
        The default field includes the baked-in floor. Set include_floor=False
        to query the preserved object-only field for contact certification.
        """
        B, N, _ = points.shape
        norm = 2.0 * (points - self.bbox_min_t) / self.range_t - 1.0
        # grid_sample expects (w, h, d) = (Z, Y, X) for tensor stored as [D=X, H=Y, W=Z]
        norm_grid = norm[..., [2, 1, 0]]
        field = self.sdf_tensor if include_floor else self.object_sdf_tensor
        out = F.grid_sample(
            field.expand(B, -1, -1, -1, -1),
            norm_grid.view(B, N, 1, 1, 3),
            align_corners=True,
            padding_mode="border",
        )
        obj_sdf = out.view(B, N)
        if include_clearance and hasattr(self, '_clearance_center'):
            cl = self._clearance_sdf(points)
            return torch.minimum(obj_sdf, cl)
        return obj_sdf


    def query_with_normals(self, points: torch.Tensor):
        """SDF look-up returning (sdf_values [B,N], inward_normals [B,N,3]).

        Inward normals = -∇s / |∇s| (pointing INTO the object).
        Works both inside and outside torch.no_grad() context.
        """
        needs_grad = torch.is_grad_enabled() or points.requires_grad

        if needs_grad:
            pts = points.detach().requires_grad_(True)
            sdf = self.query(pts)
            grad_outputs = torch.ones_like(sdf)
            (grad,) = torch.autograd.grad(sdf, pts, grad_outputs, create_graph=False)
        else:
            # Finite differences for gradient when autograd is not available
            eps = 1e-4
            grad = torch.zeros_like(points)
            for d in range(3):
                pts_p = points.clone()
                pts_m = points.clone()
                pts_p[..., d] += eps
                pts_m[..., d] -= eps
                grad[..., d] = (self.query(pts_p) - self.query(pts_m)) / (2 * eps)
            sdf = self.query(points)

        norms = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        inward = -grad / norms
        return sdf, inward


def split_force_closure_contacts(support_points, palm_point, palm_gap,
                                 env_mask, contact_tolerance=0.008):
    """Group candidates by real object contact before computing FC wrenches.

    Different groups have different numbers of contacts; padding an absent
    palm with zero wrenches would change the min-weight LP's constraints.
    """
    if palm_point is None:
        return [(env_mask, support_points, False)]
    palm_valid = palm_gap.abs() < contact_tolerance
    return [
        (env_mask & ~palm_valid, support_points, False),
        (env_mask & palm_valid,
         torch.cat((support_points, palm_point.unsqueeze(1)), dim=1), True),
    ]


# HiGHS uses implicit feasibility tolerances in this LP call (SciPy 1.12:
# `highs-ds`, `presolve=False`, no explicit primal/dual tolerance). A tiny
# positive l* can therefore be numerical zero rather than a useful FC margin.
FC_LSTAR_MIN_MARGIN = 1e-6

# Center of the exposed contact-facing (-Z) face of LEAP palm collision box 7.
LEAP_PALM_CONTACT_OFFSET = (-0.048, -0.033, -0.0345)


def palm_reference_in_base(palm_fk_matrix, palm_offset):
    """Map the FC palm-link reference into the hand base frame."""
    off_h = torch.cat((palm_offset, palm_offset.new_ones(1)))
    return (palm_fk_matrix @ off_h)[..., :3]


def force_closure_margin_passes(lstar: torch.Tensor) -> torch.Tensor:
    """Authoritative final LP acceptance, above observed 1e-8-level noise."""
    return lstar > FC_LSTAR_MIN_MARGIN


# ---------------------------------------------------------------------------
# Grasp wrench space helpers (FRoGGeR formulation)
# ---------------------------------------------------------------------------

def box_sdf_batch(query_pts, box_centers, box_rotations, box_half_extents):
    """Signed distance from query points to oriented bounding boxes.

    Uses the Quilez SDF formula: transform to box local frame, then
    compute AABB distance analytically.

    Args:
        query_pts: [B, N, 3] points to query
        box_centers: [B, 3] box centers in world frame
        box_rotations: [B, 3, 3] box rotation matrices (world frame)
        box_half_extents: [B, 3] half-extents (hx, hy, hz)

    Returns:
        sdf: [B, N] signed distance (negative = inside box)
    """
    # Transform query points to box local frame
    # q = R^T @ (p - c)
    local = query_pts - box_centers.unsqueeze(1)  # [B, N, 3]
    # Batched matmul: [B, 3, 3]^T @ [B, N, 3]^T -> [B, 3, N] -> [B, N, 3]
    q = torch.bmm(box_rotations.transpose(1, 2), local.transpose(1, 2)).transpose(1, 2)

    # Quilez SDF for AABB: d = |q| - h; sdf = length(max(d,0)) + min(max(d.x,d.y,d.z),0)
    d = q.abs() - box_half_extents.unsqueeze(1)  # [B, N, 3]
    outside = torch.clamp(d, min=0)  # [B, N, 3]
    outside_dist = outside.norm(dim=-1)  # [B, N]
    inside_dist = d.max(dim=-1).values.clamp(max=0)  # [B, N] (negative)
    return outside_dist + inside_dist  # [B, N]


def box_box_sdf_batch(centers_A, rots_A, half_A, centers_B, rots_B, half_B):
    """Analytic signed distance between two OBBs via the Separating Axis Theorem.

    For each of 15 candidate axes (3 face normals of A, 3 of B, 9 edge-edge
    cross-products), compute:
        sign_sep(n) = |(c_B - c_A) · n|  -  h_A_proj(n)  -  h_B_proj(n)
    SDF = max over axes of sign_sep(n).
      - Positive: boxes separated; the axis witnessing the tightest gap wins.
      - Negative: boxes overlap; the axis witnessing the shallowest overlap wins
        (i.e., the negative penetration depth along the easiest-to-separate axis).
    Differentiable w.r.t. centers, rotations, half-extents.

    Args:
        centers_A, centers_B: [B, 3]
        rots_A, rots_B: [B, 3, 3] — columns are box axes
        half_A, half_B: [B, 3]

    Returns:
        sdf: [B] signed distance (negative = overlapping).
    """
    B = centers_A.shape[0]
    dev = centers_A.device
    a_axes = [rots_A[:, :, i] for i in range(3)]
    b_axes = [rots_B[:, :, i] for i in range(3)]
    delta = centers_B - centers_A

    def sep_on_axis(n, valid_mask=None):
        # sign_sep(n) = |delta · n| - half-extent projections onto n.
        proj_A = (half_A * torch.stack([(a_axes[i] * n).sum(-1).abs() for i in range(3)], dim=-1)).sum(-1)
        proj_B = (half_B * torch.stack([(b_axes[i] * n).sum(-1).abs() for i in range(3)], dim=-1)).sum(-1)
        center_dist = (delta * n).sum(-1).abs()
        sep = center_dist - proj_A - proj_B
        if valid_mask is not None:
            # Invalidate axes where the cross product was ~0 (parallel edges).
            # Set their sep to a large negative so they never win the max.
            sep = torch.where(valid_mask, sep, torch.full_like(sep, -1e9))
        return sep

    # Face normals (always valid).
    sdf = sep_on_axis(a_axes[0])
    for ax in a_axes[1:] + b_axes:
        sdf = torch.maximum(sdf, sep_on_axis(ax))

    # Edge-edge cross products (mask out zero-length axes from parallel edges).
    eps_parallel = 1e-6
    for i in range(3):
        for j in range(3):
            c = torch.cross(a_axes[i], b_axes[j], dim=-1)
            cn = c.norm(dim=-1, keepdim=True)
            valid = (cn.squeeze(-1) > eps_parallel)
            n = c / cn.clamp(min=eps_parallel)
            sdf = torch.maximum(sdf, sep_on_axis(n, valid))
    return sdf


def compute_primitive_forces_torch(ns: int, mu: float, device="cuda"):
    """Pyramidal friction cone approximation.  Returns F [3, ns]."""
    nums = torch.arange(ns, dtype=torch.float32, device=device)
    scale = 1.0 / (1.0 + mu ** 2) ** 0.5
    fx = mu * scale * torch.cos(2 * np.pi * nums / ns)
    fy = mu * scale * torch.sin(2 * np.pi * nums / ns)
    fz = scale * torch.ones(ns, device=device)
    return torch.stack([fx, fy, fz], dim=0)  # [3, ns]


def compute_contact_frames(positions, inward_normals):
    """Build contact-to-object-frame transforms from positions and normals.

    Args:
        positions: [B, nc, 3] contact positions in object frame
        inward_normals: [B, nc, 3] inward-pointing normals in object frame

    Returns:
        g_OCs: [B, nc, 4, 4] homogeneous transforms
    """
    B, nc, _ = positions.shape
    dev = positions.device
    n = inward_normals  # [B, nc, 3]

    # Build tangent vectors via cross product with arbitrary axis
    zeta = torch.tensor([1.2, 2.3, 3.4], device=dev)
    zeta_p = torch.tensor([3.4, 2.3, 1.2], device=dev)

    # tx = zeta - (n·zeta)*n, handle degenerate case
    # Check if normal is too close to zeta
    diff = (n - zeta.view(1, 1, 3)).norm(dim=-1)  # [B, nc]
    arb = zeta.view(1, 1, 3).expand(B, nc, 3).clone()
    arb[diff <= 1e-6] = zeta_p

    dot_nt = (n * arb).sum(-1, keepdim=True)  # [B, nc, 1]
    tx = arb - n * dot_nt
    tx = tx / tx.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ty = torch.cross(n, tx, dim=-1)
    ty = ty / ty.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # g_OC: rotation columns = [tx, ty, n], translation = position
    R = torch.stack([tx, ty, n], dim=-1)  # [B, nc, 3, 3]
    g = torch.zeros(B, nc, 4, 4, device=dev)
    g[:, :, :3, :3] = R
    g[:, :, :3, 3] = positions
    g[:, :, 3, 3] = 1.0
    return g


def compute_grasp_matrix_torch(g_OCs):
    """Compute grasp matrix G from contact frames.

    Args:
        g_OCs: [B, nc, 4, 4] contact frame transforms

    Returns:
        G: [B, 6, nc*3] grasp matrix (hard contact model)
    """
    B, nc = g_OCs.shape[:2]
    dev = g_OCs.device

    # Following the original FRoGGeR (grasping.py):
    # 1. g_inv = g_OC^{-1} = [[R^T, -R^T p], [0, 1]]
    # 2. Ad(g_inv) = [[R^T, [(-R^T p)]× @ R^T], [0, R^T]]
    #    where [a]× is the skew-symmetric matrix of a
    # 3. G_i = Ad(g_inv)^T @ Bc
    R = g_OCs[:, :, :3, :3]  # [B, nc, 3, 3]
    p = g_OCs[:, :, :3, 3]   # [B, nc, 3]
    Rt = R.transpose(-1, -2)  # [B, nc, 3, 3]

    # p_inv = -R^T @ p (translation of g_inv)
    p_inv = -(Rt @ p.unsqueeze(-1)).squeeze(-1)  # [B, nc, 3]

    # wedge(p_inv) = skew-symmetric matrix of p_inv
    a1, a2, a3 = p_inv[..., 0], p_inv[..., 1], p_inv[..., 2]
    skew_p = torch.zeros(B, nc, 3, 3, device=dev)
    skew_p[..., 0, 1] = -a3; skew_p[..., 0, 2] = a2
    skew_p[..., 1, 0] = a3;  skew_p[..., 1, 2] = -a1
    skew_p[..., 2, 0] = -a2; skew_p[..., 2, 1] = a1

    # Ad(g_inv) = [[Rt, skew_p @ Rt], [0, Rt]]
    Ad_ginv = torch.zeros(B, nc, 6, 6, device=dev)
    Ad_ginv[:, :, :3, :3] = Rt
    Ad_ginv[:, :, 3:, 3:] = Rt
    Ad_ginv[:, :, :3, 3:] = skew_p @ Rt

    # Ad(g_inv)^T
    Ad_ginv_T = Ad_ginv.transpose(-1, -2)

    # Hard contact basis: Bc [6, 3] = [[I_3], [0_3]]
    Bc = torch.zeros(6, 3, device=dev)
    Bc[:3, :3] = torch.eye(3, device=dev)

    # G_i = Ad_ginv_T @ Bc → [B, nc, 6, 3]
    G_parts = Ad_ginv_T @ Bc.unsqueeze(0).unsqueeze(0)

    # Stack: G = [G_1, G_2, ..., G_nc] → [B, 6, nc*3]
    G = G_parts.permute(0, 2, 1, 3).reshape(B, 6, nc * 3)
    return G


def compute_wrench_matrix(G, F_prim, nc, ns):
    """Compute wrench matrix W = G @ kron(I_nc, F).

    Args:
        G: [B, 6, nc*3] grasp matrix
        F_prim: [3, ns] primitive forces
        nc: number of contacts
        ns: number of friction cone sides

    Returns:
        W: [B, 6, nc*ns] wrench matrix
    """
    B = G.shape[0]
    dev = G.device
    # kron(I_nc, F) is block diagonal: [nc*3, nc*ns]
    kron = torch.zeros(nc * 3, nc * ns, device=dev)
    for i in range(nc):
        kron[i*3:(i+1)*3, i*ns:(i+1)*ns] = F_prim
    W = G @ kron  # [B, 6, nc*ns]
    return W


def solve_min_weight_lp_batch(W_batch_np, env_mask=None, return_status=False):
    """Solve min-weight LP for a batch of wrench matrices.

    For each W [6, m], solves:
        max l  s.t. W @ alpha = 0, 1^T alpha = 1, alpha >= l

    Args:
        W_batch_np: numpy array [B, 6, m]
        env_mask: optional boolean array [B] — only solve LP for True entries.
                  If None, solve for all environments.

    Returns:
        l_stars: [B] optimal min-weight values
        alphas: [B, m] optimal weights
        lambs: [B, m] dual variables for inequality constraints
        nus: [B, 7] dual variables for equality constraints
        status: optional [B] boolean LP success mask when return_status=True
    """
    B, _, m = W_batch_np.shape
    l_stars = np.full(B, -1.0)
    alphas = np.zeros((B, m))
    lambs = np.zeros((B, m))
    nus = np.zeros((B, 7))
    status = np.zeros(B, dtype=bool)

    # LP in SciPy's inequality/equality form (SciPy minimises). The FRoGGeR
    # alpha and l variables are free; SciPy otherwise defaults each to >= 0.
    # min -l  s.t. W @ alpha = 0, 1^T alpha = 1, alpha - l >= 0
    # Decision variable: x = [alpha (m), l (1)]
    c = np.zeros(m + 1)
    c[-1] = -1.0  # minimise -l = maximise l

    # Inequality: -alpha_i + l <= 0  ↔  alpha_i >= l
    A_ub = np.zeros((m, m + 1))
    A_ub[:, :m] = -np.eye(m)
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(m)
    b_eq = np.array([0.,0.,0.,0.,0.,0.,1.])

    # Determine which envs to solve
    if env_mask is not None:
        solve_indices = np.where(env_mask)[0]
    else:
        solve_indices = np.arange(B)

    for b in solve_indices:
        W = W_batch_np[b]
        A_eq = np.zeros((7, m + 1))
        A_eq[:6, :m] = W
        A_eq[6, :m] = 1.0

        try:
            res = scipy_linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                                bounds=(None, None), method='highs-ds',
                                options={'presolve': False})
            if res.success:
                status[b] = True
                l_stars[b] = res.x[-1]
                alphas[b] = res.x[:m]
                # Extract dual variables for KKT gradient computation
                lambs[b] = -res.ineqlin.marginals  # sign convention: scipy uses negative
                nus[b] = -res.eqlin.marginals
        except Exception:
            pass

    if return_status:
        return l_stars, alphas, lambs, nus, status
    return l_stars, alphas, lambs, nus


def min_weight_gradient_batch(W_np, l_stars, alphas, lambs, nus,
                              device="cuda", valid_mask=None):
    """Compute ∂l*/∂W for a batch using the KKT sensitivity exploit (Prop. 1).

    Follows the original FRoGGeR implementation in metrics.py exactly:
    the gradient dl/dW is computed by solving C @ dz = -dH/dW for the last
    component of dz (which corresponds to dl), where C encodes the KKT system.

    Args:
        W_np: [B, 6, m] wrench matrices
        l_stars: [B] optimal min-weight values
        alphas: [B, m] primal LP solutions
        lambs: [B, m] dual variables for inequality constraints
        nus: [B, 7] dual variables for equality constraints
        valid_mask: [B] boolean LP success mask; required so a valid negative
                    margin is not confused with a failed solve

    Returns:
        dl_dW: [B, 6, m] gradient of l* w.r.t. W elements
    """
    B = W_np.shape[0]
    m = W_np.shape[2]
    dl_dW = np.zeros((B, 6, m), dtype=np.float32)
    if valid_mask is None:
        raise ValueError("LP success mask required for min-weight gradient")

    for b in range(B):
        if not valid_mask[b]:
            continue

        W = W_np[b]            # [6, m]
        alpha = alphas[b]      # [m]
        nu = nus[b]            # [7]
        lamb = lambs[b]        # [m]

        Im = np.eye(m)
        ones_m = np.ones((m, 1))

        # Constraint matrices in the original (non-standard) LP form:
        #   Ain = [-I_m | 1_m]  (m x m+1)   -->  -alpha_i + l <= 0
        #   Aeq = [[W | 0], [1^T | 0]]  (7 x m+1)
        Ain = np.hstack((-Im, ones_m))
        Aeq = np.zeros((7, m + 1))
        Aeq[:6, :m] = W
        Aeq[6, :m] = 1.0

        # C matrix from KKT: C = [[diag(lamb) @ Ain], [Aeq]]
        # Shape: (m + 7) x (m + 1)
        C = np.vstack((Ain * lamb[:, None], Aeq))

        # dH/dW: the KKT conditions depend on W only through the equality
        # constraint H3: Aeq @ x - beq = 0.  Specifically H3[:6] = W @ alpha.
        # For the full KKT vector (H2 complementarity, H3 stationarity):
        #   - H2 block (m rows): diag(lamb) @ (Ain @ x - bin) -- no W dependence
        #   - H3 block (7 rows): Aeq @ x - beq -- depends on W through W @ alpha
        #
        # dH3_i/dW_{ij} = alpha_j (for i<6), so the Jacobian tensor DH_W has
        # nonzero only in the H3 block (bottom 7 rows of the 2m+7 system).
        # However, our C is (m+7) x (m+1) [H2 has m rows, H3 has 7 rows].
        #
        # Following metrics.py: build the RHS matrix for the bottom block only,
        # then solve C @ Dl_W_row = -RHS for the last row of the solution.

        # RHS for the H3 block: DH3/dW has shape (7, 6*m)
        # DH3[i, i*m+j] = alpha[j] for i<6, row 6 is zero
        DH3_W = np.zeros((7, 6 * m))
        for i in range(6):
            DH3_W[i, i * m: i * m + m] = alpha

        # Full RHS: top m rows are zero (H2 has no W dependence)
        RHS = np.vstack((np.zeros((m, 6 * m)), DH3_W))  # (m+7, 6m)

        try:
            # Solve C @ dz = -RHS for dz, extract last row (dl component)
            # C is (m+7) x (m+1), overdetermined -> lstsq
            sol = np.linalg.lstsq(C, -RHS, rcond=None)[0]  # (m+1, 6m)
            dl_dW[b] = sol[-1].reshape(6, m)  # last row = dl/dW
        except Exception:
            pass

    return torch.tensor(dl_dW, dtype=torch.float32, device=device)


def compute_fc_proxy_loss(tip_positions, tip_normals, object_center):
    """Differentiable force-closure proxy loss.

    Encourages opposing contact normals and contact spread around the object.

    Args:
        tip_positions: [B, nc, 3] fingertip positions
        tip_normals: [B, nc, 3] inward-pointing normals
        object_center: [3] object center

    Returns:
        loss: [B] per-environment loss (lower = better force closure)
    """
    B, nc, _ = tip_positions.shape
    dev = tip_positions.device

    # 1) Normal opposition: penalise parallel contact normals
    dots = torch.bmm(tip_normals, tip_normals.transpose(1, 2))  # [B, nc, nc]
    triu_mask = torch.triu(torch.ones(nc, nc, device=dev), diagonal=1).bool()
    pair_dots = dots[:, triu_mask]  # [B, nc*(nc-1)/2]
    L_oppose = F.relu(pair_dots + 0.3).mean(-1)  # want dot < -0.3

    # 2) Contact spread: fingers should surround the object
    r = tip_positions - object_center.unsqueeze(0).unsqueeze(0)  # [B, nc, 3]
    r_norm = F.normalize(r, dim=-1)
    r_dots = torch.bmm(r_norm, r_norm.transpose(1, 2))  # [B, nc, nc]
    r_pair_dots = r_dots[:, triu_mask]
    # Want diverse approach directions (negative dots = opposing sides)
    L_spread = r_pair_dots.mean(-1)  # smaller = more spread

    return 2.0 * L_oppose + 1.0 * L_spread


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# -- Allegro hand ----------------------------------------------------------
_JOINT_LOWER = np.array(
    [
        -0.47, -0.196, -0.174, -0.227,
        -0.47, -0.196, -0.174, -0.227,
        -0.47, -0.196, -0.174, -0.227,
        0.263, -0.105, -0.189, -0.162,
    ],
    dtype=np.float32,
)
_JOINT_UPPER = np.array(
    [
        0.47, 1.61, 1.709, 1.618,
        0.47, 1.61, 1.709, 1.618,
        0.47, 1.61, 1.709, 1.618,
        1.396, 1.163, 1.644, 1.719,
    ],
    dtype=np.float32,
)

# -- LEAP hand (PK joint order: axl, mcp, pip, dip) ------------------------
_LEAP_JOINT_LOWER = np.array(
    [
        -0.314, -1.047, -0.506, -0.366,
        -0.314, -1.047, -0.506, -0.366,
        -0.314, -1.047, -0.506, -0.366,
        -0.349, -0.47, -1.2, -1.34,
    ],
    dtype=np.float32,
)
_LEAP_JOINT_UPPER = np.array(
    [
        2.23, 1.047, 1.885, 2.042,
        2.23, 1.047, 1.885, 2.042,
        2.23, 1.047, 1.885, 2.042,
        2.094, 2.443, 1.9, 1.88,
    ],
    dtype=np.float32,
)


def _link_names(hand, hand_type="allegro"):
    if hand_type == "leap":
        # LEAP URDFs always use 'rh' prefix in link names, regardless of hand variant
        h = "rh"
        tip = [f"leap_{h}_if_ds", f"leap_{h}_mf_ds", f"leap_{h}_rf_ds", f"leap_{h}_th_ds"]
        col = [
            f"leap_{h}_palm",
            f"leap_{h}_if_bs", f"leap_{h}_if_px", f"leap_{h}_if_md", f"leap_{h}_if_ds",
            f"leap_{h}_mf_bs", f"leap_{h}_mf_px", f"leap_{h}_mf_md", f"leap_{h}_mf_ds",
            f"leap_{h}_rf_bs", f"leap_{h}_rf_px", f"leap_{h}_rf_md", f"leap_{h}_rf_ds",
            f"leap_{h}_th_mp", f"leap_{h}_th_bs", f"leap_{h}_th_px", f"leap_{h}_th_ds",
        ]
        return tip, col
    # Allegro
    h = hand
    tip = [f"algr_{h}_if_ds", f"algr_{h}_mf_ds", f"algr_{h}_rf_ds", f"algr_{h}_th_ds"]
    col = [
        f"algr_{h}_palm",
        f"algr_{h}_if_bs", f"algr_{h}_if_px", f"algr_{h}_if_md", f"algr_{h}_if_ds",
        f"algr_{h}_mf_bs", f"algr_{h}_mf_px", f"algr_{h}_mf_md", f"algr_{h}_mf_ds",
        f"algr_{h}_rf_bs", f"algr_{h}_rf_px", f"algr_{h}_rf_md", f"algr_{h}_rf_ds",
        f"algr_{h}_th_mp", f"algr_{h}_th_bs", f"algr_{h}_th_px", f"algr_{h}_th_ds",
    ]
    return tip, col


# Visual mesh files per link (relative to models/<hand_type_dir>/)
def _visual_meshes(hand, hand_type="allegro"):
    if hand_type == "leap":
        return _leap_visual_meshes(hand)
    return _allegro_visual_meshes(hand)


def _allegro_visual_meshes(hand):
    h = hand
    base_mesh = "meshes/base_link_left.obj" if h == "lh" else "meshes/base_link_right.obj"
    base_pose = [0, 0, 0.095, -np.pi/2, 0, 0] if h == "lh" else [0, 0, 0.095, 0, 0, 0]
    th_mp_mesh = f"meshes/link_12.0_{'left' if h == 'lh' else 'right'}.obj"
    th_mp_pose = [0, 0, 0, np.pi, 0, 0] if h == "lh" else None
    finger_tip_pose = [0, 0, 0.0267, 0, 0, 0]
    thumb_tip_pose = [0, 0, 0.0423, 0, 0, 0]
    return {
        f"algr_{h}_palm":  [(base_mesh, base_pose)],
        f"algr_{h}_if_bs": [("meshes/link_0.0.obj", None)],
        f"algr_{h}_if_px": [("meshes/link_1.0.obj", None)],
        f"algr_{h}_if_md": [("meshes/link_2.0.obj", None)],
        f"algr_{h}_if_ds": [("meshes/link_3.0.obj", None), ("meshes/link_3.0_tip.obj", finger_tip_pose)],
        f"algr_{h}_mf_bs": [("meshes/link_0.0.obj", None)],
        f"algr_{h}_mf_px": [("meshes/link_1.0.obj", None)],
        f"algr_{h}_mf_md": [("meshes/link_2.0.obj", None)],
        f"algr_{h}_mf_ds": [("meshes/link_3.0.obj", None), ("meshes/link_3.0_tip.obj", finger_tip_pose)],
        f"algr_{h}_rf_bs": [("meshes/link_0.0.obj", None)],
        f"algr_{h}_rf_px": [("meshes/link_1.0.obj", None)],
        f"algr_{h}_rf_md": [("meshes/link_2.0.obj", None)],
        f"algr_{h}_rf_ds": [("meshes/link_3.0.obj", None), ("meshes/link_3.0_tip.obj", finger_tip_pose)],
        f"algr_{h}_th_mp": [(th_mp_mesh, th_mp_pose)],
        f"algr_{h}_th_bs": [("meshes/link_13.0.obj", None)],
        f"algr_{h}_th_px": [("meshes/link_14.0.obj", None)],
        f"algr_{h}_th_ds": [("meshes/link_15.0.obj", None), ("meshes/link_15.0_tip.obj", thumb_tip_pose)],
    }


def _leap_visual_meshes(hand):
    # LEAP URDFs always use 'rh' prefix in link names
    h = "rh"
    P = np.pi
    H = np.pi / 2
    # Visual origins differ between LH and RH for palm and th_mp
    if hand == "lh":
        palm_mesh = "meshes_obj/palm_lower_left.obj"
        palm_pose = [-0.0956, -0.1170, 0.0208, H, 0, P]
        th_mp_mesh = "meshes_obj/thumb_left_temp_base.obj"
        th_mp_pose = [0.0439, 0.0068, 0.0215, H, 0, 0]
    else:
        palm_mesh = "meshes_obj/palm_lower.obj"
        palm_pose = [-0.0201, 0.0258, -0.0347, 0, 0, 0]
        th_mp_mesh = "meshes_obj/pip.obj"
        th_mp_pose = [-0.0054, 0.0003, 0.0008, -H, -H, 0]
    bs_pose = [0.0084, 0.0078, 0.0147, 0, 0, 0]
    px_pose = [0.0096, 0.0003, 0.0008, -H, -H, 0]
    md_pose = [0.0211, -0.0084, 0.0098, -P, 0, 0]
    ds_pose = [0.0133, -0.0061, 0.0145, P, 0, 0]
    th_bs_pose = [0.0120, 0, -0.0159, H, 0, 0]
    th_px_pose = [0.0440, 0.0580, -0.0086, 0, 0, 0]
    th_ds_pose = [0.0626, 0.0785, 0.0490, 0, 0, 0]
    return {
        f"leap_{h}_palm":  [(palm_mesh, palm_pose)],
        f"leap_{h}_if_bs": [("meshes_obj/mcp_joint.obj", bs_pose)],
        f"leap_{h}_if_px": [("meshes_obj/pip.obj", px_pose)],
        f"leap_{h}_if_md": [("meshes_obj/dip.obj", md_pose)],
        f"leap_{h}_if_ds": [("meshes_obj/fingertip.obj", ds_pose)],
        f"leap_{h}_mf_bs": [("meshes_obj/mcp_joint.obj", bs_pose)],
        f"leap_{h}_mf_px": [("meshes_obj/pip.obj", px_pose)],
        f"leap_{h}_mf_md": [("meshes_obj/dip.obj", md_pose)],
        f"leap_{h}_mf_ds": [("meshes_obj/fingertip.obj", ds_pose)],
        f"leap_{h}_rf_bs": [("meshes_obj/mcp_joint.obj", bs_pose)],
        f"leap_{h}_rf_px": [("meshes_obj/pip.obj", px_pose)],
        f"leap_{h}_rf_md": [("meshes_obj/dip.obj", md_pose)],
        f"leap_{h}_rf_ds": [("meshes_obj/fingertip.obj", ds_pose)],
        f"leap_{h}_th_mp": [(th_mp_mesh, th_mp_pose)],
        f"leap_{h}_th_bs": [("meshes_obj/thumb_pip.obj", th_bs_pose)],
        f"leap_{h}_th_px": [("meshes_obj/thumb_dip.obj", th_px_pose)],
        f"leap_{h}_th_ds": [("meshes_obj/thumb_fingertip.obj", th_ds_pose)],
    }


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
class BatchedGraspOptimizer:
    """GPU-batched differentiable grasp optimiser for dexterous hands."""

    def __init__(
        self,
        sdf: BatchedSDF,
        num_envs: int = 16000,
        device: str = "cuda",
        hand: str = "lh",
        hand_type: str = "allegro",
        palm_contact: bool = False,
    ):
        self.num_envs = num_envs
        self.device = device
        self.sdf = sdf
        self.hand = hand
        self.hand_type = hand_type
        self.palm_contact = palm_contact
        self.tip_link_names, self.collision_link_names = _link_names(hand, hand_type)

        # Fingertip pad center in link frame: -X face center of the main
        # pad box (the small box at the fingertip end, not the side walls).
        # Pad faces -X in tip link frame.
        if hand_type == "leap":
            f_off = [-0.010, -0.032, 0.015]   # IF/MF/RF: pad center, shifted toward fingertip
            t_off = [-0.009, -0.035, -0.011]  # TH: pad center, shifted toward fingertip
            palm_off = LEAP_PALM_CONTACT_OFFSET
            self.palm_link = f"leap_{hand}_palm"
        else:
            th = np.pi / 4.0
            r = 0.012
            f_off = [r * np.sin(th), 0.0, 0.0267 + r * np.cos(th)]
            t_off = [r * np.sin(th), 0.0, 0.0423 + r * np.cos(th)]
            palm_off = [0.0, 0.0, 0.0]  # Allegro palm center
            self.palm_link = f"algr_{hand}_palm"
        offsets = [f_off, f_off, f_off, t_off]
        if palm_contact:
            # 4 palm contact points spread across inner surface for area contact.
            # Single point misses torque contribution. 4 spread points capture:
            # - Line contact torque (cylinder) via vertical spread
            # - Area contact torque (flat) via both vertical + horizontal spread
            # Palm inner surface: x≈-0.03, y spans -0.06 to 0.02, z≈-0.01
            palm_pts = [
                [-0.030, -0.050, -0.010],  # bottom-left
                [-0.030,  0.010, -0.010],  # top-left
                [-0.060, -0.050, -0.010],  # bottom-right
                [-0.060,  0.010, -0.010],  # top-right
            ]
            for pp in palm_pts:
                offsets.append(pp)
        self.tip_offsets = torch.tensor(offsets, dtype=torch.float32, device=device)
        self.palm_offset = torch.tensor(palm_off, dtype=torch.float32, device=device)

        # Fingertip pad sample points: center + ring of 6 around the tip offset.
        # Used for geometry-to-object surface loss (min SDF across pad).
        # The pad is in the yz-plane of the link frame (perpendicular to x-axis).
        pad_r = 0.008  # 8mm pad radius
        pad_samples = []
        for base_off in [f_off, f_off, f_off, t_off]:
            pts = [list(base_off)]  # center
            for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                pt = list(base_off)
                pt[1] += pad_r * np.cos(angle)
                pt[2] += pad_r * np.sin(angle)
                pts.append(pt)
            pad_samples.append(pts)
        if palm_contact:
            # Palm: single point (contact handled by palm proximity loss)
            pad_samples.append([palm_off])
        # Store as list of tensors (different sizes: 7 for fingers, 1 for palm)
        self.pad_offsets = [
            torch.tensor(pts, dtype=torch.float32, device=device) for pts in pad_samples
        ]

        # FK chain (URDF for correct joint-axis frames)
        if hand_type == "leap":
            urdf_path = os.path.join(
                _MODELS_DIR, f"leap_{hand}/leap_solver.urdf",
            )
        else:
            urdf_path = os.path.join(
                _MODELS_DIR, f"allegro/allegro_{hand}.urdf",
            )
        with open(urdf_path) as fh:
            self.chain = pk.build_chain_from_urdf(fh.read()).to(device=device)
        assert len(self.chain.get_joint_parameter_names()) == 16

        if hand_type == "leap":
            self.q_lo = torch.tensor(_LEAP_JOINT_LOWER, dtype=torch.float32, device=device)
            self.q_hi = torch.tensor(_LEAP_JOINT_UPPER, dtype=torch.float32, device=device)
        else:
            self.q_lo = torch.tensor(_JOINT_LOWER, dtype=torch.float32, device=device)
            self.q_hi = torch.tensor(_JOINT_UPPER, dtype=torch.float32, device=device)
        self.q_range = self.q_hi - self.q_lo

        # Precompute dense collision-check points on each link mesh
        self._precompute_collision_points()

    # -- parameterisation -------------------------------------------------
    def _u2q(self, u):
        return self.q_lo + torch.sigmoid(u) * self.q_range

    def _q2u(self, q):
        n = ((q - self.q_lo) / self.q_range).clamp(0.01, 0.99)
        return torch.log(n / (1.0 - n))

    @staticmethod
    def _rot6d_to_matrix(r6d):
        """6D rotation -> 3x3 matrix via Gram-Schmidt."""
        a1, a2 = r6d[:, :3], r6d[:, 3:]
        b1 = F.normalize(a1, dim=-1)
        dot = (b1 * a2).sum(-1, keepdim=True)
        b2 = F.normalize(a2 - dot * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)  # [B,3,3] columns

    def _base_T(self, pos, r6d):
        B = pos.shape[0]
        R = self._rot6d_to_matrix(r6d)
        T = torch.eye(4, device=pos.device).unsqueeze(0).expand(B, -1, -1).clone()
        T[:, :3, :3] = R
        T[:, :3, 3] = pos
        return T

    # -- collision point pre-computation ----------------------------------
    @staticmethod
    def _fps(verts, n):
        """Farthest-point sampling on *verts* (N×3), returning *n* points."""
        n = min(n, len(verts))
        idx = [0]
        min_d = np.full(len(verts), np.inf)
        for _ in range(n - 1):
            d = np.linalg.norm(verts - verts[idx[-1]], axis=1)
            min_d = np.minimum(min_d, d)
            idx.append(int(np.argmax(min_d)))
        return verts[idx]

    def _precompute_collision_points(self):
        """Load collision points from URDF box primitives.

        For LEAP: uses box primitives from the updated URDF (ported from
        MuJoCo Menagerie). Each box generates 26 sample points (8 corners +
        12 edge midpoints + 6 face centers) for complete coverage.

        For Allegro: falls back to mesh vertex sampling (original approach).
        """
        from scipy.spatial.transform import Rotation as ScipyR

        tip_set = set(self.tip_link_names)

        if self.hand_type == "leap":
            # URDF box volumetric collision: fill each Menagerie box primitive
            # with a uniform 3mm 3D grid. These boxes are the PHYSICAL collision
            # geometry — watertight, no inner cavity, no vertex bias.
            # Collision check: object SDF at grid point >= 0 (no penetration).
            import xml.etree.ElementTree as _ET_col
            _urdf_col = os.path.join(_MODELS_DIR, f"leap_{self.hand}/leap_solver.urdf")
            _tree_col = _ET_col.parse(_urdf_col)
            _col_link_names = list(self.collision_link_names)
            col_data = []
            col_link_ranges = []
            _offset = 0
            _pitch = 0.005  # 5mm grid (all palm boxes included)

            for nm in _col_link_names:
                link_pts = []
                _is_palm = "palm" in nm
                _le = None
                for _e in _tree_col.getroot().findall("link"):
                    if _e.get("name") == nm:
                        _le = _e
                        break
                if _le is not None:
                    for _cel in _le.findall("collision"):
                        _g = _cel.find("geometry")
                        if _g is None: continue
                        _b = _g.find("box")
                        if _b is None: continue
                        _sz = [float(x) for x in _b.get("size").split()]
                        _o = _cel.find("origin")
                        _p = np.array([float(x) for x in _o.get("xyz", "0 0 0").split()])
                        _rpy = np.array([float(x) for x in _o.get("rpy", "0 0 0").split()])
                        _R = (ScipyR.from_euler("xyz", _rpy).as_matrix()
                              if np.any(np.abs(_rpy) > 1e-6) else np.eye(3))
                        # All palm boxes are checked — the large boxes
                        # (x < -0.025) ARE the palm plate, not structural.
                        hx, hy, hz = _sz[0]/2, _sz[1]/2, _sz[2]/2
                        gx = np.arange(-hx, hx + _pitch/2, _pitch)
                        gy = np.arange(-hy, hy + _pitch/2, _pitch)
                        gz = np.arange(-hz, hz + _pitch/2, _pitch)
                        grid = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'),
                                        axis=-1).reshape(-1, 3)
                        grid = ((_R @ grid.T).T + _p).astype(np.float32)
                        link_pts.append(grid)

                if link_pts:
                    pts = np.vstack(link_pts)
                else:
                    pts = np.array([[0, 0, 0]], dtype=np.float32)
                pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
                col_data.append((nm, torch.tensor(pts_h, device=self.device)))
                col_link_ranges.append((_offset, _offset + len(pts)))
                _offset += len(pts)

            # Add visual-mesh-surface-sampled points for ds (fingertip) links.
            # The URDF boxes miss the rounded tip (~30mm uncovered).
            # Sample uniformly on the visual mesh surface + interior fill.
            mesh_dir = os.path.join(_MODELS_DIR, f"leap_{self.hand}")
            vis_meshes = _visual_meshes(self.hand, self.hand_type)
            n_tip_surface = 150  # surface samples per tip link
            n_tip_interior = 50  # interior fill samples
            tip_pts_added = 0
            for ci, (nm, _) in enumerate(col_data):
                if "_ds" not in nm:
                    continue
                if nm not in vis_meshes:
                    continue
                # Load and transform visual mesh to link frame
                all_v = []
                all_f = []
                f_offset = 0
                for mf, vp in vis_meshes[nm]:
                    path = os.path.join(mesh_dir, mf)
                    if not os.path.exists(path):
                        continue
                    lm = trimesh.load(path, force="mesh")
                    v = np.asarray(lm.vertices, dtype=np.float64)
                    faces = np.asarray(lm.faces)
                    if vp is not None:
                        vpa = np.array(vp, dtype=np.float64)
                        Rv = ScipyR.from_euler("xyz", vpa[3:]).as_matrix()
                        v = (Rv @ v.T).T + vpa[:3]
                    all_v.append(v)
                    all_f.append(faces + f_offset)
                    f_offset += len(v)
                if not all_v:
                    continue
                tip_mesh = trimesh.Trimesh(
                    vertices=np.vstack(all_v), faces=np.vstack(all_f))
                # Sample on surface (Poisson-disk-like even distribution)
                try:
                    surf_pts, _ = trimesh.sample.sample_surface_even(
                        tip_mesh, n_tip_surface)
                except Exception:
                    surf_pts, _ = trimesh.sample.sample_surface(
                        tip_mesh, n_tip_surface)
                # Surface samples only — interior fill creates conflict with surface contact
                # (interior points 5-8mm deep fight Section A's surface-seeking)
                tip_pts = surf_pts.astype(np.float32)
                # Append to this link's collision data
                tip_h = np.hstack([tip_pts, np.ones((len(tip_pts), 1), dtype=np.float32)])
                existing = col_data[ci][1]
                combined = torch.cat([existing, torch.tensor(tip_h, device=self.device)], dim=0)
                col_data[ci] = (nm, combined)
                # Update range
                old_start, old_end = col_link_ranges[ci]
                col_link_ranges[ci] = (old_start, old_end + len(tip_pts))
                # Shift subsequent ranges
                for cj in range(ci + 1, len(col_link_ranges)):
                    s, e = col_link_ranges[cj]
                    col_link_ranges[cj] = (s + len(tip_pts), e + len(tip_pts))
                _offset += len(tip_pts)
                tip_pts_added += len(tip_pts)

            self._col_data = col_data
            self._col_link_ranges = col_link_ranges
            self._n_col_links = len(col_link_ranges)
            n_total = sum(p.shape[0] for _, p in col_data)
            print(f"  Box-grid collision: {len(col_data)} links, "
                  f"{n_total} points ({_pitch*1000:.0f}mm grid + {tip_pts_added} tip surface pts)")

            # Side-aware ds collision: classify ds points as back-side vs pad-side.
            # The fingertip pad faces -y in link frame. Points near the pad tip
            # (y < -15mm) wrap around curved surfaces by design — checking them
            # for collision penalizes normal contact physics.
            # Only back-side points (y > -15mm) indicate actual penetration.
            ds_back_mask = torch.zeros(n_total, device=self.device, dtype=torch.bool)
            n_back = 0
            for li, (nm, pts) in enumerate(col_data):
                if "_ds" not in nm:
                    continue
                si, ei = col_link_ranges[li]
                local_y = pts[:, 1]  # y-coordinate in link frame
                # Back/side: y > -15mm (URDF boxes + body region)
                # Pad extension: y < -20mm (visual mesh surface only)
                back = (local_y > -0.015)
                ds_back_mask[si:ei] = back
                n_back += back.sum().item()
            self._ds_back_mask = ds_back_mask
            print(f"  Side-aware ds: {n_back} back-side points (collision-checked), "
                  f"{n_total - n_back} pad-side (wrapping allowed)")

        else:
            # Allegro: original mesh-based collision points
            self._precompute_collision_points_mesh()
            return

        # Box-grid collision margins: sdf(point) >= margin.
        # Fingertips (_ds): -1mm (contact pads touch surface — tips
        #   must get close enough for the surface loss to work)
        # All other links (including palm): 0mm (no penetration)
        margins = []
        for li, (nm, pts) in enumerate(col_data):
            si, ei = col_link_ranges[li]
            if "_ds" in nm:
                m = -0.001  # fingertip contact pads
            else:
                m = 0.0
            margins.extend([m] * (ei - si))
        self._col_margins = torch.tensor(margins, dtype=torch.float32,
                                         device=self.device)

        # Self-collision uses URDF BOX points (physical collision geometry),
        # NOT visual mesh. Visual meshes have motor housing protrusions that
        # always overlap between adjacent fingers — that's a rendering artifact,
        # not a physical collision. The Menagerie boxes represent reality.
        #
        # Build a SEPARATE _sc_data point set from URDF boxes for SC.
        # _col_data (visual mesh) is used for hand-object collision only.
        import xml.etree.ElementTree as _ET_sc
        _urdf_sc = os.path.join(_MODELS_DIR, f"leap_{self.hand}/leap_solver.urdf")
        sc_data = []  # [(link_name, box_pts_tensor)]
        if os.path.exists(_urdf_sc):
            _tree_sc = _ET_sc.parse(_urdf_sc)
            for _le in _tree_sc.getroot().findall("link"):
                _ln = _le.get("name")
                if _ln not in self.collision_link_names:
                    continue
                _bpts = []
                _sc_pitch = 0.005  # 5mm grid (same as collision)
                for _col_elem in _le.findall("collision"):
                    _g = _col_elem.find("geometry")
                    if _g is None: continue
                    _b = _g.find("box")
                    if _b is None: continue
                    _sx, _sy, _sz = [float(x) for x in _b.get("size").split()]
                    _o = _col_elem.find("origin")
                    _p = np.array([float(x) for x in _o.get("xyz", "0 0 0").split()])
                    _rpy = np.array([float(x) for x in _o.get("rpy", "0 0 0").split()])
                    _R = ScipyR.from_euler("xyz", _rpy).as_matrix() if np.any(np.abs(_rpy) > 1e-6) else np.eye(3)
                    # Fill box with grid (not just corners) for accurate SC
                    _hx, _hy, _hz = _sx/2, _sy/2, _sz/2
                    _gx = np.arange(-_hx, _hx + _sc_pitch/2, _sc_pitch)
                    _gy = np.arange(-_hy, _hy + _sc_pitch/2, _sc_pitch)
                    _gz = np.arange(-_hz, _hz + _sc_pitch/2, _sc_pitch)
                    _grid = np.stack(np.meshgrid(_gx, _gy, _gz, indexing='ij'), axis=-1).reshape(-1, 3)
                    _grid = ((_R @ _grid.T).T + _p).astype(np.float32)
                    _bpts.append(_grid)
                if _bpts:
                    _pts = np.vstack(_bpts)
                else:
                    _pts = np.array([[0, 0, 0]], dtype=np.float32)
                # Subsample: keep at most _SC_LINK_MAX points per link
                # to avoid OOM on cdist. Use evenly-spaced indices.
                _SC_LINK_MAX = 40
                if len(_pts) > _SC_LINK_MAX:
                    step = len(_pts) / _SC_LINK_MAX
                    _keep = [int(k * step) for k in range(_SC_LINK_MAX)]
                    _pts = _pts[_keep]
                _pts_h = np.hstack([_pts, np.ones((len(_pts), 1), dtype=np.float32)])
                sc_data.append((_ln, torch.tensor(_pts_h, device=self.device)))
        self._sc_data = sc_data
        n_sc_pts = sum(p.shape[0] for _, p in sc_data)

        _SC_MAX = 40  # max SC points per finger group for SC pair detection
        finger_keys = ['if', 'mf', 'rf', 'th']
        _fcol = {}
        offset = 0
        palm_idx = []
        for nm, pts in sc_data:
            n = pts.shape[0]
            if 'palm' in nm:
                palm_idx.extend(range(offset, offset + n))
            for fk in finger_keys:
                if f'_{fk}_' in nm:
                    _fcol.setdefault(fk, []).extend(range(offset, offset + n))
                    break
            offset += n

        def _subsample(idx_list):
            if len(idx_list) > _SC_MAX:
                step = len(idx_list) / _SC_MAX
                return [idx_list[int(k * step)] for k in range(_SC_MAX)]
            return idx_list

        # Build per-link index for adjacency-aware SC pairs
        _link_sc_idx = {}  # link_name -> list of SC point indices
        offset_sc = 0
        for nm, pts in sc_data:
            n = pts.shape[0]
            _link_sc_idx[nm] = list(range(offset_sc, offset_sc + n))
            offset_sc += n

        # Adjacent link pairs (physically connected — should NOT be checked)
        _adjacent = {
            ('palm', 'if_bs'), ('palm', 'mf_bs'), ('palm', 'rf_bs'), ('palm', 'th_mp'),
            ('if_bs', 'if_px'), ('if_px', 'if_md'), ('if_md', 'if_ds'),
            ('mf_bs', 'mf_px'), ('mf_px', 'mf_md'), ('mf_md', 'mf_ds'),
            ('rf_bs', 'rf_px'), ('rf_px', 'rf_md'), ('rf_md', 'rf_ds'),
            ('th_mp', 'th_bs'), ('th_bs', 'th_px'), ('th_px', 'th_ds'),
        }
        _prefix = f"leap_{self.hand}_"

        # Build LINK-LEVEL SC pairs (not finger-level).
        # Every non-adjacent link pair from different fingers is checked.
        # This catches if_md↔mf_md overlap that finger-level grouping misses.
        self._self_col_pairs = []
        all_link_names = [nm for nm, _ in sc_data]
        for i in range(len(all_link_names)):
            for j in range(i + 1, len(all_link_names)):
                nm_i = all_link_names[i]
                nm_j = all_link_names[j]
                short_i = nm_i.replace(_prefix, '')
                short_j = nm_j.replace(_prefix, '')
                # Skip adjacent pairs
                if (short_i, short_j) in _adjacent or (short_j, short_i) in _adjacent:
                    continue
                # Skip same-finger pairs (non-adjacent within same finger)
                fi = short_i.split('_')[0] if '_' in short_i else short_i
                fj = short_j.split('_')[0] if '_' in short_j else short_j
                if fi == fj and fi != 'palm':
                    continue
                # This is a valid inter-finger, non-adjacent pair
                idx_i = _link_sc_idx[nm_i]
                idx_j = _link_sc_idx[nm_j]
                if idx_i and idx_j:
                    self._self_col_pairs.append((
                        torch.tensor(idx_i, dtype=torch.long, device=self.device),
                        torch.tensor(idx_j, dtype=torch.long, device=self.device),
                    ))

        print(f"  Self-collision: {len(self._self_col_pairs)} pairs, {n_sc_pts} box pts "
              f"(separate from {sum(p.shape[0] for _, p in col_data)} collision pts)")

        # Store per-link box primitives for feasibility SC check (box-box SDF).
        # Not used in optimization loop (too slow for 1700 pairs per step),
        # but used once per grasp in feasibility evaluation.
        self._box_primitives = {}
        for _le in _tree_sc.getroot().findall("link"):
            _ln = _le.get("name")
            if _ln not in self.collision_link_names: continue
            boxes = []
            for _col_elem in _le.findall("collision"):
                _g = _col_elem.find("geometry")
                if _g is None: continue
                _b = _g.find("box")
                if _b is None: continue
                _sx, _sy, _sz = [float(x) for x in _b.get("size").split()]
                _o = _col_elem.find("origin")
                _p = np.array([float(x) for x in _o.get("xyz", "0 0 0").split()])
                _rpy = np.array([float(x) for x in _o.get("rpy", "0 0 0").split()])
                _R = ScipyR.from_euler("xyz", _rpy).as_matrix() if np.any(np.abs(_rpy) > 1e-6) else np.eye(3)
                boxes.append((
                    torch.tensor([*_p, 1.0], dtype=torch.float32, device=self.device),
                    torch.tensor(_R, dtype=torch.float32, device=self.device),
                    torch.tensor([_sx/2, _sy/2, _sz/2], dtype=torch.float32, device=self.device),
                ))
            if boxes:
                self._box_primitives[_ln] = boxes

    def _precompute_collision_points_mesh(self):
        """Fallback: sample collision points from visual meshes (for Allegro)."""
        from scipy.spatial.transform import Rotation as ScipyR
        mesh_dir = os.path.join(_MODELS_DIR, "allegro")
        vis = _visual_meshes(self.hand, self.hand_type)
        tip_set = set(self.tip_link_names)
        max_lateral = 0.019
        tip_off_np = self.tip_offsets.cpu().numpy()
        tip_off_map = dict(zip(self.tip_link_names, tip_off_np))

        col_data = []
        for nm in self.collision_link_names:
            is_tip = nm in tip_set
            if nm not in vis:
                pts = np.array([[0, 0, 0]], dtype=np.float32)
            elif is_tip:
                all_verts = []
                for mesh_file, vis_pose in vis[nm]:
                    path = os.path.join(mesh_dir, mesh_file)
                    if not os.path.exists(path):
                        continue
                    lm = trimesh.load(path, force="mesh")
                    verts = np.asarray(lm.vertices, dtype=np.float64)
                    if vis_pose is not None:
                        vp = np.array(vis_pose, dtype=np.float64)
                        Rv = ScipyR.from_euler("xyz", vp[3:]).as_matrix()
                        verts = (Rv @ verts.T).T + vp[:3]
                    all_verts.append(verts)
                if all_verts:
                    all_verts = np.vstack(all_verts)
                    t_off = tip_off_map[nm]
                    dists = np.linalg.norm(all_verts - t_off, axis=1)
                    keep = dists > 0.010
                    body_verts = all_verts[keep] if keep.sum() >= 32 else all_verts
                    if len(body_verts) > 64:
                        centered = body_verts - body_verts.mean(axis=0)
                        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                        axis = Vt[0]
                        proj = np.outer(centered @ axis, axis)
                        lat_dist = np.linalg.norm(centered - proj, axis=1)
                        mask = lat_dist < max_lateral
                        if mask.sum() >= 64:
                            body_verts = body_verts[mask]
                    pts = self._fps(body_verts, 64).astype(np.float32)
                else:
                    pts = np.array([[0, 0, 0]], dtype=np.float32)
            else:
                mesh_file, vis_pose = vis[nm][0]
                path = os.path.join(mesh_dir, mesh_file)
                if not os.path.exists(path):
                    pts = np.array([[0, 0, 0]], dtype=np.float32)
                else:
                    lm = trimesh.load(path, force="mesh")
                    verts = np.asarray(lm.vertices, dtype=np.float64)
                    if vis_pose is not None:
                        vp = np.array(vis_pose, dtype=np.float64)
                        Rv = ScipyR.from_euler("xyz", vp[3:]).as_matrix()
                        verts = (Rv @ verts.T).T + vp[:3]
                    is_palm = "palm" in nm
                    n_pts = 256 if is_palm else 48
                    if not is_palm and len(verts) > n_pts:
                        centered = verts - verts.mean(axis=0)
                        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                        axis = Vt[0]
                        proj = np.outer(centered @ axis, axis)
                        lat_dist = np.linalg.norm(centered - proj, axis=1)
                        mask = lat_dist < max_lateral
                        if mask.sum() >= n_pts:
                            verts = verts[mask]
                    pts = self._fps(verts, n_pts).astype(np.float32)
            pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
            col_data.append((nm, torch.tensor(pts_h, device=self.device)))
        self._col_data = col_data

        # Per-link ranges for per-link AL
        col_link_ranges = []
        _lr_off = 0
        for nm, pts in col_data:
            n = pts.shape[0]
            col_link_ranges.append((_lr_off, _lr_off + n))
            _lr_off += n
        self._col_link_ranges = col_link_ranges
        self._n_col_links = len(col_link_ranges)

        # Margins and self-collision (shared code)
        margins = []
        for nm, pts in col_data:
            if "palm" in nm:
                m = 0.003
            elif nm in set(self.tip_link_names):
                m = 0.005
            else:
                m = 0.002
            margins.extend([m] * pts.shape[0])
        self._col_margins = torch.tensor(margins, dtype=torch.float32, device=self.device)

        _SC_MAX = 80
        finger_keys = ['if', 'mf', 'rf', 'th']
        _fcol = {}
        offset = 0
        for nm, pts in col_data:
            n = pts.shape[0]
            for fk in finger_keys:
                if f'_{fk}_' in nm:
                    _fcol.setdefault(fk, []).extend(range(offset, offset + n))
                    break
            offset += n
        self._self_col_pairs = []
        fk_list = [k for k in finger_keys if k in _fcol]
        for i in range(len(fk_list)):
            for j in range(i + 1, len(fk_list)):
                idx1 = _fcol[fk_list[i]]
                idx2 = _fcol[fk_list[j]]
                if len(idx1) > _SC_MAX:
                    step = len(idx1) / _SC_MAX
                    idx1 = [idx1[int(k * step)] for k in range(_SC_MAX)]
                if len(idx2) > _SC_MAX:
                    step = len(idx2) / _SC_MAX
                    idx2 = [idx2[int(k * step)] for k in range(_SC_MAX)]
                self._self_col_pairs.append((
                    torch.tensor(idx1, dtype=torch.long, device=self.device),
                    torch.tensor(idx2, dtype=torch.long, device=self.device),
                ))

    # -- per-stage metrics (for diagnostic) ----------------------------------
    def _snap_metrics(self, tag):
        """Record per-env feasibility metrics at this stage.

        Stores arrays of [B] metrics so we can analyze trajectories later:
        surf_err, ds_back_worst, ds_pad_worst, max_col_viol, sc_worst_sdf.
        """
        if not hasattr(self, '_metrics_log'):
            self._metrics_log = {}
        with torch.no_grad():
            B = self.num_envs
            dev = self.device
            q = self._u2q(self.u)
            bT = self._base_T(self.pos, self.rot6d)
            fk = self.chain.forward_kinematics(q)
            # Tip SDF at support finger tips (tip_offset point)
            surf = torch.zeros(B, 4, device=dev)
            for fi in range(4):
                nm = self.tip_link_names[fi]
                if nm not in fk: continue
                wT = bT @ fk[nm].get_matrix()
                off_h = torch.cat([self.tip_offsets[fi], torch.ones(1, device=dev)])
                tip = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                surf[:, fi] = self.sdf.query(tip.unsqueeze(1)).squeeze(1).abs()
            # surf_err (excluding actuation finger) — the max among support fingers
            if hasattr(self, 'amap_t'):
                is_act = torch.zeros(B, 4, device=dev, dtype=torch.bool)
                for b in range(B):
                    is_act[b, int(self.amap[b, 0])] = True
                surf_masked = torch.where(is_act, torch.zeros_like(surf), surf)
                surf_err = surf_masked.max(dim=-1).values
            else:
                surf_err = surf.max(dim=-1).values
            # Collision points
            if hasattr(self, '_col_data'):
                # Build cs per link
                ds_back_worst = torch.zeros(B, device=dev)
                ds_pad_worst = torch.zeros(B, device=dev)
                max_col_viol = torch.zeros(B, device=dev)
                for li, (nm, lp) in enumerate(self._col_data):
                    if nm not in fk: continue
                    lwT = bT @ fk[nm].get_matrix()
                    lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                    lsdf = self.sdf.query(lwp)
                    if "_ds" in nm:
                        si, ei = self._col_link_ranges[li]
                        back_m = self._ds_back_mask[si:ei]
                        # Skip actuation finger's ds link per-env
                        # Determine which finger this ds is
                        prefixes_ds = ['if', 'mf', 'rf', 'th']
                        ds_fi = next((pi for pi, p in enumerate(prefixes_ds) if f"_{p}_ds" in nm), None)
                        not_act = (self.amap_t[:, 0] != ds_fi) if ds_fi is not None else torch.ones(B, dtype=torch.bool, device=dev)
                        if back_m.any():
                            b_sdf = lsdf[:, back_m].min(-1).values
                            ds_back_worst = torch.where(not_act, torch.minimum(ds_back_worst, b_sdf), ds_back_worst)
                        if (~back_m).any():
                            p_sdf = lsdf[:, ~back_m].min(-1).values
                            ds_pad_worst = torch.where(not_act, torch.minimum(ds_pad_worst, p_sdf), ds_pad_worst)
                    else:
                        # Non-ds: relu(-sdf) (no margin, stricter than feasibility -3mm)
                        viol = F.relu(-lsdf).max(-1).values
                        max_col_viol = torch.maximum(max_col_viol, viol)
            else:
                ds_back_worst = torch.zeros(B, device=dev)
                ds_pad_worst = torch.zeros(B, device=dev)
                max_col_viol = torch.zeros(B, device=dev)
            # sc box-box SDF (only if box primitives exist)
            sc_worst = torch.full((B,), 1.0, device=dev)
            if hasattr(self, '_box_primitives'):
                _prefix = f"leap_{self.hand}_"
                _adj = {('palm','if_bs'),('palm','mf_bs'),('palm','rf_bs'),('palm','th_mp'),
                        ('if_bs','if_px'),('if_px','if_md'),('if_md','if_ds'),
                        ('mf_bs','mf_px'),('mf_px','mf_md'),('mf_md','mf_ds'),
                        ('rf_bs','rf_px'),('rf_px','rf_md'),('rf_md','rf_ds'),
                        ('th_mp','th_bs'),('th_bs','th_px'),('th_px','th_ds')}
                link_wT = {}
                for nm in self._box_primitives:
                    if nm in fk:
                        link_wT[nm] = bT @ fk[nm].get_matrix()
                for nm_i in self._box_primitives:
                    if nm_i not in link_wT: continue
                    for nm_j in self._box_primitives:
                        if nm_j not in link_wT or nm_i >= nm_j: continue
                        si_ = nm_i.replace(_prefix, ''); sj_ = nm_j.replace(_prefix, '')
                        if (si_, sj_) in _adj or (sj_, si_) in _adj: continue
                        fi_ = si_.split('_')[0]; fj_ = sj_.split('_')[0]
                        if fi_ == fj_ and fi_ != 'palm': continue
                        for bi_c, bi_r, bi_h in self._box_primitives[nm_i]:
                            for bj_c, bj_r, bj_h in self._box_primitives[nm_j]:
                                ci_w = (link_wT[nm_i] @ bi_c.unsqueeze(-1)).squeeze(-1)[:, :3]
                                ri_w = link_wT[nm_i][:, :3, :3] @ bi_r.unsqueeze(0)
                                cj_w = (link_wT[nm_j] @ bj_c.unsqueeze(-1)).squeeze(-1)[:, :3]
                                rj_w = link_wT[nm_j][:, :3, :3] @ bj_r.unsqueeze(0)
                                sd = box_box_sdf_batch(ci_w, ri_w, bi_h.unsqueeze(0).expand(B, -1),
                                                       cj_w, rj_w, bj_h.unsqueeze(0).expand(B, -1))
                                sc_worst = torch.minimum(sc_worst, sd)
            self._metrics_log[tag] = {
                'surf_err': surf_err.cpu().numpy(),
                'ds_back_worst': ds_back_worst.cpu().numpy(),
                'ds_pad_worst': ds_pad_worst.cpu().numpy(),
                'max_col_viol': max_col_viol.cpu().numpy(),
                'sc_worst': sc_worst.cpu().numpy(),
            }
            s = self._metrics_log[tag]
            import numpy as _np
            print(f"  [METRICS {tag}] surf_err med={_np.median(s['surf_err'])*1000:.1f}mm "
                  f"ds_back med={_np.median(s['ds_back_worst'])*1000:.1f}mm "
                  f"ds_pad med={_np.median(s['ds_pad_worst'])*1000:.1f}mm "
                  f"col_viol med={_np.median(s['max_col_viol'])*1000:.1f}mm "
                  f"sc med={_np.median(s['sc_worst'])*1000:.1f}mm")

    # -- phase snapshot (for tracing optimisation) --------------------------
    def _snapshot(self, tag, idx=0):
        """Save a snapshot of a single environment's grasp state."""
        with torch.no_grad():
            q = self._u2q(self.u)
            bT = self._base_T(self.pos, self.rot6d)
            fk = self.chain.forward_kinematics(q)
            tp, cp, _ = self._get_points(fk, bT)
            ts = self.sdf.query(tp)
            cs = self.sdf.query(cp)

            # Extract scalar state for env idx
            snap = {
                "tag": tag,
                "q_joints": q[idx].cpu().numpy(),
                "base_pos": self.pos[idx].cpu().numpy(),
                "base_rot": self._rot6d_to_matrix(self.rot6d[idx:idx+1])[0].cpu().numpy(),
                "tip_sdf": ts[idx].cpu().numpy(),
                "col_sdf": cs[idx].cpu().numpy(),
                "tip_sdf_abs_mean": ts[idx].abs().mean().item(),
                "col_min_sdf": cs[idx].min().item(),
                "col_inside_pct": (cs[idx] < 0).float().mean().item() * 100,
                "col_margin_violated": (self._col_margins - cs[idx] > 0).float().mean().item() * 100,
            }
            if not hasattr(self, '_phase_snapshots'):
                self._phase_snapshots = []
            self._phase_snapshots.append(snap)

            # Palm orientation check
            R_all = self._rot6d_to_matrix(self.rot6d)
            palm_inward = R_all[:, :, 0]  # base +X = palm inward
            B_check = min(self.num_envs, self.pos.shape[0])
            if hasattr(self, '_obj_center'):
                to_c = self._obj_center.unsqueeze(0) - self.pos[:B_check].detach()
                to_c = to_c / to_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                dots = (palm_inward[:B_check] * to_c).sum(-1)
                n_face = (dots > 0.3).sum().item()
                n_away = (dots < -0.3).sum().item()
            else:
                n_face = n_away = -1
                dots = torch.zeros(1)

            print(f"  [SNAP] {tag}: tip_sdf={snap['tip_sdf_abs_mean']*1000:.1f}mm "
                  f"col_inside={snap['col_inside_pct']:.1f}% "
                  f"col_min={snap['col_min_sdf']*1000:.1f}mm "
                  f"palm_facing={n_face}/{B_check} away={n_away}")

            # Save best 10 grasps — use consistent ordering across all stages.
            # _final_order is set once at the end of optimization and reused
            # for all snapshots so the same grasp index = same grasp across stages.
            if hasattr(self, '_final_order'):
                order = self._final_order.cpu().numpy()
            elif hasattr(self, '_opt_quality_order'):
                order = self._opt_quality_order.cpu().numpy()
            elif hasattr(self, '_act_sort_order'):
                order = self._act_sort_order.cpu().numpy()
            else:
                order = np.arange(B_check)
            # Pre-compute sigma_min for all envs if we have the needed data
            sigma_min_all = None
            if tag == "after_optimization" and hasattr(self, '_F_prim') and hasattr(self, '_ns'):
                try:
                    F_prim_snap = self._F_prim
                    ns_snap = self._ns
                    # Compute tip positions for 4 fingertips
                    tp_fc = torch.stack([
                        (bT @ fk[self.tip_link_names[fi]].get_matrix()
                         @ torch.cat([self.tip_offsets[fi], torch.ones(1, device=self.device)]).unsqueeze(-1)
                        ).squeeze(-1)[:, :3]
                        for fi in range(4)], dim=1)  # [B, 4, 3]
                    # Query SDF normals via finite differences
                    eps_fd = 5e-4
                    fd_o = torch.zeros(3, 3, device=self.device)
                    for d3 in range(3): fd_o[d3, d3] = eps_fd
                    gx = (self.sdf.query(tp_fc+fd_o[0]) - self.sdf.query(tp_fc-fd_o[0])) / (2*eps_fd)
                    gy = (self.sdf.query(tp_fc+fd_o[1]) - self.sdf.query(tp_fc-fd_o[1])) / (2*eps_fd)
                    gz = (self.sdf.query(tp_fc+fd_o[2]) - self.sdf.query(tp_fc-fd_o[2])) / (2*eps_fd)
                    tn = -torch.stack([gx, gy, gz], dim=-1)
                    tn = tn / tn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    # Compute contact frames, grasp matrix, wrench matrix
                    g_OCs = compute_contact_frames(tp_fc, tn)
                    G = compute_grasp_matrix_torch(g_OCs)
                    W = compute_wrench_matrix(G, F_prim_snap, 4, ns_snap)
                    sigma_min_all = torch.linalg.svdvals(W)[:, -1]  # [B]
                except Exception as e:
                    print(f"  [SNAP] Warning: sigma_min computation failed: {e}")
                    sigma_min_all = None

            grasps = []
            for rank in range(min(10, B_check)):
                i = int(order[rank]) if rank < len(order) else rank
                sm = sigma_min_all[i].item() if sigma_min_all is not None else 0.0
                g_dict = {
                    "q_joints": q[i].cpu().numpy(),
                    "base_pos": self.pos[i].detach().cpu().numpy(),
                    "base_rot": R_all[i].cpu().numpy(),
                    "sigma_min": sm, "l_star": 0.0, "feasible": False,
                }
                if hasattr(self, '_init_surf_pts') and i < len(self._init_surf_pts):
                    g_dict["surf_pt"] = self._init_surf_pts[i].numpy()
                    g_dict["outward_normal"] = self._init_outward[i].numpy()
                    g_dict["z_hat_init"] = self._init_z_hat[i].numpy()
                g_dict["act_finger"] = int(self.amap[i, 0])
                g_dict["env_idx"] = int(i)
                grasps.append(g_dict)
            if _STAGE_DEBUG:
                save_dir = os.path.join("out", "grasp_debug")
                os.makedirs(save_dir, exist_ok=True)
                import torch as _t
                _t.save(grasps, os.path.join(save_dir, f"stage_{tag}.pt"))

    # -- FK point extraction ----------------------------------------------
    def _get_points(self, fk, bT):
        dev = self.device
        tips = []
        tip_x_axes = []  # x-axis of each fingertip frame (push direction)
        for i, nm in enumerate(self.tip_link_names):
            wT = bT @ fk[nm].get_matrix()  # [B, 4, 4]
            oh = torch.cat([self.tip_offsets[i], torch.ones(1, device=dev)])
            tips.append((wT @ oh.unsqueeze(-1)).squeeze(-1)[:, :3])
            tip_x_axes.append(-wT[:, :3, 0])  # -x of tip link = pad push direction
        # Add palm contact points if enabled (4 spread points for area contact)
        if self.palm_contact and self.palm_link in fk:
            wT_palm = bT @ fk[self.palm_link].get_matrix()
            n_palm_pts = len(self.tip_offsets) - 4  # offsets after the 4 finger tips
            for pi in range(n_palm_pts):
                palm_off_i = self.tip_offsets[4 + pi]
                oh = torch.cat([palm_off_i, torch.ones(1, device=dev)])
                tips.append((wT_palm @ oh.unsqueeze(-1)).squeeze(-1)[:, :3])
                # Palm contact normal: +x in base frame (inner surface faces +x)
                tip_x_axes.append(wT_palm[:, :3, 0])
        cols = []
        for nm, local_pts in self._col_data:
            if nm in fk:
                wT = bT @ fk[nm].get_matrix()          # [B, 4, 4]
                wp = (wT @ local_pts.T)[:, :3, :].transpose(1, 2)  # [B, N_i, 3]
                cols.append(wp)
        nc = len(tips)  # 4 or 5
        return (torch.stack(tips, 1), torch.cat(cols, dim=1),
                torch.stack(tip_x_axes, 1))  # [B, nc, 3]

    def _get_sc_points(self, fk, bT):
        """Get self-collision box points in world frame (from URDF boxes)."""
        sc_cols = []
        for nm, local_pts in self._sc_data:
            if nm in fk:
                wT = bT @ fk[nm].get_matrix()
                wp = (wT @ local_pts.T)[:, :3, :].transpose(1, 2)
                sc_cols.append(wp)
        return torch.cat(sc_cols, dim=1) if sc_cols else None

    def _get_pad_points(self, fk, bT):
        """Get fingertip pad sample points in world frame.

        Returns: [B, nc, n_pad, 3] where n_pad=7 for fingers, 1 for palm.
        For the surface loss, use min(sdf) over the pad dimension.
        """
        dev = self.device
        all_pads = []
        for i, nm in enumerate(self.tip_link_names):
            wT = bT @ fk[nm].get_matrix()  # [B, 4, 4]
            pad_pts = self.pad_offsets[i]  # [n_pad, 3]
            pad_h = torch.cat([pad_pts, torch.ones(pad_pts.shape[0], 1, device=dev)], -1)  # [n_pad, 4]
            wp = (wT @ pad_h.T)[:, :3, :].transpose(1, 2)  # [B, n_pad, 3]
            all_pads.append(wp)
        if self.palm_contact and self.palm_link in fk:
            wT_palm = bT @ fk[self.palm_link].get_matrix()
            palm_pad = self.pad_offsets[-1]  # [1, 3]
            palm_h = torch.cat([palm_pad, torch.ones(1, 1, device=dev)], -1)
            wp = (wT_palm @ palm_h.T)[:, :3, :].transpose(1, 2)
            all_pads.append(wp)
        return all_pads  # list of [B, n_pad_i, 3]

    # -- initialisation ---------------------------------------------------
    def _init(self, center, n_act, act_positions=None, act_directions=None):
        """Initialise by sampling palm contact points ON the object surface.

        Strategy: pick random surface points on the object body, compute the
        base pose that places the palm inner surface at each point with the
        correct orientation (palm tangent to surface). This guarantees the
        palm starts in contact with the object.
        """
        B, dev = self.num_envs, self.device

        if self.hand_type == "leap":
            # Seed: moderate curl for all fingers
            dq = torch.tensor(
                [1.0, 0.0, 0.7, 0.7,   # IF
                 1.0, 0.0, 0.7, 0.7,   # MF
                 1.0, 0.0, 0.7, 0.7,   # RF
                 1.0, 0.4, 0.7, 0.7],  # TH
                device=dev,
            )
        else:
            dq = torch.tensor(
                [0.0, 0.5, 0.5, 0.5,
                 0.0, 0.5, 0.5, 0.5,
                 0.0, 0.5, 0.5, 0.5,
                 1.2, 0.5, 0.5, 0.4],
                device=dev,
            )
        du = self._q2u(dq)
        self.u = (du + 0.3 * torch.randn(B, 16, device=dev)).detach().requires_grad_(True)

        c = torch.tensor(center, dtype=torch.float32, device=dev)

        # --- Surface-based palm placement ---
        # 1. Sample surface points near the actuation target (if any)
        # 2. Get surface normals at those points
        # 3. Compute base pose that places palm inner surface AT each point
        with torch.no_grad():
            # Sample surface points UNIFORMLY from the object mesh
            # (not mesh vertices — vertices are non-uniformly distributed).
            # Uses trimesh area-weighted sampling for uniform coverage.
            obj_mesh_W = trimesh.Trimesh(
                vertices=self.sdf._verts_W,
                faces=self.sdf._faces)
            n_pool = max(B * 4, 10000)  # oversample, then filter
            pool_pts, _ = trimesh.sample.sample_surface(obj_mesh_W, n_pool)
            pool_pts = torch.tensor(pool_pts, dtype=torch.float32, device=dev)

            z_range = pool_pts[:, 2].max() - pool_pts[:, 2].min()
            z_min = pool_pts[:, 2].min()

            # Sampling: 50% uniform on body, 50% biased near actuation target.
            # Biasing toward actuation ensures enough palm positions can reach the button.
            z_cutoff = z_min + 0.15 * z_range
            valid_mask = pool_pts[:, 2] > z_cutoff
            valid_pts = pool_pts[valid_mask] if valid_mask.sum() >= B else pool_pts

            # 70/30 split (was 50/50): more uniform palm placements help
            # multi-region objects (air_blower) where palm-near-trigger leaves
            # supports unable to reach the body. Keep some biased samples for
            # small objects whose trigger requires palm proximity.
            n_uniform = int(B * 0.7)
            n_biased = B - n_uniform
            # Uniform half
            idx_uniform = torch.randint(0, valid_pts.shape[0], (n_uniform,), device=dev)
            # Biased half: prefer points near the actuation target
            if act_positions is not None and len(act_positions) > 0:
                act_pt = torch.tensor(act_positions[0], dtype=torch.float32, device=dev)
                dists_to_act = torch.norm(valid_pts - act_pt.unsqueeze(0), dim=-1)
                # Weight inversely with distance (closer = more likely)
                weights = 1.0 / (dists_to_act + 0.01)
                weights = weights / weights.sum()
                idx_biased = torch.multinomial(weights, n_biased, replacement=True)
            else:
                idx_biased = torch.randint(0, valid_pts.shape[0], (n_biased,), device=dev)
            idx = torch.cat([idx_uniform, idx_biased])
            surf_pts = valid_pts[idx]

            # Get outward normals via SDF gradient (uniform points, not vertices)
            surf_pts_q = surf_pts.unsqueeze(1)  # [B, 1, 3]
            _, surf_normals = self.sdf.query_with_normals(surf_pts_q)
            outward_normals = -surf_normals[:, 0, :]  # negate inward to get outward
            outward_normals = F.normalize(outward_normals, dim=-1)

            # Palm contact face faces base +X (fingertip curl test: dot=0.989).
            # In link frame, base +X = link -Z (R_palm^T @ [1,0,0] = [0,0,-1]).
            # Contact center: area-weighted -Z face centers, link→base transformed.
            palm_contact_base = torch.tensor([0.023, -0.000, 0.048], device=dev)

            # --- Structured sampling: (surface_point, distance, 1-of-4 rotations) ---

            # Palm inward = base +X (verified: fingertip curl aligns 0.989 with +X).
            # So base +X = toward object = -outward.
            x_hat = -outward_normals  # [B, 3]

            # Y-axis: one of 4 canonical choices from OBB axes.
            # Project all 3 OBB axes onto the plane ⊥ x_hat.
            obb_axes = self._obb_axes  # [3, 3]
            projections = []
            for ai in range(3):
                ax = obb_axes[:, ai].unsqueeze(0).expand(B, -1)
                proj = ax - (ax * x_hat).sum(-1, keepdim=True) * x_hat
                proj_norm = proj.norm(dim=-1)
                projections.append((proj, proj_norm))

            norms = torch.stack([p[1] for p in projections], dim=-1)
            _, top2_idx = norms.topk(2, dim=-1)

            y_cand0 = torch.zeros(B, 3, device=dev)
            y_cand1 = torch.zeros(B, 3, device=dev)
            for b_idx in range(B):
                i0, i1 = top2_idx[b_idx, 0].item(), top2_idx[b_idx, 1].item()
                y_cand0[b_idx] = projections[i0][0][b_idx]
                y_cand1[b_idx] = projections[i1][0][b_idx]
            y_cand0 = F.normalize(y_cand0, dim=-1)
            y_cand1 = F.normalize(y_cand1, dim=-1)

            choice = torch.arange(B, device=dev) % 4
            y_hat = torch.where((choice == 0).unsqueeze(-1), y_cand0,
                    torch.where((choice == 1).unsqueeze(-1), -y_cand0,
                    torch.where((choice == 2).unsqueeze(-1), y_cand1,
                    -y_cand1)))

            z_hat = torch.cross(x_hat, y_hat, dim=-1)
            z_hat = F.normalize(z_hat, dim=-1)
            R_base = torch.stack([x_hat, y_hat, z_hat], dim=-1)  # [B, 3, 3]

            # Palm-to-surface distance: was [0, 2cm], now [0, 5cm]. Tight palm
            # (close to surface) forces fingers to wrap tight and proximal links
            # to dig into the object for thick bodies (air_blower body 65mm).
            # Wider range gives some envs the "loose wrap" geometry where the
            # palm sits farther out and fingers stretch over the body without
            # link penetration. Hard objects need this; easy small objects
            # (grinder) still work at small d.
            d = 0.05 * torch.rand(B, device=dev)
            target = surf_pts + d.unsqueeze(-1) * outward_normals

        # 5) Store rotation (no noise — canonical poses are exact)
        r6d = torch.cat([x_hat, y_hat], dim=-1)
        self.rot6d = r6d.detach().requires_grad_(False)  # FROZEN

        # 6) Position: exact, using the actual stored rotation
        R_actual = self._rot6d_to_matrix(self.rot6d)
        if self.hand_type == "leap":
            # Place the same exposed palm point used by the FC certificate.
            palm_fk = self.chain.forward_kinematics(self._u2q(self.u))
            palm_mat = palm_fk[self.palm_link].get_matrix()
            palm_base_point = palm_reference_in_base(palm_mat, self.palm_offset)
        else:
            # Preserve the historical Allegro initialization reference.
            palm_base_point = palm_contact_base
        contact_in_world = (R_actual @ palm_base_point.unsqueeze(-1)).squeeze(-1)
        base_pos = target - contact_in_world
        self.pos = base_pos.detach().requires_grad_(True)
        self._rot6d_init = self.rot6d.detach().clone()

        # Save init debug info (surface points, normals, z_hat) for visualization
        self._init_surf_pts = surf_pts.detach().cpu()
        self._obj_z_min = float(valid_pts[:, 2].min().item())
        self._init_outward = outward_normals.detach().cpu()
        self._init_z_hat = z_hat.detach().cpu()

        # Actuation-finger assignment
        self.amap = np.zeros((B, max(n_act, 1)), dtype=np.int64)
        if n_act:
            for b in range(B):
                self.amap[b] = [(b + i) % 4 for i in range(n_act)]
        else:
            for b in range(B):
                self.amap[b] = [0]
        self.amap_t = torch.tensor(self.amap, dtype=torch.long, device=dev)
        if hasattr(self, '_stage_times'):
            import time as _t
            if dev == "cuda": torch.cuda.synchronize()
            self._stage_times["_t_act_ik"] = _t.time()

        # ================================================================
        # Actuation finger IK: position + pad direction
        # ================================================================
        if n_act and act_positions is not None:
            act_pos = torch.tensor(act_positions[0], dtype=torch.float32, device=dev)
            act_dir = None
            if act_directions is not None and act_directions[0] is not None:
                act_dir = F.normalize(
                    torch.tensor(act_directions[0], dtype=torch.float32, device=dev), dim=0)

            # Multi-assignment IK: each env tries all 4 actuation finger
            # assignments and keeps the one that reaches the target closest.
            # Per env, instead of committing to amap=(b%4) we give each base
            # pose 4 shots at finding a finger that can reach the target.
            # Expected: candidate pool grows substantially for objects where
            # only specific fingers work from any given base pose (e.g.
            # hot_glue_gun with 4/4000 in single-assignment mode).
            # Toggle off with CODEX_GRASP_NO_MULTI=1 for baseline ablation.
            trial_fis = [None] if _DISABLE_MULTI_ASSIGN else list(range(4))
            u_init_all = self.u.detach().clone()

            best_u_full = u_init_all.clone()
            best_act_dist = torch.full((B,), float('inf'), device=dev)
            best_score = torch.full((B,), float('inf'), device=dev)
            best_fi = torch.zeros(B, device=dev, dtype=torch.long)

            prefixes_ik = ['if', 'mf', 'rf', 'th']
            sfx_ik = [['bs', 'px', 'md']] * 3 + [['mp', 'bs', 'px']]

            for trial_fi in trial_fis:
                # trial_fi=None → keep per-env amap; otherwise all envs try trial_fi.
                u_act = u_init_all.clone().requires_grad_(True)
                opt_act = torch.optim.Adam([u_act], lr=0.05)
                act_joint_mask = torch.zeros(B, 16, device=dev, dtype=torch.bool)
                if trial_fi is not None:
                    act_joint_mask[:, trial_fi*4:trial_fi*4+4] = True
                    fi_list = [trial_fi]  # only this finger contributes
                else:
                    # Per-env: each env keeps its existing amap
                    for b in range(B):
                        fi_b = int(self.amap[b, 0])
                        act_joint_mask[b, fi_b*4:fi_b*4+4] = True
                    fi_list = list(range(4))  # loop all so every env sees its own

                # Total IK budget = phase1 + phase2 (two-phase) or 150 (single).
                # Two-phase rationale: pipeline-IK pos_loss stagnates when dir+col
                # (30-50× larger at init) dominate Adam's gradient budget. Phase 1
                # uses pure pos loss to let tip reach target; phase 2 adds dir+col
                # with 10× pos weight to preserve reach while refining.
                # Single-phase IK with rebalanced losses (same fix we applied
                # to support IK): linear distance, softplus-smooth facing loss,
                # mean-over-points collision. Two-phase was a workaround for
                # the loss-balance bug; now that the balance is correct we
                # don't need the phase split.
                total_ik_steps = 250

                for ik_step in range(total_ik_steps):
                    opt_act.zero_grad()
                    q_ik = self._u2q(u_act)
                    bT_ik = self._base_T(self.pos.detach(), self.rot6d.detach())
                    fk_ik = self.chain.forward_kinematics(q_ik)

                    loss_ik = torch.zeros(B, device=dev)
                    for fi_this in fi_list:
                        if trial_fi is not None:
                            mask_fi = torch.ones(B, device=dev)
                        else:
                            mask_fi = (self.amap_t[:, 0] == fi_this).float()
                            if mask_fi.sum() == 0: continue
                        nm = self.tip_link_names[fi_this]
                        off_h = torch.cat([self.tip_offsets[fi_this], torch.ones(1, device=dev)])
                        wT_tip = bT_ik @ fk_ik[nm].get_matrix()
                        tip_pos = (wT_tip @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                        tip_pad_dir = -wT_tip[:, :3, 0]

                        # Linear distance (constant gradient magnitude) so pos
                        # loss can compete with aux losses when tip is far.
                        pos_dist = torch.norm(tip_pos - act_pos, dim=-1)
                        loss_ik += mask_fi * 500 * pos_dist
                        if act_dir is not None:
                            # Direction matters more for actuation (finger must
                            # PUSH in the right direction to trigger the
                            # mechanism). Use a smooth threshold at cos>0.5 —
                            # near-zero when reasonably aligned, grows smoothly
                            # as alignment degrades.
                            cos_align = (tip_pad_dir * act_dir).sum(-1)
                            bad_face = F.softplus((0.5 - cos_align) * 10) / 10
                            loss_ik += mask_fi * 50 * bad_face ** 2

                        # Mean-over-points collision (not sum) so col doesn't
                        # dominate pos through point-count amplification.
                        for suf in sfx_ik[fi_this]:
                            lnm = f"leap_{self.hand}_{prefixes_ik[fi_this]}_{suf}"
                            for cnm, lp in self._col_data:
                                if cnm == lnm and lnm in fk_ik:
                                    lwT = bT_ik @ fk_ik[lnm].get_matrix()
                                    lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                                    lsdf = self.sdf.query(lwp, include_clearance=False)
                                    loss_ik += mask_fi * 100 * F.relu(-lsdf).mean(-1)

                    non_act_reg = ((u_act - u_init_all) ** 2 * (~act_joint_mask).float()).sum(-1)
                    loss_ik += 100 * non_act_reg
                    loss_ik.mean().backward()
                    with torch.no_grad():
                        u_act.grad[~act_joint_mask] = 0.0
                    opt_act.step()

                # Score this trial per env. In single-assign mode we only have
                # one trial so scoring is moot — just commit.
                with torch.no_grad():
                    q_trial = self._u2q(u_act.detach())
                    bT_trial = self._base_T(self.pos, self.rot6d)
                    fk_trial = self.chain.forward_kinematics(q_trial)

                    trial_dist = torch.zeros(B, device=dev)
                    for fi_this in fi_list:
                        if trial_fi is not None:
                            mask_fi = torch.ones(B, device=dev, dtype=torch.bool)
                        else:
                            mask_fi = (self.amap_t[:, 0] == fi_this)
                            if not mask_fi.any(): continue
                        nm = self.tip_link_names[fi_this]
                        off_h = torch.cat([self.tip_offsets[fi_this], torch.ones(1, device=dev)])
                        wT_tip_f = bT_trial @ fk_trial[nm].get_matrix()
                        tip_pos_f = (wT_tip_f @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                        d = torch.norm(tip_pos_f - act_pos, dim=-1)
                        trial_dist = torch.where(mask_fi, d, trial_dist)

                    palm_worst = torch.zeros(B, device=dev)
                    for cnm_p, lp_p in self._col_data:
                        if cnm_p != self.palm_link or cnm_p not in fk_trial: continue
                        lwT_p = bT_trial @ fk_trial[cnm_p].get_matrix()
                        lwp_p = (lwT_p @ lp_p.T)[:, :3, :].transpose(1, 2)
                        palm_worst = self.sdf.query(lwp_p, include_clearance=False).min(-1).values
                        break

                    # Palm-aware: reject trials with palm penetration >3mm.
                    palm_pen = F.relu(-palm_worst - 0.003)
                    trial_score = trial_dist + 10.0 * palm_pen

                    fi_tensor = (torch.full((B,), trial_fi, device=dev, dtype=torch.long)
                                 if trial_fi is not None else self.amap_t[:, 0])
                    improved = trial_score < best_score
                    best_score = torch.where(improved, trial_score, best_score)
                    best_u_full = torch.where(improved.unsqueeze(-1), u_act.detach(), best_u_full)
                    best_act_dist = torch.where(improved, trial_dist, best_act_dist)
                    best_fi = torch.where(improved, fi_tensor, best_fi)

            # Commit best assignment per env
            with torch.no_grad():
                self.u = best_u_full.clone().requires_grad_(True)
                self.amap[:, 0] = best_fi.cpu().numpy()
                self.amap_t = torch.from_numpy(self.amap).to(dev)

            # Report IK success (using the chosen assignment)
            with torch.no_grad():
                act_dists = best_act_dist
                n_close = (act_dists < 0.010).sum().item()
                n_vclose = (act_dists < 0.005).sum().item()
                # Distribution of chosen assignments
                assign_counts = [int((best_fi == fi).sum().item()) for fi in range(4)]
                print(f"  Actuation IK (multi-assign): {n_vclose}/{B} within 5mm, "
                      f"{n_close}/{B} within 10mm, median={act_dists.median()*1000:.1f}mm")
                print(f"    assignment counts: if={assign_counts[0]} mf={assign_counts[1]} "
                      f"rf={assign_counts[2]} th={assign_counts[3]}")
                self._act_sort_order = act_dists.argsort()

        # ================================================================
        # Co-optimize: palm tangent slide + actuation finger re-projection
        # Goal: clear palm-back collision while keeping actuation finger on target.
        # ================================================================
        if n_act and act_positions is not None:
            with torch.no_grad():
                R_slide = self._rot6d_to_matrix(self.rot6d)
                y_dir = R_slide[:, :, 1]
                z_dir = R_slide[:, :, 2]
                tangent_dirs = [y_dir, -y_dir, z_dir, -z_dir]
                total_shift = torch.zeros(B, device=dev)
                act_joint_mask = torch.zeros(B, 16, device=dev, dtype=torch.bool)
                for b in range(B):
                    fi = self.amap[b, 0]
                    act_joint_mask[b, fi*4:fi*4+4] = True

            for co_step in range(30):
                with torch.no_grad():
                    # 1) Check palm-back + actuation finger collision
                    q_s = self._u2q(self.u)
                    bT_s = self._base_T(self.pos, self.rot6d)
                    fk_s = self.chain.forward_kinematics(q_s)
                    R_s = self._rot6d_to_matrix(self.rot6d)

                    # Palm back-side points
                    palm_pts_list = []
                    for nm, lp in self._col_data:
                        if "palm" not in nm: continue
                        if nm in fk_s:
                            wT = bT_s @ fk_s[nm].get_matrix()
                            palm_pts_list.append((wT @ lp.T)[:, :3, :].transpose(1, 2))
                    palm_pts = torch.cat(palm_pts_list, dim=1) if palm_pts_list else None

                    # Actuation finger link points (all links except ds fingertip)
                    prefixes_act = ['if', 'mf', 'rf', 'th']
                    sfx_act = [['bs', 'px', 'md']] * 3 + [['mp', 'bs', 'px']]  # exclude ds (touching object)
                    act_pts_list = []
                    for b_fi in range(4):
                        mask_fi = (self.amap_t[:, 0] == b_fi)
                        if not mask_fi.any(): continue
                        for suf in sfx_act[b_fi]:
                            nm = f"leap_{self.hand}_{prefixes_act[b_fi]}_{suf}"
                            for cnm, lp in self._col_data:
                                if cnm == nm and nm in fk_s:
                                    wT = bT_s @ fk_s[nm].get_matrix()
                                    wp = (wT @ lp.T)[:, :3, :].transpose(1, 2)
                                    # Only include for envs where this is the actuation finger
                                    act_pts_list.append((mask_fi, wp))

                    # Compute min SDF across palm-back + actuation links
                    if palm_pts is None: break
                    palm_sdf = self.sdf.query(palm_pts)
                    pts_bx = ((palm_pts - self.pos.unsqueeze(1)) * R_s[:, :, 0].unsqueeze(1)).sum(-1)
                    palm_sdf_back = torch.where(pts_bx < 0.020, palm_sdf, torch.ones_like(palm_sdf))
                    min_sdf = palm_sdf_back.min(dim=-1).values

                    # Add actuation finger collision (ignore clearance zone — that's where it goes)
                    for mask_fi, wp in act_pts_list:
                        act_sdf = self.sdf.query(wp, include_clearance=False).min(dim=-1).values  # [B]
                        # Only affect envs where this finger is actuation
                        min_sdf = torch.where(mask_fi, torch.minimum(min_sdf, act_sdf), min_sdf)

                    colliding = min_sdf < 0
                    if not colliding.any(): break

                    # 2) Slide palm: try 4 tangent directions, pick best
                    step_size = 0.002
                    best_dir = torch.zeros(B, 3, device=dev)
                    best_imp = torch.full((B,), -1e9, device=dev)
                    for td in tangent_dirs:
                        tp = self.pos.data + step_size * td
                        bT_t = self._base_T(tp, self.rot6d)
                        # Palm back SDF
                        tps = []
                        for nm, lp in self._col_data:
                            if "palm" not in nm: continue
                            if nm in fk_s:
                                tps.append((bT_t @ fk_s[nm].get_matrix() @ lp.T)[:, :3, :].transpose(1, 2))
                        tpp = torch.cat(tps, dim=1)
                        ts = self.sdf.query(tpp)
                        tbx = ((tpp - tp.unsqueeze(1)) * R_s[:, :, 0].unsqueeze(1)).sum(-1)
                        test_min = torch.where(tbx < 0.020, ts, torch.ones_like(ts)).min(-1).values
                        # Actuation finger SDF (ignore clearance — act finger belongs there)
                        for mask_fi, wp_orig in act_pts_list:
                            # wp moves with base position
                            delta = step_size * td
                            wp_shifted = wp_orig + delta.unsqueeze(1)
                            act_s = self.sdf.query(wp_shifted, include_clearance=False).min(-1).values
                            test_min = torch.where(mask_fi, torch.minimum(test_min, act_s), test_min)
                        imp = test_min - min_sdf
                        better = imp > best_imp
                        best_imp[better] = imp[better]
                        best_dir[better] = td[better]

                    can_move = colliding & (total_shift < 0.05) & (best_imp > 0)
                    self.pos.data[can_move] += step_size * best_dir[can_move]
                    total_shift[can_move] += step_size

                # 3) Joint actuation IK: simultaneously reach target AND avoid collision.
                # Previously split into sequential collision-then-snap phases that
                # fought each other. Joint optimization finds configurations where
                # the finger routes AROUND the object body to reach the target.
                u_act_ik = self.u.detach().clone().requires_grad_(True)
                opt_act_ik = torch.optim.Adam([u_act_ik], lr=0.02)
                for act_ik_step in range(30):
                    opt_act_ik.zero_grad()
                    q_ai = self._u2q(u_act_ik)
                    bT_ai = self._base_T(self.pos.detach(), self.rot6d.detach())
                    fk_ai = self.chain.forward_kinematics(q_ai)
                    loss_ai = torch.zeros(B, device=dev)
                    for b_fi in range(4):
                        mask_fi = (self.amap_t[:, 0] == b_fi)
                        if not mask_fi.any(): continue
                        # Target: fingertip on actuation point
                        nm = self.tip_link_names[b_fi]
                        wT = bT_ai @ fk_ai[nm].get_matrix()
                        off_h = torch.cat([self.tip_offsets[b_fi], torch.ones(1, device=dev)])
                        tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                        loss_ai += mask_fi.float() * 1000 * ((tp - act_pos) ** 2).sum(-1)
                        if act_dir is not None:
                            pad = -wT[:, :3, 0]
                            loss_ai += mask_fi.float() * 100 * (1.0 - (pad * act_dir).sum(-1)) ** 2
                        # Collision: penalize body links inside object (ignore clearance zone)
                        for suf in sfx_act[b_fi]:
                            lnm = f"leap_{self.hand}_{prefixes_act[b_fi]}_{suf}"
                            for cnm, lp in self._col_data:
                                if cnm == lnm and lnm in fk_ai:
                                    lwT = bT_ai @ fk_ai[lnm].get_matrix()
                                    lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                                    lsdf = self.sdf.query(lwp, include_clearance=False)
                                    loss_ai += mask_fi.float() * 500 * F.relu(-lsdf).sum(-1)
                    loss_ai.mean().backward()
                    with torch.no_grad():
                        u_act_ik.grad[~act_joint_mask] = 0.0
                    opt_act_ik.step()
                with torch.no_grad():
                    self.u = u_act_ik.detach().requires_grad_(True)

            # Report
            with torch.no_grad():
                q_f = self._u2q(self.u)
                bT_f = self._base_T(self.pos, self.rot6d)
                fk_f = self.chain.forward_kinematics(q_f)
                pp_f = []
                for nm, lp in self._col_data:
                    if "palm" not in nm: continue
                    if nm in fk_f:
                        pp_f.append((bT_f @ fk_f[nm].get_matrix() @ lp.T)[:, :3, :].transpose(1, 2))
                if pp_f:
                    ppf = torch.cat(pp_f, dim=1)
                    sf = self.sdf.query(ppf)
                    bx = ((ppf - self.pos.unsqueeze(1)) * R_s[:, :, 0].unsqueeze(1)).sum(-1)
                    sf_back = torch.where(bx < 0.020, sf, torch.ones_like(sf)).min(-1).values
                    nc = (sf_back >= 0).sum().item()
                    ns = (total_shift > 0.001).sum().item()
                    print(f"  Palm slide+reproj: {nc}/{B} back-clean, {ns} shifted")

            self._snap_metrics("S2_after_act_ik_palm_slide")
            if hasattr(self, '_stage_times') and "_t_act_ik" in self._stage_times:
                import time as _t
                if dev == "cuda": torch.cuda.synchronize()
                self._stage_times["stage_act_ik_and_slide"] = _t.time() - self._stage_times["_t_act_ik"]

        # Support finger curling + IK is done in optimize() after
        # filtering to grasps where actuation IK succeeded.

    # -- main loop --------------------------------------------------------
    def optimize(
        self,
        actuation_targets: List[Tuple[np.ndarray, Optional[np.ndarray]]],
        object_center: np.ndarray,
        steps: int = 800,
        lr: float = 0.005,
        mu: float = 0.7,
        ns: int = 4,
        save_path: Optional[str] = None,
        opt_variant: str = "P",
        trajectory_log: Optional[list] = None,
    ):
        """Optimise grasps using the FRoGGeR formulation.

        opt_variant: "P" (projected gradient descent on tip-surface, default)
                     or "A" (no projection). Other variants have been removed.

        Phase 1: Get fingertips onto the object surface (SDF-based).
        Phase 2: Maximise the min-weight metric l* for force closure while
                 keeping fingertips on the surface and avoiding collisions.

        Args:
            actuation_targets: list of (position, direction) pairs
            object_center: object center in world frame
            steps: total optimisation steps
            lr: learning rate
            mu: friction coefficient for friction cone
            ns: number of friction cone pyramid sides
            save_path: if given, save results to this .pt file
        """
        assert opt_variant in ("A", "P"), opt_variant
        n_act = len(actuation_targets)
        B, dev = self.num_envs, self.device
        # Stage timing: torch.cuda.synchronize before each time.time() so we
        # attribute GPU wait time to the right stage. Saved on self._stage_times.
        self._stage_times = {}
        def _tick(label):
            if dev == "cuda": torch.cuda.synchronize()
            self._stage_times[label] = time.time()
        def _tock(label, start_label):
            if dev == "cuda": torch.cuda.synchronize()
            # The matching _tick may not have run (e.g. a stage skipped because a
            # batch produced zero candidates); record nothing in that case.
            if start_label in self._stage_times:
                self._stage_times[label] = time.time() - self._stage_times[start_label]
        _tick("_t0")
        nc = (4 + len(self.tip_offsets) - 4) if self.palm_contact else 4  # 4 fingers + N palm pts
        m = nc * ns  # total basis wrenches

        # Compute OBB of object mesh for initialization
        verts_W = self.sdf._verts_W if hasattr(self.sdf, '_verts_W') else None
        if verts_W is None:
            # Fallback: use identity OBB axes
            self._obb_axes = torch.eye(3, device=dev)
            self._obb_lengths = torch.ones(3, device=dev)
        else:
            cov = np.cov(verts_W.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvecs = eigvecs[:, order]
            proj = verts_W @ eigvecs
            lengths = proj.max(axis=0) - proj.min(axis=0)
            self._obb_axes = torch.tensor(eigvecs, dtype=torch.float32, device=dev)
            self._obb_lengths = torch.tensor(lengths, dtype=torch.float32, device=dev)

        # Extract actuation positions and directions for biased init
        act_pos_list = [t[0] for t in actuation_targets] if n_act else None
        act_dir_list = [t[1] for t in actuation_targets] if n_act else None
        _tick("_t_init")
        self._init(object_center, n_act=n_act,
                   act_positions=act_pos_list, act_directions=act_dir_list)
        _tock("stage_init", "_t_init")
        _tick("_t_filt_supIK")

        # Precompute friction cone primitive forces
        F_prim = compute_primitive_forces_torch(ns, mu, device=dev)  # [3, ns]
        self._F_prim = F_prim
        self._ns = ns

        ap = torch.stack([torch.tensor(t[0], dtype=torch.float32, device=dev)
                          for t in actuation_targets]) if n_act else None
        # Actuation directions (normalised push direction per target)
        ad = None
        if n_act:
            ad_list = []
            for t in actuation_targets:
                if t[1] is not None:
                    d = torch.tensor(t[1], dtype=torch.float32, device=dev)
                    ad_list.append(F.normalize(d, dim=0))
                else:
                    ad_list.append(None)
            ad = ad_list  # list of (3,) tensors or None per target

        obj_c = torch.tensor(object_center, dtype=torch.float32, device=dev)
        self._obj_center = obj_c  # for palm orientation check in snapshots
        self._snapshot("after_init")
        self._snap_metrics("S1_after_init")

        # ================================================================
        # Filter to actuation-successful candidates, then support finger IK
        # ================================================================
        if n_act and hasattr(self, '_act_sort_order'):
            with torch.no_grad():
                # Stage-4 hard gate: palm and actuation finger are FIXED after
                # this point (for variant A; for variant P base translates but
                # relative geometry is fixed). All palm+act-finger constraints
                # must pass here because main opt can't fix them.
                q_check = self._u2q(self.u)
                bT_check = self._base_T(self.pos, self.rot6d)
                fk_check = self.chain.forward_kinematics(q_check)

                act_dists = torch.zeros(B, device=dev)
                # Act direction alignment per env (cos between pad and act_dir)
                act_dir_align = torch.ones(B, device=dev)
                # Worst SDF over actuation finger's non-ds links (bs, px, md).
                # ds is contact surface — expected to touch or enter clearance.
                act_link_worst = torch.zeros(B, device=dev)
                prefixes_f4 = ['if', 'mf', 'rf', 'th']
                sfx_f4 = [['bs', 'px', 'md']] * 3 + [['mp', 'bs', 'px']]
                for b_fi in range(4):
                    mask_fi = (self.amap_t[:, 0] == b_fi)
                    if not mask_fi.any(): continue
                    nm = self.tip_link_names[b_fi]
                    wT = bT_check @ fk_check[nm].get_matrix()
                    off_h = torch.cat([self.tip_offsets[b_fi], torch.ones(1, device=dev)])
                    tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                    act_dists += mask_fi.float() * torch.norm(tp - ap[0], dim=-1)
                    # Direction: pad faces -x in tip frame
                    if ad is not None and ad[0] is not None:
                        pad_dir = -wT[:, :3, 0]
                        cos_a = (pad_dir * ad[0].unsqueeze(0)).sum(-1)
                        act_dir_align = torch.where(mask_fi, cos_a, act_dir_align)
                    # Non-ds link collision (exclude clearance — act finger
                    # may legitimately enter the clearance zone).
                    for suf in sfx_f4[b_fi]:
                        lnm = f"leap_{self.hand}_{prefixes_f4[b_fi]}_{suf}"
                        if lnm not in fk_check: continue
                        for cnm, lp in self._col_data:
                            if cnm != lnm: continue
                            lwT = bT_check @ fk_check[lnm].get_matrix()
                            lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                            lmin = self.sdf.query(lwp, include_clearance=False).min(-1).values
                            # Accumulate worst (most negative) per env
                            act_link_worst = torch.where(
                                mask_fi & (lmin < act_link_worst), lmin, act_link_worst)

                # Palm check — includes clearance (palm MUST NOT enter the
                # actuation clearance zone; that's support/act-finger territory).
                worst_palm_init = torch.zeros(B, device=dev)
                for cnm, lp in self._col_data:
                    if "palm" not in cnm or cnm not in fk_check: continue
                    lwT = bT_check @ fk_check[cnm].get_matrix()
                    lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                    worst_palm_init = torch.minimum(
                        worst_palm_init, self.sdf.query(lwp).min(-1).values)

                # Hard gate: every palm + actuation constraint that main opt
                # cannot fix. Thresholds:
                #   act_dist < 10mm     (position reach)
                #   act_dir_align > 0.80 (≤37° misalignment — grasp unusable otherwise)
                #   act_link_worst > -3mm (non-ds finger not buried in object)
                #   worst_palm_init > -3mm (palm not inside object or clearance)
                pass_pos   = act_dists < 0.010
                pass_dir   = act_dir_align > 0.80
                pass_acol  = act_link_worst > -0.003
                pass_palm  = worst_palm_init > -0.003
                good = pass_pos & pass_dir & pass_acol & pass_palm
                n_good = good.sum().item()
                n_act_ok = pass_pos.sum().item()
                n_dir_fail = (pass_pos & ~pass_dir).sum().item()
                n_acol_fail = (pass_pos & pass_dir & ~pass_acol).sum().item()
                n_palm_fail = (pass_pos & pass_dir & pass_acol & ~pass_palm).sum().item()
                print(f"  Actuation filter: {n_good}/{B} passed "
                      f"({n_act_ok} pos-ok; -{n_dir_fail} dir, -{n_acol_fail} act-col, "
                      f"-{n_palm_fail} palm)")

            if n_good > 0:
                # Randomize support finger joint init across the FULL valid
                # range. Each env independently samples a starting topology,
                # which lets surface-seek + tip-tip distance + normal diversity
                # explore different basins (some envs end as body wraps, some
                # as semi-wraps, some as side grips). The previous deterministic
                # CMC pattern (act_cmc + 1.0 + si*0.7) collapsed every env to
                # the same "fingers fan out from act" topology, which on flat-
                # faced objects guaranteed all supports landing on one face.
                jl = torch.tensor(_LEAP_JOINT_LOWER, device=dev)
                jh = torch.tensor(_LEAP_JOINT_UPPER, device=dev)
                cmc_min, cmc_max = -0.3, 2.2  # full usable CMC range
                with torch.no_grad():
                    for b in range(B):
                        if not good[b]: continue
                        act_fi = self.amap[b, 0]
                        sup_fingers = [fi for fi in range(4) if fi != act_fi]
                        for fi in sup_fingers:
                            j0 = fi * 4
                            # Each joint sampled uniformly over its full
                            # valid range — no hand-tuned per-joint subranges.
                            ranges = [
                                (cmc_min, cmc_max),    # CMC: full lateral spread
                                (-0.3, 0.6),           # MCP: slight flex
                                (0.0, 1.5),            # PIP: any curl
                                (0.0, 1.3),            # DIP: any curl
                            ]
                            for ji, (lo, hi) in enumerate(ranges):
                                q_val = lo + (hi - lo) * torch.rand(1, device=dev)
                                u_val = (q_val - jl[j0+ji]) / (jh[j0+ji] - jl[j0+ji])
                                u_val = torch.log(u_val.clamp(1e-6, 1-1e-6) / (1 - u_val.clamp(1e-6, 1-1e-6)))
                                self.u.data[b, j0+ji] = u_val.item()

                # Support finger IK: minimize SDF + avoid actuation finger.
                u_sup = self.u.detach().clone().requires_grad_(True)
                opt_sup = torch.optim.Adam([u_sup], lr=0.03)

                sup_joint_mask = torch.zeros(B, 16, device=dev, dtype=torch.bool)
                sup_finger_mask = torch.zeros(B, 4, device=dev, dtype=torch.bool)
                for b in range(B):
                    if not good[b]: continue
                    act_fi = self.amap[b, 0]
                    for fi in range(4):
                        if fi != act_fi:
                            # All 4 joints (including CMC) during support IK.
                            # CMC is frozen later during optimization.
                            sup_joint_mask[b, fi*4:fi*4+4] = True
                            sup_finger_mask[b, fi] = True

                if hasattr(self, '_stage_times'):
                    import time as _t
                    if dev == "cuda": torch.cuda.synchronize()
                    self._stage_times["_t_supIK_loop"] = _t.time()
                for ik_step in range(400):
                    opt_sup.zero_grad()
                    q_ik = self._u2q(u_sup)
                    bT_ik = self._base_T(self.pos.detach(), self.rot6d.detach())
                    fk_ik = self.chain.forward_kinematics(q_ik)

                    # Mid-way re-randomize: if any support finger still overlaps
                    # actuation after 100 steps, re-roll its curl
                    if ik_step == 100:
                        with torch.no_grad():
                            sc_check = self._get_sc_points(fk_ik, bT_ik)
                            if sc_check is not None:
                                fidx_check = {}
                                soff = 0
                                for nm, pts in self._sc_data:
                                    n = pts.shape[0]
                                    for ffi in range(4):
                                        for s in FINGER_LINK_SUFFIXES[ffi]:
                                            if f'_{FINGER_PREFIXES[ffi]}_{s}' in nm:
                                                fidx_check.setdefault(ffi, []).extend(range(soff, soff+n))
                                    soff += n
                                for b in range(B):
                                    if not good[b]: continue
                                    act_fi = self.amap[b, 0]
                                    if act_fi not in fidx_check: continue
                                    act_p = sc_check[b, fidx_check[act_fi]]
                                    for fi in range(4):
                                        if fi == act_fi or fi not in fidx_check: continue
                                        sup_p = sc_check[b, fidx_check[fi]]
                                        d = torch.cdist(sup_p.unsqueeze(0), act_p.unsqueeze(0))[0].min()
                                        if d < 0.008:  # still overlapping
                                            j0 = fi * 4
                                            for ji in range(4):
                                                lo, hi = ranges[ji]
                                                qv = lo + (hi - lo) * torch.rand(1, device=dev)
                                                uv = (qv - jl[j0+ji]) / (jh[j0+ji] - jl[j0+ji])
                                                uv = torch.log(uv.clamp(1e-6, 1-1e-6) / (1-uv.clamp(1e-6, 1-1e-6)))
                                                u_sup.data[b, j0+ji] = uv.item()
                            # Re-init optimizer after re-roll
                            opt_sup = torch.optim.Adam([u_sup], lr=0.03)
                            continue

                    # Get actuation finger tip position for proximity check
                    act_tip_pos = torch.zeros(B, 3, device=dev)
                    for b_fi in range(4):
                        mf = (self.amap_t[:, 0] == b_fi)
                        if not mf.any(): continue
                        nm_a = self.tip_link_names[b_fi]
                        wT_a = bT_ik @ fk_ik[nm_a].get_matrix()
                        off_a = torch.cat([self.tip_offsets[b_fi], torch.ones(1, device=dev)])
                        act_tip_pos[mf] = (wT_a @ off_a.unsqueeze(-1)).squeeze(-1)[mf, :3]

                    loss_sup = torch.zeros(B, device=dev)
                    for fi in range(4):
                        nm = self.tip_link_names[fi]
                        wT = bT_ik @ fk_ik[nm].get_matrix()
                        off_h = torch.cat([self.tip_offsets[fi], torch.ones(1, device=dev)])
                        tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                        tp_sdf = self.sdf.query(tp.unsqueeze(1)).squeeze(1)

                        # Surface seeking: ALWAYS active (never replaced by repulsion)
                        sdf_loss = tp_sdf ** 2
                        below_obj = F.relu(self._obj_z_min - tp[:, 2]) ** 2
                        loss_sup += sup_finger_mask[:, fi].float() * (500 * sdf_loss + 2000 * below_obj)

                        # Pad orientation: only penalize pad FACING AWAY from
                        # the surface (align<0). Fine alignment (align>0.95) is
                        # main opt's job — during support IK the tip is still
                        # 10-40mm from surface and demanding tight alignment
                        # wastes gradient budget that should go to pos.
                        pad_dir = -wT[:, :3, 0]  # [B, 3] — pad push direction
                        _, inward_n = self.sdf.query_with_normals(tp.unsqueeze(1))
                        inward_n = inward_n[:, 0]  # [B, 3]
                        align = (pad_dir * inward_n).sum(-1)
                        loss_sup += sup_finger_mask[:, fi].float() * 500 * F.relu(0.95 - align) ** 2

                        # Actuation repulsion: ADDED on top of surface loss (not replacing)
                        dist_to_act_tip = torch.norm(tp - act_tip_pos, dim=-1)
                        dist_to_act_target = torch.norm(tp - ap[0], dim=-1) if ap is not None else dist_to_act_tip
                        dist_to_act = torch.minimum(dist_to_act_tip, dist_to_act_target)
                        too_close = (dist_to_act < 0.040) & sup_finger_mask[:, fi]
                        loss_sup += too_close.float() * 500 * F.relu(0.040 - dist_to_act) ** 2

                    # Support finger link collision + below-object penalty
                    for fi in range(4):
                        if not sup_finger_mask[:, fi].any(): continue
                        for suf in FINGER_LINK_SUFFIXES[fi]:
                            ln = f"leap_{self.hand}_{FINGER_PREFIXES[fi]}_{suf}"
                            # Below-object check (link origin)
                            if ln in fk_ik:
                                lp = (bT_ik @ fk_ik[ln].get_matrix())[:, :3, 3]
                                below = F.relu(self._obj_z_min - lp[:, 2]) ** 2
                                loss_sup += sup_finger_mask[:, fi].float() * 1000 * below
                            # Link collision: non-ds links penalize any penetration.
                            if suf in ('bs', 'px', 'md', 'mp'):
                                for cnm, clp in self._col_data:
                                    if cnm == ln and ln in fk_ik:
                                        lwT = bT_ik @ fk_ik[ln].get_matrix()
                                        lwp = (lwT @ clp.T)[:, :3, :].transpose(1, 2)
                                        lsdf = self.sdf.query(lwp)
                                        col_pen = F.relu(-lsdf).sum(-1)
                                        loss_sup += sup_finger_mask[:, fi].float() * 500 * col_pen
                            elif suf == 'ds':
                                # Fingertip ds: back-side only (pad wrapping is expected).
                                for ci_ik, (cnm, clp) in enumerate(self._col_data):
                                    if cnm == ln and ln in fk_ik:
                                        si_ik, ei_ik = self._col_link_ranges[ci_ik]
                                        back_m_ik = self._ds_back_mask[si_ik:ei_ik]
                                        if not back_m_ik.any(): continue
                                        lwT = bT_ik @ fk_ik[ln].get_matrix()
                                        lwp = (lwT @ clp[back_m_ik].T)[:, :3, :].transpose(1, 2)
                                        lsdf = self.sdf.query(lwp)
                                        deep_pen = F.relu(-lsdf - 0.003).max(-1).values
                                        loss_sup += sup_finger_mask[:, fi].float() * 200 * deep_pen ** 2

                    # Support vs actuation finger repulsion using SC box points.
                    # _sc_data has box keypoints per link — much better than origins.
                    sc_pts_world = self._get_sc_points(fk_ik, bT_ik)  # [B, N_sc, 3]
                    if sc_pts_world is not None:
                        # Build index mapping: which SC points belong to which finger
                        finger_sc_ranges = {}  # fi -> list of (start, end) in SC point array
                        sc_offset = 0
                        for nm, pts in self._sc_data:
                            n = pts.shape[0]
                            for fi in range(4):
                                for suf in FINGER_LINK_SUFFIXES[fi]:
                                    if f"_{FINGER_PREFIXES[fi]}_{suf}" in nm:
                                        finger_sc_ranges.setdefault(fi, []).append((sc_offset, sc_offset + n))
                            sc_offset += n

                        for fi in range(4):
                            if not sup_finger_mask[:, fi].any(): continue
                            sup_ranges = finger_sc_ranges.get(fi, [])
                            if not sup_ranges: continue
                            sup_idx = []
                            for s, e in sup_ranges:
                                sup_idx.extend(range(s, e))
                            sup_pts = sc_pts_world[:, sup_idx]  # [B, n_sup, 3]

                            # Get actuation finger SC points
                            for b_fi in range(4):
                                mask_fi = (self.amap_t[:, 0] == b_fi)
                                if not mask_fi.any(): continue
                                act_ranges = finger_sc_ranges.get(b_fi, [])
                                if not act_ranges: continue
                                act_idx = []
                                for s, e in act_ranges:
                                    act_idx.extend(range(s, e))
                                act_pts = sc_pts_world[:, act_idx]  # [B, n_act, 3]

                                # Sum of penalties across ALL close pairs (not just min)
                                dists = torch.cdist(sup_pts, act_pts)  # [B, n_sup, n_act]
                                pair_pen = F.relu(0.020 - dists) ** 2  # [B, n_sup, n_act]
                                repulsion = pair_pen.sum(dim=(-2, -1))  # [B]
                                loss_sup += (sup_finger_mask[:, fi] & mask_fi).float() * 50 * repulsion

                    # Spread support fingertips apart + diverse normals
                    sup_tips = []
                    sup_normals = []
                    for fi in range(4):
                        nm = self.tip_link_names[fi]
                        wT = bT_ik @ fk_ik[nm].get_matrix()
                        off_h = torch.cat([self.tip_offsets[fi], torch.ones(1, device=dev)])
                        tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                        sup_tips.append(tp)
                        # Tip SDF normal (surface normal where the tip touches)
                        _, tn = self.sdf.query_with_normals(tp.unsqueeze(1))
                        sup_normals.append(tn[:, 0])  # inward normal
                    sup_tips = torch.stack(sup_tips, dim=1)  # [B, 4, 3]
                    sup_normals = torch.stack(sup_normals, dim=1)  # [B, 4, 3]

                    # Pairwise tip distance: penalize support tips being too close
                    for fi in range(4):
                        for fj in range(fi + 1, 4):
                            both_sup = sup_finger_mask[:, fi] & sup_finger_mask[:, fj]
                            if not both_sup.any(): continue
                            td = torch.norm(sup_tips[:, fi] - sup_tips[:, fj], dim=-1)
                            loss_sup += both_sup.float() * 300 * F.relu(0.04 - td) ** 2

                    # Normal diversity: penalize parallel normals between support fingers
                    for fi in range(4):
                        for fj in range(fi + 1, 4):
                            both_sup = sup_finger_mask[:, fi] & sup_finger_mask[:, fj]
                            if not both_sup.any(): continue
                            ndot = (sup_normals[:, fi] * sup_normals[:, fj]).sum(-1)
                            # Penalize high similarity (dot > 0.5 = within 60°)
                            loss_sup += both_sup.float() * 30 * F.relu(ndot - 0.3) ** 2

                    # Don't change actuation finger or non-good envs
                    act_reg = ((u_sup - self.u.detach()) ** 2 * (~sup_joint_mask).float()).sum(-1)
                    loss_sup += 200 * act_reg

                    loss_sup.mean().backward()
                    with torch.no_grad():
                        u_sup.grad[~sup_joint_mask] = 0.0
                    opt_sup.step()

                if hasattr(self, '_stage_times') and "_t_supIK_loop" in self._stage_times:
                    import time as _t
                    if dev == "cuda": torch.cuda.synchronize()
                    self._stage_times["stage_supIK_loop"] = _t.time() - self._stage_times["_t_supIK_loop"]
                with torch.no_grad():
                    self.u = u_sup.detach().requires_grad_(True)

                # Report
                with torch.no_grad():
                    q_f = self._u2q(self.u)
                    bT_f = self._base_T(self.pos, self.rot6d)
                    fk_f = self.chain.forward_kinematics(q_f)
                    sup_sdfs = []
                    for fi in range(4):
                        nm = self.tip_link_names[fi]
                        wT = bT_f @ fk_f[nm].get_matrix()
                        off_h = torch.cat([self.tip_offsets[fi], torch.ones(1, device=dev)])
                        tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                        sup_sdfs.append(self.sdf.query(tp.unsqueeze(1)).squeeze(1).abs())
                    all_sdf = torch.stack(sup_sdfs, dim=1)
                    mean_sdf = all_sdf[good].mean(-1) if good.any() else torch.zeros(1)
                    n_touch = (mean_sdf < 0.005).sum().item()
                    print(f"  Support IK ({n_good} candidates): {n_touch} with mean SDF < 5mm, "
                          f"median={mean_sdf.median()*1000:.1f}mm")

        self._snapshot("after_support_ik")
        self._snap_metrics("S3_after_support_ik")

        # ================================================================
        # Optimization: improve grasp quality on filtered candidates
        # Frozen: palm pose, actuation finger. Optimized: 3 support fingers.
        # ================================================================
        if n_act and hasattr(self, '_act_sort_order'):
            with torch.no_grad():
                # Filter: actuation < 10mm, mean tip SDF < 15mm
                q_filt = self._u2q(self.u)
                bT_filt = self._base_T(self.pos, self.rot6d)
                fk_filt = self.chain.forward_kinematics(q_filt)
                act_d = torch.zeros(B, device=dev)
                tip_sdf_mean = torch.zeros(B, device=dev)
                for b_fi in range(4):
                    mask_fi = (self.amap_t[:, 0] == b_fi)
                    if not mask_fi.any(): continue
                    nm = self.tip_link_names[b_fi]
                    wT = bT_filt @ fk_filt[nm].get_matrix()
                    off_h = torch.cat([self.tip_offsets[b_fi], torch.ones(1, device=dev)])
                    tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                    act_d += mask_fi.float() * torch.norm(tp - ap[0], dim=-1)
                for fi in range(4):
                    nm = self.tip_link_names[fi]
                    wT = bT_filt @ fk_filt[nm].get_matrix()
                    off_h = torch.cat([self.tip_offsets[fi], torch.ones(1, device=dev)])
                    tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                    s = self.sdf.query(tp.unsqueeze(1)).squeeze(1).abs()
                    tip_sdf_mean += s / 4
                # Also check: no support link deeper than -5mm
                worst_sup_link = torch.zeros(B, device=dev)
                for fi in range(4):
                    for ci, (cnm, lp) in enumerate(self._col_data):
                        is_sup = False
                        for suf in (['bs','px','md'] if fi < 3 else ['mp','bs','px']):
                            if cnm == f"leap_{self.hand}_{['if','mf','rf','th'][fi]}_{suf}":
                                is_sup = True
                        if not is_sup or cnm not in fk_filt: continue
                        lwT = bT_filt @ fk_filt[cnm].get_matrix()
                        lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                        lsdf = self.sdf.query(lwp).min(-1).values
                        # Only for envs where this is a support finger
                        for b in range(B):
                            if self.amap[b, 0] != fi:
                                worst_sup_link[b] = torch.minimum(worst_sup_link[b], lsdf[b])

                # Also check actuation finger link collision (not just support).
                # Use include_clearance=False: the actuation finger legitimately
                # enters the clearance cylinder to reach the trigger. Previously
                # we used the default (include_clearance=True), which treated
                # act-finger-in-clearance as collision, over-filtering hot_glue
                # et al. where the thumb reaches deep into the trigger recess.
                worst_act_link = torch.zeros(B, device=dev)
                for fi in range(4):
                    for cnm, lp in self._col_data:
                        is_act_link = False
                        for suf in (['bs','px','md'] if fi < 3 else ['mp','bs','px']):
                            if cnm == f"leap_{self.hand}_{['if','mf','rf','th'][fi]}_{suf}":
                                is_act_link = True
                        if not is_act_link or cnm not in fk_filt: continue
                        lwT = bT_filt @ fk_filt[cnm].get_matrix()
                        lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                        lsdf = self.sdf.query(lwp, include_clearance=False).min(-1).values
                        for b in range(B):
                            if self.amap[b, 0] == fi:
                                worst_act_link[b] = torch.minimum(worst_act_link[b], lsdf[b])

                opt_mask = ((act_d < 0.010) & (tip_sdf_mean < 0.015)
                           & (worst_sup_link > -0.005) & (worst_act_link > -0.010))
                n_opt = opt_mask.sum().item()
                n_act_filtered = ((worst_act_link <= -0.010) & (act_d < 0.010)).sum().item()
                # SC pre-filter: envs whose support fingers already self-collide
                # by more than 10 mm at the end of support IK cannot recover
                # SC > -1 mm in main opt (the SC loss is structurally weak; see
                # project_sc_loss_inert.md). Filter them out before optimization
                # to avoid spending compute on lost causes and to keep the
                # ranking pool clean.
                n_sc_filtered = 0
                if hasattr(self, '_metrics_log'):
                    s3 = self._metrics_log.get('S3_after_support_ik')
                    if s3 is not None and 'sc_worst' in s3:
                        sc_pass = torch.as_tensor(s3['sc_worst'], device=dev) > -0.010
                        before = opt_mask.sum().item()
                        opt_mask = opt_mask & sc_pass
                        n_sc_filtered = before - opt_mask.sum().item()
                        n_opt = opt_mask.sum().item()
                print(f"  Optimization candidates: {n_opt}/{B} ({n_act_filtered} filtered by act collision"
                      + (f", {n_sc_filtered} filtered by sc<-10mm" if n_sc_filtered else "") + ")")

            if n_opt > 0:
                # Build support masks
                opt_joint_mask = torch.zeros(B, 16, device=dev, dtype=torch.bool)
                sup_finger_mask_opt = torch.zeros(B, 4, device=dev, dtype=torch.bool)
                for b in range(B):
                    if not opt_mask[b]: continue
                    act_fi = self.amap[b, 0]
                    for fi in range(4):
                        if fi != act_fi:
                            if opt_variant == "P":
                                # Projected gradient descent: CMC free.
                                # Hard surface projection after each Adam step
                                # prevents the lateral drift seen without projection.
                                opt_joint_mask[b, fi*4:fi*4+4] = True
                            else:
                                # MCP, PIP, DIP only. CMC stays frozen: unfreezing
                                # let fingers drift away (surf 20-47mm everywhere
                                # in ablation). CMC spread is fixed by init.
                                opt_joint_mask[b, fi*4+1:fi*4+4] = True
                            sup_finger_mask_opt[b, fi] = True

                prefixes_opt = ['if', 'mf', 'rf', 'th']
                sfx_no_ds = [['bs', 'px', 'md']] * 3 + [['mp', 'bs', 'px']]

                # Direct-q parameterization: optimize joint angles directly with clamp projection.
                # Eliminates sigmoid gradient compression (14x weaker at joint limits).
                q_opt = self._u2q(self.u).detach().clone().requires_grad_(True)
                q_init = q_opt.detach().clone()  # for frozen-joint regularization

                # Pre-compute support finger collision link names
                prefixes_opt = ['if', 'mf', 'rf', 'th']
                sfx_no_ds = [['bs', 'px', 'md']] * 3 + [['mp', 'bs', 'px']]

                # Pre-compute: which _col_data entries belong to each support finger
                sup_col_idx = {}  # fi -> list of (col_data_index, link_name)
                for fi in range(4):
                    sup_col_idx[fi] = []
                    for ci, (cnm, lp) in enumerate(self._col_data):
                        for suf in sfx_no_ds[fi]:
                            if cnm == f"leap_{self.hand}_{prefixes_opt[fi]}_{suf}":
                                sup_col_idx[fi].append((ci, cnm))

                # ============================================================
                # Optimization loop: dispatch based on opt_variant
                # "PGD" = original projected gradient descent
                # "A"   = soft penalty Adam (sections A+B+C+D)
                # "B"   = min-k unified surface/collision
                # "C"   = min-k with adaptive FC contacts
                # ============================================================
                opt_steps = min(steps, 300)
                eps_fd = 5e-4
                fd_ofst = torch.zeros(3, 3, device=dev)
                for d3 in range(3): fd_ofst[d3, d3] = eps_fd
                sigma = torch.zeros(B, device=dev)

                # Pre-compute ds (fingertip) collision indices for variants B/C
                sup_col_idx_ds = {}  # fi -> list of (col_data_index, link_name) for ds links
                for fi in range(4):
                    sup_col_idx_ds[fi] = []
                    ds_suf = 'ds'
                    cnm_ds = f"leap_{self.hand}_{prefixes_opt[fi]}_{ds_suf}"
                    for ci, (cnm, lp) in enumerate(self._col_data):
                        if cnm == cnm_ds:
                            sup_col_idx_ds[fi].append((ci, cnm))

                if opt_variant in ("A", "P"):
                    # ── Adam loop (variants A and P) ──
                    # Variant P additionally projects onto the surface constraint
                    # after each Adam step (hard constraint on fingertip contact)
                    # AND unfreezes base pose (translation+rotation) so the whole
                    # hand can translate/rotate to resolve collision, SC and
                    # act_dist close-misses that joint-only opt can't budge.
                    if opt_variant == "P" and not _DISABLE_BASE_UNFREEZE:
                        pos_opt = self.pos.detach().clone().requires_grad_(True)
                        rot6d_opt = self.rot6d.detach().clone().requires_grad_(True)
                        pos_init = pos_opt.detach().clone()
                        rot6d_init = rot6d_opt.detach().clone()
                        # Base pose LRs chosen so a full 300-step optimisation
                        # can move the base at most a few cm / few degrees
                        # (hard-capped by the regularizer, not LR alone).
                        opt_adam = torch.optim.Adam([
                            {"params": [q_opt], "lr": 0.003},
                            {"params": [pos_opt], "lr": 0.0003},    # sub-mm / step
                            {"params": [rot6d_opt], "lr": 0.0005},  # ~0.03° / step
                        ])
                    else:
                        pos_opt = None
                        rot6d_opt = None
                        opt_adam = torch.optim.Adam([q_opt], lr=0.003)
                    fc_start_step = 50
                    fc_weight = 1.0
                    mink_k = 10  # number of lowest SDF points for variants B/C
                    print(f"  Variant {opt_variant} Adam optimization ({opt_steps} steps, direct-q)")
                    _tock("stage_filt_supIK", "_t_filt_supIK")
                    _tick("_t_mainopt")
                    # Record q_opt into self.u so snap_metrics sees the right state
                    with torch.no_grad():
                        self.u = self._q2u(q_opt.detach()).requires_grad_(True)
                    self._snap_metrics("S4_opt_step0")

                    for opt_step in range(opt_steps):
                        opt_adam.zero_grad()
                        q_o = q_opt
                        if opt_variant == "P" and pos_opt is not None:
                            bT_o = self._base_T(pos_opt, rot6d_opt)
                        else:
                            bT_o = self._base_T(self.pos.detach(), self.rot6d.detach())
                        fk_o = self.chain.forward_kinematics(q_o)

                        total_loss = torch.zeros(B, device=dev)

                        # ── Section A: Surface loss (multi-point pad contact) ──
                        # Single-point tip_offset caused pad to angle INTO surfaces
                        # (contact center at SDF=0 but pad tip 20mm inside). Now use
                        # 3 points on the pad line: heel, center, tip. Contact center
                        # must be at SDF=0; heel/tip must be OUTSIDE (SDF>=0). This
                        # enforces the pad to lie flat against the surface.
                        if opt_variant in ("A", "P"):
                            for fi in range(4):
                                if not sup_finger_mask_opt[:, fi].any(): continue
                                nm = self.tip_link_names[fi]
                                wT = bT_o @ fk_o[nm].get_matrix()
                                # Center contact point (existing tip_offset)
                                off_h = torch.cat([self.tip_offsets[fi], torch.ones(1, device=dev)])
                                tip = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                                tip_sdf = self.sdf.query(tip.unsqueeze(1)).squeeze(1)
                                # Huber: pulls center to surface (SDF=0)
                                tip_sdf_abs = tip_sdf.abs()
                                surf_loss = torch.where(tip_sdf_abs > 0.003,
                                    tip_sdf_abs - 0.0015,
                                    tip_sdf ** 2 / 0.006)
                                total_loss += sup_finger_mask_opt[:, fi].float() * 1000 * surf_loss

                                # Pad alignment: pad face (-X) aligns with surface inward normal.
                                # Tight threshold 0.95 (cos 18°); 45° slop left 5-8mm peripheral pen.
                                pad_dir = -wT[:, :3, 0]
                                _, inward_n = self.sdf.query_with_normals(tip.unsqueeze(1))
                                inward_n = inward_n[:, 0]
                                align = (pad_dir * inward_n).sum(-1)
                                total_loss += sup_finger_mask_opt[:, fi].float() * 500 * F.relu(0.95 - align) ** 2

                                # Pad corners: penalize only DEEP pad penetration (>3mm).
                                # Shallow penetration (<3mm) is acceptable — pad in contact with
                                # surface naturally has small SDF variation. Heavy penalty on
                                # all pad points caused the optimizer to LIFT the pad entirely
                                # (surf=21mm with pad=0mm — pad held above surface avoiding penalty).
                                center_off = self.tip_offsets[fi]
                                pad_corners = [
                                    torch.tensor([0.0,  0.017,  0.017], device=dev),
                                    torch.tensor([0.0,  0.017, -0.017], device=dev),
                                    torch.tensor([0.0, -0.017,  0.017], device=dev),
                                    torch.tensor([0.0, -0.017, -0.017], device=dev),
                                    torch.tensor([0.0,  0.017,  0.0],   device=dev),
                                    torch.tensor([0.0, -0.017,  0.0],   device=dev),
                                    torch.tensor([0.0,  0.0,    0.017], device=dev),
                                    torch.tensor([0.0,  0.0,   -0.017], device=dev),
                                ]
                                for delta in pad_corners:
                                    ph = torch.cat([center_off + delta, torch.ones(1, device=dev)])
                                    pw = (wT @ ph.unsqueeze(-1)).squeeze(-1)[:, :3]
                                    p_sdf = self.sdf.query(pw.unsqueeze(1)).squeeze(1)
                                    # Only penalize beyond 3mm (preserve -5mm feasibility margin).
                                    # Quadratic for smooth gradient when deep.
                                    deep_pen = F.relu(-p_sdf - 0.003)
                                    total_loss += sup_finger_mask_opt[:, fi].float() * 2000 * deep_pen ** 2

                        # ── Section B: Collision loss ──
                        if opt_variant in ("A", "P"):
                            # B1: relu(-SDF) for non-ds links
                            for fi in range(4):
                                if not sup_finger_mask_opt[:, fi].any(): continue
                                for ci, cnm in sup_col_idx[fi]:
                                    lp = self._col_data[ci][1]
                                    if cnm in fk_o:
                                        lwT = bT_o @ fk_o[cnm].get_matrix()
                                        lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                                        lsdf = self.sdf.query(lwp)
                                        total_loss += sup_finger_mask_opt[:, fi].float() * 500 * F.relu(-lsdf).sum(-1)

                            # B2: Fingertip ds collision — back-side strict, pad-side tolerant.
                            # Back: 1mm contact allowed, heavy penalty beyond.
                            # Pad: 3mm wrapping allowed on curved surfaces, penalty beyond.
                            for fi in range(4):
                                if not sup_finger_mask_opt[:, fi].any(): continue
                                for ci, cnm in sup_col_idx_ds[fi]:
                                    lp = self._col_data[ci][1]
                                    if cnm not in fk_o: continue
                                    si, ei = self._col_link_ranges[ci]
                                    back_m = self._ds_back_mask[si:ei]
                                    lwT = bT_o @ fk_o[cnm].get_matrix()
                                    # Back-side: strict (1mm contact threshold)
                                    if back_m.any():
                                        lwp_b = (lwT @ lp[back_m].T)[:, :3, :].transpose(1, 2)
                                        lsdf_b = self.sdf.query(lwp_b)
                                        pen_back = F.relu(-lsdf_b - 0.001).max(-1).values
                                        total_loss += sup_finger_mask_opt[:, fi].float() * 1000 * pen_back ** 2
                                    # Pad-side: tolerant (3mm wrap allowance),
                                    # LINEAR penalty past the margin for strong
                                    # gradient at borderline cases. Previous
                                    # quadratic form was 500×0.002²=0.002 at
                                    # 2mm excess — too weak to push many
                                    # borderline grasps below the feasibility
                                    # threshold (-5mm). Linear keeps gradient
                                    # magnitude constant so 0-10mm excess all
                                    # get pushed out equally hard.
                                    if (~back_m).any():
                                        lwp_p = (lwT @ lp[~back_m].T)[:, :3, :].transpose(1, 2)
                                        lsdf_p = self.sdf.query(lwp_p)
                                        pen_pad = F.relu(-lsdf_p - 0.003).max(-1).values
                                        total_loss += sup_finger_mask_opt[:, fi].float() * 1000 * pen_pad

                        # ── Section C: Force closure (σ_min) ──
                        if opt_step >= fc_start_step:
                            all_tips = torch.stack([
                                (bT_o @ fk_o[self.tip_link_names[fi]].get_matrix()
                                 @ torch.cat([self.tip_offsets[fi], torch.ones(1, device=dev)]).unsqueeze(-1)
                                ).squeeze(-1)[:, :3]
                                for fi in range(4)], dim=1)
                            palm_pt = None
                            if self.palm_contact and self.palm_link in fk_o:
                                wT_palm = bT_o @ fk_o[self.palm_link].get_matrix()
                                palm_oh = torch.cat([self.palm_offset, torch.ones(1, device=dev)])
                                palm_pt = (wT_palm @ palm_oh.unsqueeze(-1)).squeeze(-1)[:, :3]
                            palm_gap = (self.sdf.query(palm_pt.unsqueeze(1),
                                        include_clearance=False, include_floor=False).squeeze(1).detach()
                                        if palm_pt is not None else None)

                            for act_fi in range(4):
                                group_mask = opt_mask & (self.amap_t[:, 0] == act_fi)
                                if not group_mask.any(): continue
                                sup_fi = [fi for fi in range(4) if fi != act_fi]

                                support_tp = torch.stack([all_tips[:, fi] for fi in sup_fi], dim=1)
                                for contact_mask, tp_fc, _ in split_force_closure_contacts(
                                        support_tp, palm_pt, palm_gap, group_mask):
                                    if not contact_mask.any(): continue
                                    nc_fc = tp_fc.shape[1]
                                    def object_sdf(pts):
                                        return self.sdf.query(pts, include_clearance=False, include_floor=False)
                                    gx = (object_sdf(tp_fc+fd_ofst[0])-object_sdf(tp_fc-fd_ofst[0]))/(2*eps_fd)
                                    gy = (object_sdf(tp_fc+fd_ofst[1])-object_sdf(tp_fc-fd_ofst[1]))/(2*eps_fd)
                                    gz = (object_sdf(tp_fc+fd_ofst[2])-object_sdf(tp_fc-fd_ofst[2]))/(2*eps_fd)
                                    tn = -torch.stack([gx,gy,gz],dim=-1)
                                    tn = tn / tn.norm(dim=-1,keepdim=True).clamp(min=1e-8)

                                    # FC proxy and LP use the same physically present contacts.
                                    fc_proxy = compute_fc_proxy_loss(tp_fc, tn, obj_c)
                                    total_loss += contact_mask.float() * fc_weight * 2.0 * fc_proxy

                                    g_OCs = compute_contact_frames(tp_fc, tn)
                                    G = compute_grasp_matrix_torch(g_OCs)
                                    W = compute_wrench_matrix(G, F_prim, nc_fc, ns)
                                    s = torch.linalg.svdvals(W)[:, -1]
                                    sigma[contact_mask] = s[contact_mask].detach()
                                    total_loss += contact_mask.float() * fc_weight * 0.5 * (-s)

                                    _lp_step = opt_step - fc_start_step
                                    if _lp_step < 50:
                                        _lp_freq = 20
                                    elif _lp_step < 150:
                                        _lp_freq = 5
                                    else:
                                        _lp_freq = 10
                                    if opt_step % _lp_freq == 0:
                                        with torch.no_grad():
                                            W_np = W.detach().cpu().numpy()
                                            lp_mask_np = contact_mask.cpu().numpy() & (s.detach().cpu().numpy() > 0.01)
                                            ls, al, la, nu, lp_ok = solve_min_weight_lp_batch(
                                                W_np, env_mask=lp_mask_np, return_status=True)
                                            dl_dW = min_weight_gradient_batch(
                                                W_np, ls, al, la, nu, device=dev,
                                                valid_mask=lp_ok)
                                        lstar_loss = -(dl_dW * W).sum(dim=(1, 2))
                                        valid = torch.as_tensor(lp_ok, device=dev)
                                        total_loss += (contact_mask & valid).float() * fc_weight * 3.0 * lstar_loss

                        # ── Section D: Inter-finger self-collision (box-box SDF) ──
                        # Use box-box SDF directly — matches the feasibility metric.
                        # Only process opt_mask envs and skip distant pairs for efficiency.
                        # Computed every 5 steps to amortize cost.
                        if hasattr(self, '_box_primitives') and opt_step % 5 == 0:
                            _prefix_d = f"leap_{self.hand}_"
                            _adj_d = {
                                ('palm','if_bs'),('palm','mf_bs'),('palm','rf_bs'),('palm','th_mp'),
                                ('if_bs','if_px'),('if_px','if_md'),('if_md','if_ds'),
                                ('mf_bs','mf_px'),('mf_px','mf_md'),('mf_md','mf_ds'),
                                ('rf_bs','rf_px'),('rf_px','rf_md'),('rf_md','rf_ds'),
                                ('th_mp','th_bs'),('th_bs','th_px'),('th_px','th_ds')}
                            # Only process opt_mask envs to save memory
                            opt_idx_d = torch.where(opt_mask)[0]
                            B_d = len(opt_idx_d)
                            if B_d > 0:
                                bT_d = bT_o[opt_idx_d]
                                fk_d_matrices = {}
                                for nm_d in self._box_primitives:
                                    if nm_d in fk_o:
                                        fk_d_matrices[nm_d] = fk_o[nm_d].get_matrix()[opt_idx_d]
                                _link_wT_d = {}
                                for nm_d in self._box_primitives:
                                    if nm_d in fk_d_matrices:
                                        _link_wT_d[nm_d] = bT_d @ fk_d_matrices[nm_d]
                                sc_loss_d = torch.zeros(B_d, device=dev)
                                for nm_i in self._box_primitives:
                                    if nm_i not in _link_wT_d: continue
                                    for nm_j in self._box_primitives:
                                        if nm_j not in _link_wT_d or nm_i >= nm_j: continue
                                        si_d = nm_i.replace(_prefix_d, '')
                                        sj_d = nm_j.replace(_prefix_d, '')
                                        if (si_d, sj_d) in _adj_d or (sj_d, si_d) in _adj_d: continue
                                        fi_d = si_d.split('_')[0]; fj_d = sj_d.split('_')[0]
                                        if fi_d == fj_d and fi_d != 'palm': continue
                                        for bi_c, bi_r, bi_h in self._box_primitives[nm_i]:
                                            for bj_c, bj_r, bj_h in self._box_primitives[nm_j]:
                                                ci_w = (_link_wT_d[nm_i] @ bi_c.unsqueeze(-1)).squeeze(-1)[:, :3]
                                                ri_w = _link_wT_d[nm_i][:, :3, :3] @ bi_r.unsqueeze(0)
                                                cj_w = (_link_wT_d[nm_j] @ bj_c.unsqueeze(-1)).squeeze(-1)[:, :3]
                                                rj_w = _link_wT_d[nm_j][:, :3, :3] @ bj_r.unsqueeze(0)
                                                # Skip distant pairs (centers > 40mm apart)
                                                cdist = (ci_w - cj_w).norm(dim=-1)
                                                close = cdist < 0.040
                                                if not close.any(): continue
                                                sd = box_box_sdf_batch(
                                                    ci_w, ri_w, bi_h.unsqueeze(0).expand(B_d, -1),
                                                    cj_w, rj_w, bj_h.unsqueeze(0).expand(B_d, -1))
                                                overlap = F.relu(-sd - 0.001)
                                                sc_loss_d += 5000 * overlap ** 2
                                # Scatter back to full batch
                                sc_loss_full = torch.zeros(B, device=dev)
                                sc_loss_full[opt_idx_d] = sc_loss_d
                                total_loss += sc_loss_full

                        # ── Section E: Actuation area exclusion ──
                        # Support tips within 35mm of actuation FINGER get pushed away
                        # (uses actual FK position, not the target point on the object)
                        if n_act:
                            # Compute actual actuation finger tip position per env
                            act_tip_actual = torch.zeros(B, 3, device=dev)
                            for act_fi in range(4):
                                amask = (self.amap_t[:, 0] == act_fi)
                                if not amask.any(): continue
                                anm = self.tip_link_names[act_fi]
                                awT = bT_o @ fk_o[anm].get_matrix()
                                aoff = torch.cat([self.tip_offsets[act_fi], torch.ones(1, device=dev)])
                                atp = (awT @ aoff.unsqueeze(-1)).squeeze(-1)[:, :3]
                                act_tip_actual[amask] = atp[amask]
                            # Push support tips away from actuation finger
                            for fi in range(4):
                                if not sup_finger_mask_opt[:, fi].any(): continue
                                nm = self.tip_link_names[fi]
                                wT = bT_o @ fk_o[nm].get_matrix()
                                off_h = torch.cat([self.tip_offsets[fi], torch.ones(1, device=dev)])
                                tip = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                                dist_to_act = torch.norm(tip - act_tip_actual, dim=-1)
                                total_loss += sup_finger_mask_opt[:, fi].float() * 200 * F.relu(0.035 - dist_to_act) ** 2

                        # Section F: Support-in-clearance penalty. Uses MAX
                        # over points (matches feasibility's worst-point check).
                        # Mean was too weak: only a few points out of ~400/link
                        # penetrate the clearance, mean(-1) dilutes signal to
                        # negligible. Max gives the gradient a clear target.
                        if hasattr(self.sdf, '_clearance_center'):
                            for fi in range(4):
                                if not sup_finger_mask_opt[:, fi].any(): continue
                                for ci, cnm in sup_col_idx[fi]:
                                    lp = self._col_data[ci][1]
                                    if cnm not in fk_o: continue
                                    lwT = bT_o @ fk_o[cnm].get_matrix()
                                    lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                                    cl_sdf = self.sdf._clearance_sdf(lwp)
                                    worst_pen = F.relu(-cl_sdf).max(-1).values
                                    total_loss += sup_finger_mask_opt[:, fi].float() * 2000 * worst_pen
                                for ci, cnm in sup_col_idx_ds[fi]:
                                    lp = self._col_data[ci][1]
                                    if cnm not in fk_o: continue
                                    lwT = bT_o @ fk_o[cnm].get_matrix()
                                    lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                                    cl_sdf = self.sdf._clearance_sdf(lwp)
                                    worst_pen = F.relu(-cl_sdf).max(-1).values
                                    total_loss += sup_finger_mask_opt[:, fi].float() * 2000 * worst_pen

                        # Freeze non-support joints (regularize back to initial values)
                        total_loss += 100 * ((q_opt - q_init) ** 2 * (~opt_joint_mask).float()).sum(-1)

                        if opt_variant == "P" and pos_opt is not None:
                            # Base losses apply ONLY to opt_mask envs. Otherwise
                            # the Adam optimiser happily drifts the base pose for
                            # inactive envs (their non-masked gradients still land
                            # on pos_opt/rot6d_opt since those are shared leaves).
                            om_f = opt_mask.float()
                            # Base regularizer: strong spring toward init pose.
                            total_loss += om_f * 5000 * ((pos_opt - pos_init) ** 2).sum(-1)
                            total_loss += om_f * 5000 * ((rot6d_opt - rot6d_init) ** 2).sum(-1)

                            # Act-tip tracking: base unfreeze moves the hand
                            # rigidly; without this, the act tip drifts off
                            # the target and the grasp fails the feasibility
                            # gate (act_dist<10mm). Flashlight observed drift
                            # 15-40mm even with quadratic-5000 penalty —
                            # quadratic's gradient is too weak when far.
                            # Use hinge: small quadratic pull everywhere for
                            # smooth gradient + strong linear penalty past
                            # 10mm to pull hand back below the gate threshold.
                            if n_act and ap is not None:
                                act_tip_w = torch.zeros(B, 3, device=dev)
                                for act_fi in range(4):
                                    amask = (self.amap_t[:, 0] == act_fi)
                                    if not amask.any(): continue
                                    anm = self.tip_link_names[act_fi]
                                    awT = bT_o @ fk_o[anm].get_matrix()
                                    aoff_h = torch.cat([self.tip_offsets[act_fi], torch.ones(1, device=dev)])
                                    atp = (awT @ aoff_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                                    act_tip_w = torch.where(amask.unsqueeze(-1), atp, act_tip_w)
                                act_dist_world = torch.norm(act_tip_w - ap[0].unsqueeze(0), dim=-1)
                                # Quadratic near target (weak, smooth)
                                total_loss += om_f * 1000 * act_dist_world ** 2
                                # Hinge: linear penalty past 10mm threshold
                                excess = F.relu(act_dist_world - 0.010)
                                total_loss += om_f * 10000 * excess

                            # Palm collision: previously palm was frozen after
                            # stage-4 gate so no main-opt loss tracked it. With
                            # base unfrozen the palm can now move — guard it.
                            palm_nm = self.palm_link
                            if palm_nm in fk_o:
                                for ci, (cnm, lp) in enumerate(self._col_data):
                                    if cnm != palm_nm: continue
                                    lwT = bT_o @ fk_o[cnm].get_matrix()
                                    lwp = (lwT @ lp.T)[:, :3, :].transpose(1, 2)
                                    lsdf = self.sdf.query(lwp, include_clearance=False)
                                    # Heavy penalty: palm-through-object is the
                                    # most egregious failure the user called out.
                                    total_loss += om_f * 2000 * F.relu(-lsdf).max(-1).values ** 2

                        total_loss.mean().backward()
                        with torch.no_grad():
                            q_opt.grad[~opt_joint_mask] = 0.0
                        opt_adam.step()
                        # Project onto joint limits
                        with torch.no_grad():
                            q_opt.clamp_(self.q_lo, self.q_lo + self.q_range)

                        # ── Projection: re-attach fingertips to the object surface ──
                        # After the Adam step (which can move fingers laterally to
                        # improve FC/SC/pad objectives) we project each support tip
                        # back onto SDF=0 by running a few SGD iterations on pure
                        # surface loss. This is projected-gradient descent: hard
                        # surface constraint, soft objective. Only variant P does
                        # this — it's the whole point of P.
                        if opt_variant == "P":
                            proj_iters = 3
                            proj_lr = 0.03
                            for _proj_it in range(proj_iters):
                                q_opt.grad = None
                                if pos_opt is not None and pos_opt.grad is not None: pos_opt.grad = None
                                if rot6d_opt is not None and rot6d_opt.grad is not None: rot6d_opt.grad = None
                                # Use detached base — projection only touches joints.
                                if pos_opt is not None:
                                    bT_p = self._base_T(pos_opt.detach(), rot6d_opt.detach())
                                else:
                                    bT_p = self._base_T(self.pos.detach(), self.rot6d.detach())
                                fk_p = self.chain.forward_kinematics(q_opt)
                                proj_loss = torch.zeros(B, device=dev)
                                for fi in range(4):
                                    if not sup_finger_mask_opt[:, fi].any(): continue
                                    nm = self.tip_link_names[fi]
                                    wT = bT_p @ fk_p[nm].get_matrix()
                                    off_h = torch.cat([self.tip_offsets[fi], torch.ones(1, device=dev)])
                                    tip = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                                    tip_sdf = self.sdf.query(tip.unsqueeze(1)).squeeze(1)
                                    proj_loss += sup_finger_mask_opt[:, fi].float() * tip_sdf ** 2
                                proj_loss.mean().backward()
                                with torch.no_grad():
                                    g = q_opt.grad
                                    g[~opt_joint_mask] = 0.0
                                    q_opt.data -= proj_lr * g
                                    q_opt.clamp_(self.q_lo, self.q_lo + self.q_range)
                            q_opt.grad = None

                        # Metrics snapshot at key checkpoints
                        if opt_step in (50, 150, 299):
                            with torch.no_grad():
                                self.u = self._q2u(q_opt.detach()).requires_grad_(True)
                                if opt_variant == "P" and pos_opt is not None:
                                    self.pos = pos_opt.detach().clone().requires_grad_(True)
                                    self.rot6d = rot6d_opt.detach().clone().requires_grad_(False)
                            self._snap_metrics(f"S4_opt_step{opt_step+1}")

                        # ── Logging + trajectory ──
                        if opt_step % 50 == 0:
                            with torch.no_grad():
                                q_e = q_opt
                                if opt_variant == "P" and pos_opt is not None:
                                    bT_e = self._base_T(pos_opt, rot6d_opt)
                                else:
                                    bT_e = self._base_T(self.pos, self.rot6d)
                                fk_e = self.chain.forward_kinematics(q_e)
                                sup_sdf = []
                                for fi in range(4):
                                    nm = self.tip_link_names[fi]
                                    wT = bT_e @ fk_e[nm].get_matrix()
                                    off_h = torch.cat([self.tip_offsets[fi], torch.ones(1, device=dev)])
                                    tp = (wT @ off_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                                    sup_sdf.append(self.sdf.query(tp.unsqueeze(1)).squeeze(1).abs())
                                sup_sdf = torch.stack(sup_sdf, dim=1)
                                se = sup_sdf[opt_mask].mean().item() if opt_mask.any() else 0
                                sig_val = sigma[opt_mask].mean().item() if opt_mask.any() else 0
                                print(f"    [{opt_variant}] step {opt_step}: surf={se*1000:.1f}mm sigma={sig_val:.4f}")
                                if trajectory_log is not None:
                                    trajectory_log.append({
                                        "step": opt_step,
                                        "surface_mm": se * 1000,
                                        "sigma": sig_val,
                                    })

                _tock("stage_main_opt", "_t_mainopt")
                _tick("_t_feas")

                self._snapshot("after_optimization")

            # ==============================================================
            # Final evaluation: rank by σ_min + l* + feasibility
            # ==============================================================
            with torch.no_grad():
                q_final = self._u2q(self.u)
                bT_final = self._base_T(self.pos, self.rot6d)
                fk_final = self.chain.forward_kinematics(q_final)

                # Tip positions + collision points
                tp_final, cp_final, tip_x_final = self._get_points(fk_final, bT_final)
                ts_final = self.sdf.query(tp_final)
                # Object+floor SDF only (no clearance). Used for all object collision checks.
                cs_final = self.sdf.query(cp_final, include_clearance=False)
                # Separate clearance SDF for per-finger enforcement.
                cs_final_cl = (self.sdf._clearance_sdf(cp_final)
                               if hasattr(self.sdf, '_clearance_center')
                               else torch.full_like(cs_final, float('inf')))
                # Per-point mask: True if point belongs to the ACTUATION finger.
                # Only SUPPORT finger points are checked against clearance (they
                # must NEVER enter — stricter than object check, no margin).
                act_point_mask = torch.zeros(B, cp_final.shape[1], device=dev, dtype=torch.bool)
                prefixes_cs = ['if', 'mf', 'rf', 'th']
                for li_cs, (nm_cs, _) in enumerate(self._col_data):
                    si_cs, ei_cs = self._col_link_ranges[li_cs]
                    link_fi = next((pi for pi, p in enumerate(prefixes_cs) if f"_{p}_" in nm_cs), None)
                    if link_fi is None: continue
                    is_act_env = (self.amap_t[:, 0] == link_fi)
                    act_point_mask[:, si_cs:ei_cs] = is_act_env.unsqueeze(1)
                # Clearance penetration for support points: min across all support points.
                # "Support" = not actuation finger. Palm is also considered support.
                sup_point_mask = ~act_point_mask  # [B, N_pts]
                # For each env, find worst clearance SDF among its support points.
                # Set non-support clearance to +inf so they don't affect min.
                cl_for_sup = torch.where(sup_point_mask, cs_final_cl, torch.full_like(cs_final_cl, float('inf')))
                sup_clearance_worst = cl_for_sup.min(dim=-1).values  # [B]

                # FC from the three support fingertips and a palm point only
                # when that palm point is near the object surface.
                all_tips_final = tp_final[:, :4]  # [B, 4, 3]
                palm_pt_final = None
                if self.palm_contact and self.palm_link in fk_final:
                    wT_palm = bT_final @ fk_final[self.palm_link].get_matrix()
                    palm_oh = torch.cat([self.palm_offset, torch.ones(1, device=dev)])
                    palm_pt_final = (wT_palm @ palm_oh.unsqueeze(-1)).squeeze(-1)[:, :3]
                palm_gap_final = (self.sdf.query(palm_pt_final.unsqueeze(1),
                                  include_clearance=False, include_floor=False).squeeze(1)
                                  if palm_pt_final is not None else None)
                palm_used_all = torch.zeros(B, dtype=torch.bool, device=dev)

                eps_fd = 5e-4
                fd_o = torch.zeros(3, 3, device=dev)
                for d3 in range(3): fd_o[d3, d3] = eps_fd

                sigma_all = torch.zeros(B, device=dev)
                final_lstars = np.full(B, -1.0)
                final_lp_valid = np.zeros(B, dtype=bool)
                # Seed tp_fc with a dummy (all tips + palm) so wrap_dirs below
                # still has something to reference when no opt_mask envs exist.
                _fc_seed = [all_tips_final[:, fi] for fi in range(3)]
                if palm_pt_final is not None: _fc_seed.append(palm_pt_final)
                tp_fc = torch.stack(_fc_seed, dim=1)
                for act_fi in range(4):
                    group = opt_mask & (self.amap_t[:, 0] == act_fi)
                    if not group.any(): continue
                    sup_fi = [fi for fi in range(4) if fi != act_fi]
                    support_tp = torch.stack([all_tips_final[:, fi] for fi in sup_fi], dim=1)
                    for contact_mask, tp_fc, uses_palm in split_force_closure_contacts(
                            support_tp, palm_pt_final, palm_gap_final, group):
                        if not contact_mask.any(): continue
                        nc_fc = tp_fc.shape[1]
                        palm_used_all[contact_mask] = uses_palm
                        def object_sdf(pts):
                            return self.sdf.query(pts, include_clearance=False, include_floor=False)
                        gx = (object_sdf(tp_fc+fd_o[0])-object_sdf(tp_fc-fd_o[0]))/(2*eps_fd)
                        gy = (object_sdf(tp_fc+fd_o[1])-object_sdf(tp_fc-fd_o[1]))/(2*eps_fd)
                        gz = (object_sdf(tp_fc+fd_o[2])-object_sdf(tp_fc-fd_o[2]))/(2*eps_fd)
                        sdf_grad = torch.stack([gx, gy, gz], dim=-1)
                        tip_normals = -sdf_grad / sdf_grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                        g_OCs = compute_contact_frames(tp_fc, tip_normals)
                        G = compute_grasp_matrix_torch(g_OCs)
                        W = compute_wrench_matrix(G, F_prim, nc_fc, ns)
                        s = torch.linalg.svdvals(W)[:, -1]
                        sigma_all[contact_mask] = s[contact_mask]

                        # LPs for three and four real contacts have different
                        # matrix widths; solve each group separately.
                        W_np = W.cpu().numpy()
                        mask_np = contact_mask.cpu().numpy()
                        ls, _, _, _, lp_ok = solve_min_weight_lp_batch(
                            W_np, env_mask=mask_np, return_status=True)
                        final_lstars[mask_np] = ls[mask_np]
                        final_lp_valid[mask_np] = lp_ok[mask_np]

                final_lstars_t = torch.tensor(final_lstars, dtype=torch.float32, device=dev)
                has_lstar = (torch.as_tensor(final_lp_valid, device=dev)
                             & force_closure_margin_passes(final_lstars_t))

                # Surface error: max |SDF| across support fingers
                surf_err = ts_final[:, :4].abs().max(dim=-1).values  # [B]

                # Collision: margin-adjusted violations (exclude ds links)
                # ds links are contact surfaces — their back side penetrates
                # on curved objects by design. Only check non-ds links.
                non_ds_mask = torch.ones(cs_final.shape[1], device=dev, dtype=torch.bool)
                for li, (nm, _) in enumerate(self._col_data):
                    if "_ds" in nm:
                        si, ei = self._col_link_ranges[li]
                        non_ds_mask[si:ei] = False
                col_violation = F.relu(self._col_margins[non_ds_mask] - cs_final[:, non_ds_mask])
                max_col_viol = col_violation.max(-1).values

                # Self-collision: box-box SDF for accurate overlap detection
                # The 40-point subsample is structurally incapable of detecting
                # full link overlap. Box-box SDF is exact for box primitives.
                sc_worst_sdf = torch.full((B,), 1.0, device=dev)
                if hasattr(self, '_box_primitives'):
                    _prefix = f"leap_{self.hand}_"
                    _adj_feas = {
                        ('palm','if_bs'),('palm','mf_bs'),('palm','rf_bs'),('palm','th_mp'),
                        ('if_bs','if_px'),('if_px','if_md'),('if_md','if_ds'),
                        ('mf_bs','mf_px'),('mf_px','mf_md'),('mf_md','mf_ds'),
                        ('rf_bs','rf_px'),('rf_px','rf_md'),('rf_md','rf_ds'),
                        ('th_mp','th_bs'),('th_bs','th_px'),('th_px','th_ds')}
                    link_wT = {}
                    for nm in self._box_primitives:
                        if nm in fk_final:
                            link_wT[nm] = bT_final @ fk_final[nm].get_matrix()
                    for nm_i in self._box_primitives:
                        if nm_i not in link_wT: continue
                        for nm_j in self._box_primitives:
                            if nm_j not in link_wT or nm_i >= nm_j: continue
                            si = nm_i.replace(_prefix, '')
                            sj = nm_j.replace(_prefix, '')
                            if (si, sj) in _adj_feas or (sj, si) in _adj_feas: continue
                            fi_f = si.split('_')[0]; fj_f = sj.split('_')[0]
                            if fi_f == fj_f and fi_f != 'palm': continue
                            # Check all box pairs between these two links
                            for bi_c, bi_r, bi_h in self._box_primitives[nm_i]:
                                for bj_c, bj_r, bj_h in self._box_primitives[nm_j]:
                                    ci_w = (link_wT[nm_i] @ bi_c.unsqueeze(-1)).squeeze(-1)[:, :3]
                                    ri_w = link_wT[nm_i][:, :3, :3] @ bi_r.unsqueeze(0)
                                    cj_w = (link_wT[nm_j] @ bj_c.unsqueeze(-1)).squeeze(-1)[:, :3]
                                    rj_w = link_wT[nm_j][:, :3, :3] @ bj_r.unsqueeze(0)
                                    sd = box_box_sdf_batch(
                                        ci_w, ri_w, bi_h.unsqueeze(0).expand(B, -1),
                                        cj_w, rj_w, bj_h.unsqueeze(0).expand(B, -1))
                                    sc_worst_sdf = torch.minimum(sc_worst_sdf, sd)

                # Actuation distance
                act_dist = torch.full((B,), 999.0, device=dev)
                if n_act and ap is not None:
                    for j in range(n_act):
                        fi = self.amap_t[:, j]
                        act_dist = torch.norm(tp_final[torch.arange(B, device=dev), fi] - ap[j], dim=-1)

                # ds penetration: check BOTH back-side (strict) and pad-side (tolerant).
                # Back-side (y > -15mm): strict threshold — back of fingertip should never
                # be inside (that's finger-through-object insertion).
                # Pad-side (y < -15mm): tolerant — shallow wrapping on curved surfaces is OK,
                # but deep penetration (>5mm) means the pad is piercing the surface.
                # Compute ds collision per-link, excluding the actuation finger's ds link
                # per env (its pad is supposed to be at the trigger / inside clearance).
                ds_back_worst = torch.zeros(B, device=dev)
                ds_pad_worst = torch.zeros(B, device=dev)
                prefixes_ds = ['if', 'mf', 'rf', 'th']
                for li, (nm, _) in enumerate(self._col_data):
                    if "_ds" not in nm: continue
                    # Which finger does this ds belong to?
                    ds_fi = next((pi for pi, p in enumerate(prefixes_ds)
                                  if f"_{p}_ds" in nm), None)
                    if ds_fi is None: continue
                    # Mask: envs where this is NOT the actuation finger
                    not_act = (self.amap_t[:, 0] != ds_fi)  # [B]
                    si, ei = self._col_link_ranges[li]
                    back_mask_li = self._ds_back_mask[si:ei]
                    if back_mask_li.any():
                        ds_back_sdf = cs_final[:, si:ei][:, back_mask_li].min(-1).values
                        # Only update for support envs
                        ds_back_worst = torch.where(
                            not_act, torch.minimum(ds_back_worst, ds_back_sdf), ds_back_worst)
                    if (~back_mask_li).any():
                        ds_pad_sdf = cs_final[:, si:ei][:, ~back_mask_li].min(-1).values
                        ds_pad_worst = torch.where(
                            not_act, torch.minimum(ds_pad_worst, ds_pad_sdf), ds_pad_worst)

                # Feasibility: only candidates that passed opt_mask + quality checks.
                # NOTE: ds_pad/ds_back/max_col_viol now use OBJECT-ONLY SDF (cs_final above).
                # Support finger clearance entry is checked separately with NO margin
                # (support must never enter the actuation clearance zone).
                feasible = (opt_mask
                            & (surf_err < 0.008)  # 8mm surface
                            & (max_col_viol < 0.003)  # 3mm non-ds object collision margin
                            & (ds_back_worst > -0.003)  # 3mm back-side ds pen
                            & (ds_pad_worst > -0.005)  # 5mm pad-side ds pen
                            & (sup_clearance_worst >= 0.0)  # support MUST NOT enter clearance
                            & (sc_worst_sdf > -0.001)  # box-box SDF > -1mm
                            & (sigma_all > 0.01)  # grasp matrix full rank (necessary for FC)
                            & has_lstar)  # FC LP has a positive margin beyond numerical noise
                if n_act:
                    feasible = feasible & (act_dist < 0.010)  # 10mm actuation

                # Wrapping quality uses each candidate's own three support
                # fingertips. The previous last-group tp_fc was incorrectly
                # reused for all candidates and could include a distant palm.
                support_index = torch.tensor(
                    [[fi for fi in range(4) if fi != act] for act in range(4)], device=dev)
                support_index = support_index[self.amap_t[:, 0]]
                wrap_tips = torch.gather(all_tips_final, 1,
                                         support_index.unsqueeze(-1).expand(-1, -1, 3))
                wrap_dirs = F.normalize(wrap_tips - obj_c.unsqueeze(0).unsqueeze(0), dim=-1)
                wrap_balance = wrap_dirs.sum(dim=1).norm(dim=-1)
                wrap_quality = 1.0 - wrap_balance.clamp(max=1.5) / 1.5

                # Composite ranking: l* first (the authoritative FC metric), then σ_min
                # Grasps with a robust positive l* rank above numerical-zero LPs.
                # Infeasible-grasp rank penalizes ALL collision violations + low sigma.
                # Without sigma weight, grasps with σ=0 (no FC — useless) could outrank
                # σ>0.03 grasps if their geometric penalties were lower.
                ds_pad_viol = F.relu(-0.005 - ds_pad_worst)   # 0 if pass, positive if fail
                ds_back_viol = F.relu(-0.003 - ds_back_worst)
                sc_viol = F.relu(-0.001 - sc_worst_sdf)
                sigma_viol = F.relu(0.01 - sigma_all)  # viol if σ<0.01
                infeas_penalty = (5.0 * surf_err + 10.0 * max_col_viol
                                  + 10.0 * ds_pad_viol + 15.0 * ds_back_viol
                                  + 10.0 * sc_viol + 50.0 * sigma_viol)
                rank_score = torch.where(
                    feasible & has_lstar,
                    10.0 + final_lstars_t + 0.3 * wrap_quality,  # l*>0 grasps rank highest
                    torch.where(
                        feasible,
                        sigma_all + 0.3 * wrap_quality,  # feasible but no l*
                        # Boost sigma 10x so σ=0.05 vs σ=0 is a 0.5 difference
                        torch.tensor(-10.0, device=dev) + 10.0 * sigma_all - infeas_penalty,
                    ),
                )
                order = rank_score.argsort(descending=True)
                self._final_order = order.clone()

                n_feasible = feasible.sum().item()
                n_fc = (sigma_all > 0.01).sum().item()
                n_lstar = has_lstar.sum().item()
                n_surf_ok = (surf_err < 0.005).sum().item()
                n_col_ok = (max_col_viol < 0.002).sum().item()
                n_back_ok = (ds_back_worst > -0.003).sum().item()
                n_pad_ok = (ds_pad_worst > -0.005).sum().item()
                n_sc_ok = (sc_worst_sdf > -0.001).sum().item()
                print("\n  === Optimization Results ===")
                print(f"  {n_feasible}/{n_opt} feasible | l*>{FC_LSTAR_MIN_MARGIN:g}: {n_lstar} | σ>0.01: {n_fc} | "
                      f"surf<5mm: {n_surf_ok} | col<2mm: {n_col_ok} | "
                      f"back>-3: {n_back_ok} | pad>-5: {n_pad_ok} | sc>-1: {n_sc_ok}")
                bi = order[0].item()
                print(f"  Best: idx={bi} σ_min={sigma_all[bi]:.4f} l*={final_lstars[bi]:.4f} "
                      f"surf={surf_err[bi]*1000:.1f}mm col={max_col_viol[bi]*1000:.1f}mm "
                      f"ds_back={ds_back_worst[bi]*1000:.1f}mm ds_pad={ds_pad_worst[bi]*1000:.1f}mm "
                      f"sc_sdf={sc_worst_sdf[bi]*1000:.1f}mm")

                # Build result list
                R_all = self._rot6d_to_matrix(self.rot6d)
                feas_order = order[feasible[order]]
                infeas_order = order[~feasible[order]]
                final_order = torch.cat([feas_order, infeas_order])

                res = []
                for i in range(min(10, len(final_order))):
                    ix = final_order[i].item()
                    if not opt_mask[ix]:
                        continue
                    res.append({
                        "q_joints": q_final[ix].cpu().numpy(),
                        "base_pos": self.pos[ix].detach().cpu().numpy(),
                        "base_rot": R_all[ix].cpu().numpy(),
                        "score": float(rank_score[ix]),
                        "l_star": float(final_lstars[ix]),
                        "fc_lp_valid": bool(final_lp_valid[ix]),
                        "fc_contact_count": 3 + int(palm_used_all[ix]),
                        "l_star_acceptance_margin": FC_LSTAR_MIN_MARGIN,
                        "l_bar": float((3 + int(palm_used_all[ix])) * ns * final_lstars[ix]),
                        "feasible": bool(feasible[ix]),
                        "act_assignment": self.amap[ix].tolist(),
                        "act_dist": float(act_dist[ix]) if n_act else 0.0,
                        "surf_err": float(surf_err[ix]),
                        "min_col": float(cs_final[ix].min()),
                        "max_col_viol": float(max_col_viol[ix]),
                        "ds_back_worst": float(ds_back_worst[ix]),
                        "ds_pad_worst": float(ds_pad_worst[ix]),
                        "sigma_min": float(sigma_all[ix]),
                        "palm_gap_m": (float(palm_gap_final[ix])
                                       if palm_gap_final is not None else None),
                        "palm_used_for_fc": bool(palm_used_all[ix]),
                        "sc_min_dist": float(sc_worst_sdf[ix]),
                    })
                    # Add init metadata
                    if hasattr(self, '_init_surf_pts') and ix < len(self._init_surf_pts):
                        res[-1]["surf_pt"] = self._init_surf_pts[ix].numpy()
                        res[-1]["outward_normal"] = self._init_outward[ix].numpy()
                    res[-1]["act_finger"] = int(self.amap[ix, 0])
                    res[-1]["env_idx"] = int(ix)

            # Post-solve box-grid verification
            if self.hand_type == "leap" and res:
                from scipy.spatial.transform import Rotation as _ScipyR
                from scipy.spatial import cKDTree as _cKDTree
                import xml.etree.ElementTree as _ET_v
                _urdf_v = os.path.join(_MODELS_DIR, f"leap_{self.hand}/leap_solver.urdf")
                _tree_v = _ET_v.parse(_urdf_v)
                _vp = 0.005  # 5mm verification grid
                _verify_pts = {}
                for _le in _tree_v.getroot().findall("link"):
                    _ln = _le.get("name")
                    _lpts = []
                    for _cel in _le.findall("collision"):
                        _g = _cel.find("geometry")
                        if _g is None: continue
                        _b = _g.find("box")
                        if _b is None: continue
                        _sz = [float(x) for x in _b.get("size").split()]
                        _o = _cel.find("origin")
                        _p = np.array([float(x) for x in _o.get("xyz", "0 0 0").split()])
                        _rpy = np.array([float(x) for x in _o.get("rpy", "0 0 0").split()])
                        _R = (_ScipyR.from_euler("xyz", _rpy).as_matrix()
                              if np.any(np.abs(_rpy) > 1e-6) else np.eye(3))
                        hx, hy, hz = _sz[0]/2, _sz[1]/2, _sz[2]/2
                        gx = np.arange(-hx, hx + _vp/2, _vp)
                        gy = np.arange(-hy, hy + _vp/2, _vp)
                        gz = np.arange(-hz, hz + _vp/2, _vp)
                        grid = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'),
                                        axis=-1).reshape(-1, 3)
                        grid = ((_R @ grid.T).T + _p).astype(np.float32)
                        _lpts.append(grid)
                    if _lpts:
                        _verify_pts[_ln] = np.vstack(_lpts)

                # Add visual mesh surface samples for ds links (URDF boxes
                # stop at y=-20mm, pad tip extends to y=-49.5mm — those points
                # are critical for detecting fingertip insertion).
                for _li_v, (_lnv, _ptsv) in enumerate(self._col_data):
                    if "_ds" not in _lnv: continue
                    _si, _ei = self._col_link_ranges[_li_v]
                    # Only visual mesh samples (not the URDF box grid part).
                    # Visual samples were appended after the box grid.
                    _n_urdf_box = sum(1 for _cel in _tree_v.getroot().findall("link")
                                      if _cel.get("name") == _lnv
                                      for _cc in _cel.findall("collision")
                                      for _g in [_cc.find("geometry")] if _g is not None
                                      for _b in [_g.find("box")] if _b is not None)
                    # Take pad-side samples (y < -15mm) to extend coverage past URDF boxes
                    _pts_np = _ptsv[:, :3].cpu().numpy()
                    _back = self._ds_back_mask[_si:_ei].cpu().numpy()
                    _pad_pts = _pts_np[~_back].astype(np.float32)
                    if len(_pad_pts) > 0 and _lnv in _verify_pts:
                        _verify_pts[_lnv] = np.vstack([_verify_pts[_lnv], _pad_pts])

                _adj = {('palm','if_bs'),('palm','mf_bs'),('palm','rf_bs'),('palm','th_mp'),
                        ('if_bs','if_px'),('if_px','if_md'),('if_md','if_ds'),
                        ('mf_bs','mf_px'),('mf_px','mf_md'),('mf_md','mf_ds'),
                        ('rf_bs','rf_px'),('rf_px','rf_md'),('rf_md','rf_ds'),
                        ('th_mp','th_bs'),('th_bs','th_px'),('th_px','th_ds')}

                print("\n  === BOX-GRID VERIFICATION (5mm) ===")
                for i, r in enumerate(res):
                    q_v = torch.tensor(r["q_joints"], dtype=torch.float32, device=dev).unsqueeze(0)
                    fk_v = self.chain.forward_kinematics(q_v)
                    bT_v = np.eye(4)
                    bT_v[:3, :3] = r["base_rot"]; bT_v[:3, 3] = r["base_pos"]

                    total_v = 0; total_pen = 0; worst_sdf = 0.0; worst_link = ""
                    for nm, pts in _verify_pts.items():
                        if nm not in fk_v: continue
                        lt = fk_v[nm].get_matrix()[0].detach().cpu().numpy()
                        wT = bT_v @ lt
                        pw = (wT[:3, :3] @ pts.T).T + wT[:3, 3]
                        sv = self.sdf.query(
                            torch.tensor(pw, dtype=torch.float32, device=dev).unsqueeze(0)
                        )[0].cpu().numpy()
                        total_v += len(sv)
                        n_pen = (sv < -0.001).sum()
                        total_pen += n_pen
                        link_worst = sv.min()
                        if link_worst < worst_sdf:
                            worst_sdf = link_worst
                            worst_link = nm.split("leap_rh_")[-1] if "leap_rh_" in nm else nm

                    pct = 100 * total_pen / total_v if total_v > 0 else 0
                    r["mesh_pen_pct"] = pct
                    r["mesh_pen_worst"] = float(worst_sdf)
                    if pct > 5.0:
                        r["feasible"] = False

                    # Self-collision check (box points)
                    _lv_col = {}
                    for _cnm, _cpts in self._sc_data:  # same point set as optimizer
                        if _cnm in fk_v:
                            _cwT = bT_v @ fk_v[_cnm].get_matrix()[0].detach().cpu().numpy()
                            _cp_np = _cpts[:, :3].cpu().numpy()
                            _lv_col[_cnm] = (_cwT[:3, :3] @ _cp_np.T).T + _cwT[:3, 3]
                    worst_sc = 999.0; sc_bad = []
                    _lnames = sorted(_lv_col.keys())
                    for _ii in range(len(_lnames)):
                        if len(_lv_col[_lnames[_ii]]) < 2: continue
                        _tree_sc = _cKDTree(_lv_col[_lnames[_ii]])
                        for _jj in range(_ii + 1, len(_lnames)):
                            if len(_lv_col[_lnames[_jj]]) < 2: continue
                            _ni = _lnames[_ii].split('leap_rh_')[-1]
                            _nj = _lnames[_jj].split('leap_rh_')[-1]
                            if (_ni, _nj) in _adj or (_nj, _ni) in _adj: continue
                            # Skip same-finger pairs (non-adjacent but still same finger)
                            _fi = _ni.split('_')[0] if '_' in _ni else _ni
                            _fj = _nj.split('_')[0] if '_' in _nj else _nj
                            if _fi == _fj and _fi != 'palm': continue
                            _dd, _ = _tree_sc.query(_lv_col[_lnames[_jj]], k=1)
                            _md = _dd.min()
                            if _md < worst_sc: worst_sc = _md
                            if _md < 0.003: sc_bad.append(f"{_ni}-{_nj}")
                    r["sc_worst"] = float(worst_sc)
                    if worst_sc < 0.0005:
                        r["feasible"] = False

                    f_tag = "FEAS" if r["feasible"] else "FAIL"
                    sc_str = f"sc={worst_sc*1000:.1f}mm" if worst_sc < 999 else ""
                    sc_bad_str = f" [{','.join(sc_bad[:3])}]" if sc_bad else ""
                    print(f"    G{i} [{f_tag}] σ={r['sigma_min']:.4f} l*={r['l_star']:.4f} "
                          f"surf={r['surf_err']*1000:.1f}mm pen={pct:.1f}%@{worst_link} "
                          f"{sc_str}{sc_bad_str}")

            if save_path is not None:
                import torch as _torch
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                _torch.save(res, save_path)
                print(f"  Results saved to {save_path}")
                # Save per-stage metrics log for diagnostic analysis
                if hasattr(self, '_metrics_log'):
                    metrics_path = save_path.replace('.pt', '_metrics.pt')
                    _torch.save(self._metrics_log, metrics_path)
                    print(f"  Per-stage metrics saved to {metrics_path}")
            _tock("stage_feas_eval", "_t_feas")
            _tock("total", "_t0")
            if "stage_init" in self._stage_times:
                t_init = self._stage_times.get("stage_init", 0)
                t_act = self._stage_times.get("stage_act_ik_and_slide", 0)
                t_filt = self._stage_times.get("stage_filt_supIK", 0)
                t_sup_loop = self._stage_times.get("stage_supIK_loop", 0)
                t_filt_only = max(0, t_filt - t_sup_loop)
                t_main = self._stage_times.get("stage_main_opt", 0)
                t_feas = self._stage_times.get("stage_feas_eval", 0)
                t_total = self._stage_times.get("total", 0)
                t_palm = t_init - t_act
                t_other = t_total - t_init - t_filt - t_main - t_feas
                print(f"  [TIMING] palm_init={t_palm:.1f}s  act_ik={t_act:.1f}s  "
                      f"filt={t_filt_only:.1f}s  supIK_400={t_sup_loop:.1f}s  "
                      f"main_opt={t_main:.1f}s  feas={t_feas:.1f}s  "
                      f"other={t_other:.1f}s  total={t_total:.1f}s")
            return res

        # No candidates passed entry criteria — return empty
        print("  WARNING: No candidates passed entry criteria for optimization")
        if save_path is not None:
            import torch as _torch
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            _torch.save([], save_path)
        return []
