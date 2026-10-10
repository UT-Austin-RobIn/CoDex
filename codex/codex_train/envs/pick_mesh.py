from typing import Any, Dict, List, Union, Literal
import os
import json
import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as R
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig, SceneConfig
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from codex.codex_train.configs.primitive import DOMAIN_RANDOMIZATION
from codex.codex_train.envs.custom_mesh_builder import get_custom_mesh_builder
from codex.codex_train.envs.layered_table_scene import LayeredTableSceneBuilder
from codex.codex_train.agents.load_grasp_qpos import (
    load_grasp_config, transform_points, reorder_finger_config, mat_to_pos_euler, compute_hand_pose)
from codex.codex_train.agents.leap_rh import FloatingLeapHandRight  # noqa: F401  (registers the agent)

# Import the mesh builder from our new file.


CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# The functional object's own friction. PhysX averages the coefficients of two
# touching surfaces, so with this fixed the hand's and the table's materials
# set the hand/object and table/object coefficients independently.
OBJECT_MATERIAL_FRICTION = 0.3


def draw(spec, n):
    """n values from a (low, high) range, or n copies of a single number."""
    if np.ndim(spec) == 0:
        return np.full(n, float(spec))
    low, high = spec
    return np.random.uniform(float(low), float(high), size=n)


def pair_material(coefficient):
    """Material that makes `coefficient` the friction of its contacts with the object."""
    mu = 2.0 * float(coefficient) - OBJECT_MATERIAL_FRICTION
    if mu < 0:
        raise ValueError(f"Friction {coefficient} is below the lowest representable "
                         f"{OBJECT_MATERIAL_FRICTION / 2} (OBJECT_MATERIAL_FRICTION / 2)")
    return sapien.physx.PhysxMaterial(static_friction=mu, dynamic_friction=mu, restitution=0.0)


# Radius (m) of the torsional friction patch at the fingertips: about a
# fingertip contact (it was 0.1 m; success changes by at most 3 points).
FINGERTIP_LINKS = ("leap_rh_if_ds", "leap_rh_mf_ds", "leap_rh_rf_ds", "leap_rh_th_ds")
FINGERTIP_PATCH_RADIUS = 0.01


def set_material(components, material, patch_radius=None):
    for component in components:
        for shape in getattr(component, "get_collision_shapes", lambda: [])():
            shape.set_physical_material(material)
            if patch_radius is not None:
                shape.set_patch_radius(patch_radius)
                shape.set_min_patch_radius(patch_radius)


def load_sim_grasp_config(grasp_path: str, offset: np.ndarray):
    """Load a grasp with the single actuation finger required by the simulator."""
    grasp_name = os.path.basename(grasp_path).removesuffix(".txt").removesuffix("_optimized")
    metadata_path = os.path.join(os.path.dirname(grasp_path), f"{grasp_name}_metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Simulation grasp {grasp_path} requires companion actuation metadata "
            f"{metadata_path} with actuation_finger_indices=[0..3]."
        )
    try:
        joint_config, hand_pose, act_idx = load_grasp_config(grasp_path, offset=offset)
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Invalid actuation metadata companion {metadata_path}: expected a JSON object "
            f"with actuation_finger_indices=[0..3] ({exc})."
        ) from exc
    if (not isinstance(act_idx, list) or len(act_idx) != 1
            or type(act_idx[0]) is not int or not 0 <= act_idx[0] < 4):
        raise ValueError(
            f"Invalid actuation metadata companion {metadata_path}: "
            "actuation_finger_indices must be a one-element list of integer finger indices 0..3."
        )
    return joint_config, hand_pose, act_idx


def sub_scene_offsets(num_envs, spacing):
    """Offsets of the GPU sub-scenes on a cube grid, `spacing` apart on every axis.

    ManiSkill lays the sub-scenes out on a square grid at z = 0, so all of them
    share one z range. With a few thousand environments PhysX 5.3.1's GPU
    broadphase then keeps stale shape pairs alive, and they pile up episode
    after episode: each step gets slower and GPU memory grows (fixed upstream
    in PhysX 5.6, which SAPIEN 3.0.3 does not ship). Spreading the sub-scenes
    over all three axes avoids it."""
    n = int(np.ceil(num_envs ** (1 / 3) - 1e-9))
    return [[(i % n - n // 2) * spacing,
             ((i // n) % n - n // 2) * spacing,
             (i // (n * n) - n // 2) * spacing] for i in range(num_envs)]


@register_env("PickMesh-v1")
class PickMeshEnv(BaseEnv):
    """Scene of one functional object (a scene's CoACD mesh) on a table with a
    floating LEAP hand.

    One functional object per parallel environment, standing upright at the
    origin (meshes are stored canonical, z-up). Each environment draws its own
    hand/object and table/object friction and the object's mass and size
    multipliers (`hand_friction`, `table_friction`, `mass_scale`,
    `size_scale`: a (low, high) range or a fixed value; defaults in
    configs/primitive.py DOMAIN_RANDOMIZATION). The
    grasp (`grasp_dir`/`grasp_configs`) and actuation contact
    (`actuation_dir`/fn_key_pts.json) are loaded once and shared by all
    environments. Subclasses implement the grasp-move-actuate primitive.
    """
    SUPPORTED_ROBOTS = ["floating_leap_hand_right"]
    agent: Any

    @property
    def _default_sim_config(self):
        return SimConfig(
            # solver_iterations=20,
            scene_config=SceneConfig(
                solver_position_iterations=20,
                solver_velocity_iterations=1,
            ),
            gpu_memory_config=GPUMemoryConfig(
                # Keep the original per-environment allowance used up to
                # 1024 lanes; independent scenes need linear reservations.
                max_rigid_contact_count=self.num_envs * 1024 * 20,
                max_rigid_patch_count=self.num_envs * 1024 * 10,
                found_lost_pairs_capacity=2**26,
            )
        )

    def _setup_scene(self):
        super()._setup_scene()
        # Must run before anything is added: actors take their sub-scene's offset when built.
        if self._sim_device.is_cuda() and self.num_envs > 1:
            offsets = sub_scene_offsets(self.num_envs, self.sim_config.spacing)
            for sub_scene, offset in zip(self.scene.sub_scenes, offsets):
                self.scene.px.set_scene_offset(sub_scene, offset)

    def __init__(
        self,
        *args,
        grasp_configs=None,
        robot_uids="floating_leap_hand_right",
        robot_init_qpos_noise=0.1,
        init_at_loaded_grasp=False,
        init_at_standing_pose=False,
        init_at_fixed_rot=False,
        reconfiguration_freq=None,
        mesh_dir: str = None,
        grasp_dir: str = None,
        actuation_dir: str = None,
        hand_friction=DOMAIN_RANDOMIZATION["hand_friction"],
        table_friction=DOMAIN_RANDOMIZATION["table_friction"],
        mass_scale=DOMAIN_RANDOMIZATION["mass_scale"],
        size_scale=DOMAIN_RANDOMIZATION["size_scale"],
        **kwargs,
    ):
        self.randomization = dict(hand_friction=hand_friction, table_friction=table_friction,
                                  mass_scale=mass_scale, size_scale=size_scale)
        if mesh_dir is None:
            raise ValueError("mesh_dir must be specified (e.g. /path/to/meshes_coacd)")
        if grasp_dir is None:
            raise ValueError("grasp_dir must be specified (e.g. /path/to/grasps)")
        if actuation_dir is None:
            raise ValueError("actuation_dir must be specified (e.g. /path/to/actuation_3d)")
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.mesh_dir = mesh_dir
        self.mesh_path = os.path.join(mesh_dir, "mesh.glb")
        self.grasp_dir = grasp_dir
        self.actuation_dir = actuation_dir
        self.init_at_loaded_grasp = init_at_loaded_grasp
        self.init_at_standing_pose = init_at_standing_pose
        self.init_at_fixed_rot = init_at_fixed_rot

        mesh_meta_path = os.path.join(self.mesh_dir, "metadata.json")
        with open(mesh_meta_path, "r") as f:
            self.mesh_info = json.load(f)

        # Grasp files store the hand pose in the canonical mesh frame shifted so
        # the mesh bottom sits at z=0. Scene meshes are origin-centered, so the
        # bottom shift equals half the bbox height. (Asserted against the actual
        # collision mesh bounds in _load_collision_meshes.)
        mesh_height = self.mesh_info["bbox"][2]
        mesh_offset = np.array([0.0, 0.0, mesh_height]) / 2.0
        if grasp_configs is None:
            raise ValueError("grasp_configs must be provided when using scene-level mesh loading.")

        self.loaded_grasps = []
        for grasp_config_name in grasp_configs:
            grasp_path = os.path.join(self.grasp_dir, f"{grasp_config_name}.txt")
            if not os.path.exists(grasp_path):
                raise FileNotFoundError(f"Grasp file {grasp_path} does not exist.")
            joint_config, hand_rel_pose_mat, act_idx = load_sim_grasp_config(grasp_path, offset=mesh_offset)

            joint_config = torch.tensor(joint_config)
            hand_rel_pose_mat = torch.tensor(hand_rel_pose_mat)
            act_idx = torch.tensor(act_idx)
            self.loaded_grasps.append((joint_config, hand_rel_pose_mat, act_idx))

        # Load the actuation contact from fn_key_pts.json in the actuation directory.
        keypoints_path = os.path.join(self.actuation_dir, "fn_key_pts.json")
        with open(keypoints_path, "r") as f:
            kp_data = json.load(f)
        act_contact = kp_data["act_contact"]
        actuation_contacts = np.array([[np.array(act_contact["pos"]), np.array(act_contact["dir"])]])
        actuation_contacts = torch.tensor(actuation_contacts).float()
        self.mesh_info["actuation_contacts"] = actuation_contacts

        # if reconfiguration_freq is None:
        #     reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi/2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # Position the floating hand's root at an offset.
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        builder_type = (LayeredTableSceneBuilder
                        if self._sim_device.is_cuda() and self.num_envs > 1 else TableSceneBuilder)
        self.table_scene = builder_type(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        b = self.num_envs
        # Physical parameters of each environment (see the class docstring).
        self.physics = {k: draw(v, b) for k, v in self.randomization.items()}
        for i in range(b):
            set_material(self.table_scene.table._objs[i].components, pair_material(self.physics["table_friction"][i]))
            material = pair_material(self.physics["hand_friction"][i])
            for link in self.agent.robot.links:
                tip = link.name in FINGERTIP_LINKS
                set_material([link._objs[i]], material, FINGERTIP_PATCH_RADIUS if tip else None)
        builders = []
        for i in range(b):
            builder = get_custom_mesh_builder(
                scene=self.scene,
                mesh_path=self.mesh_path,
                add_collision=True,
                add_visual=True,
                friction=OBJECT_MATERIAL_FRICTION,
                density=1000.0 * self.physics["mass_scale"][i],
                scale=self.physics["size_scale"][i],
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            builders.append(builder)
        self._size_scales = torch.as_tensor(self.physics["size_scale"], dtype=torch.float32)

        self._objs: List[Actor] = []
        for i in range(b):
            builder = builders[i]
            # Build the object and add it to the list.
            _obj = builder.build(name=f"mesh-{i}")
            # Optional center-of-mass override from metadata ("com", canonical
            # frame). Uniform density over-weights thin hollow appendages
            # (e.g. the air blower's nozzle); the override restores a
            # physically plausible mass center without touching the geometry.
            if self.mesh_info.get("com") is not None:
                com = np.asarray(self.mesh_info["com"], dtype=np.float32) * self.physics["size_scale"][i]
                for body in _obj._bodies:
                    body.cmass_local_pose = sapien.Pose(p=com)
            self._objs.append(_obj)
        self.obj_free = Actor.merge(self._objs, name="mesh_object",)
        # self.obj = Actor.merge(self._objs, name="mesh_object")
        
        self.rotation_locked = False
        if self.init_at_fixed_rot:
            self.rotation_locked = True
            self._objs_locked: List[Actor] = []
            for i in range(b):
                builder = get_custom_mesh_builder(
                    scene=self.scene,
                    mesh_path=self.mesh_path,
                    add_collision=True,
                    add_visual=True,
                    friction=OBJECT_MATERIAL_FRICTION,
                    density=1000.0 * self.physics["mass_scale"][i],
                    scale=self.physics["size_scale"][i],
                )
                builder.initial_pose = sapien.Pose(p=[0.4, 0.4, 0.3])
                builder.set_scene_idxs([i])
                # Build the object and add it to the list.
                _obj_free = builder.build(name=f"mesh-{i}-free")
                self._objs_locked.append(_obj_free)
            self.obj_locked = Actor.merge(self._objs_locked, name="mesh_object_locked")
            self.obj_locked.set_locked_motion_axes([False, False, False, True, True, False])

        self.obj = self.obj_free
        self.add_to_state_dict_registry(self.obj_free)
            
        # Build the goal marker.
        self.actuaction_site = actors.build_sphere(
            self.scene,
            radius=0.01,
            color=[0, 1, 0, 1],
            name="actuation_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.actuaction_site)

    def _after_reconfigure(self, options: dict):
        # Precompute each object's height from its collision bounding box.
        self.object_zs = []
        self.object_heights = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            # merged_obj = Actor.merge(obj, name="temp")
            bbox = collision_mesh.bounding_box.bounds
            self.object_zs.append(-bbox[0, 2])
            self.object_heights.append(bbox[1, 2] - bbox[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)
        self.object_heights = common.to_tensor(self.object_heights, device=self.device)
        # Grasp poses were loaded assuming an origin-centered mesh (bottom shift
        # == bbox_height/2). Warn if the actual geometry disagrees, which would
        # misplace the hand by the difference.
        expected_z = self.mesh_info["bbox"][2] / 2.0 * float(self.physics["size_scale"][0])
        actual_z = float(self.object_zs[0])
        if abs(actual_z - expected_z) > 0.005:
            print(f"WARNING: mesh bottom offset {actual_z:.4f}m differs from the "
                  f"centered-mesh assumption {expected_z:.4f}m; loaded grasp poses "
                  f"may be vertically offset by {abs(actual_z - expected_z)*1000:.1f}mm.")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Load the actuation contacts
            actuation_contacts = self.mesh_info["actuation_contacts"]
            self.act_pos_local, self.act_dir_local = actuation_contacts[:, 0, :], actuation_contacts[:, 1, :]
            self.update_sites()

            # Set object pose: upright (meshes are stored canonical, z-up),
            # optionally turned by a random rotation about Z.
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = self.object_zs[env_idx]
            R = torch.eye(3, device=self.device).expand(b, 3, 3)
            if self.init_at_standing_pose:
                z_euler = torch.zeros(b, 3, device=self.device)
                z_euler[:, 2] = torch.rand(b, device=self.device) * 2 * np.pi
                R = euler_angles_to_matrix(z_euler, convention="ZYX") @ R
            qs = matrix_to_quaternion(R.contiguous())

            if self.init_at_fixed_rot:
                free_xyz = torch.ones((b, 3)) * 0.5
                obj_free_pose = Pose.create_from_pq(free_xyz, qs)
                self.obj_free.set_pose(obj_free_pose)
                self.obj = self.obj_locked
                self.rotation_locked = True

            obj_pose = Pose.create_from_pq(xyz, qs)
            self.obj.set_pose(obj_pose)

            # Select and save the grasp configuration.
            finger_joints, hand_rel_pose_mat, actuation_indices = self.loaded_grasps[0]
            finger_joints = torch.tile(finger_joints, (b, 1))
            hand_rel_pose_mat = torch.tile(hand_rel_pose_mat, (b, 1, 1))
            actuation_indices = torch.tile(actuation_indices, (b, 1))
            self.loaded_grasp = (finger_joints, hand_rel_pose_mat, actuation_indices)

            if self.init_at_loaded_grasp:
                hand_pose = compute_hand_pose(hand_rel_pose_mat, obj_pose)
                self.set_hand_qpos(
                    pose=hand_pose,
                    finger_joints=finger_joints,
                    finger_joint_order="names",
                )
            
            # Otherwise start the hand 1 m above the object, 10 cm to the side.
            else:
                hand_pos = torch.zeros((b, 3))
                hand_pos[:, :2] = 0.1
                hand_pos[:, 2] = 1.0 + xyz[:, 2]
                keyframe = self.agent.keyframes["palm_down"]
                default_qpos = keyframe.qpos.copy()  # shape (22,)
                qpos_tensor = torch.tensor(default_qpos, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(b, 1)
                if self._enhanced_determinism:
                    finger_noise = self._batched_episode_rng[env_idx].normal(0, self.robot_init_qpos_noise, (b, 16))
                else:
                    finger_noise = self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, 16))
                qpos_tensor[:, :3] = hand_pos
                qpos_tensor[:, 6:] += torch.tensor(finger_noise, device=self.device, dtype=qpos_tensor.dtype)
                self.agent.reset(qpos_tensor)
            
            # This is the origin of the floating gripper (root link)
            self.agent.robot.set_pose(Pose.create_from_pq())

    def _get_obs_extra(self, info: Dict):
        proprio_obs = self.agent.get_proprioception()
        obs = dict(
            tip_poses=proprio_obs["tip_poses"],
            qpos=self.agent.robot.get_qpos(),
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - proprio_obs["palm_pose"][..., :3],
                tip_to_obj_pos=self.obj.pose.p - proprio_obs["tip_poses"][..., :3],
            )
        return obs
    
    def render(self):
        self.update_sites()
        return super().render()
    
    def update_sites(self):
        actuation_pos = self.get_actuation_pos()
        self.actuaction_site.set_pose(Pose.create_from_pq(actuation_pos))

    def evaluate(self):
        # Success is judged by the primitive's own step (compute_reward).
        return {}

    def get_object_pose(self):
        return self.obj.pose
    
    def get_hand_pose(self):
        if type(self.agent) is FloatingLeapHandRight:
            return Pose.create(self.agent.palm_pose.raw_pose.to(self.agent.device))
        proprio_obs = self.agent.get_proprioception()
        palm_pose_vec = proprio_obs["palm_pose"]
        palm_pose = Pose.create(palm_pose_vec)
        return palm_pose
    
    def get_tip_poses(self):
        return self.agent.tip_poses
    
    def get_finger_link_poses(self):
        return self.agent.finger_link_poses
    
    def get_actuation_pos(self):
        """The actuation point in the world; it scales with each environment's object."""
        T = self.get_object_pose().to_transformation_matrix()
        local = self.act_pos_local[0].to(T.device) * self._size_scales.to(T.device)[:, None]
        return (T[:, :3, :3] @ local[..., None])[..., 0] + T[:, :3, 3]
    
    def get_actuation_dir(self):
        obj_pose = self.get_object_pose()
        obj_pose_centerd = Pose.create_from_pq(p=None, q=obj_pose.q)
        act_dir = transform_points(self.act_dir_local, obj_pose_centerd)[0] # We take the first one, -> shape [b, 3]
        act_dir = act_dir / torch.linalg.norm(act_dir, dim=-1, keepdim=True)
        return act_dir

    def compute_grasp_pose(self, hand_rel_pose):
        object_pose = self.get_object_pose()
        hand_pose_mat = compute_hand_pose(hand_rel_pose, object_pose)
        return hand_pose_mat
    
    def get_finger_joints(self, order: Literal["names", "tree"] = "tree"):
        finger_joints = self.agent.robot.get_qpos()[:, 6:].clone()
        if order == "names":
            finger_joints = reorder_finger_config(
                finger_joints, mode="tree2names", is_leap=self.is_leap)
        return finger_joints
    
    def get_hand_qpos(self):
        """
        Get the hand qpos (joint positions).
        """
        hand_qpos = self.agent.robot.get_qpos()
        return hand_qpos
    
    @property
    def is_leap(self):
        return "leap" in self.agent.__class__.__name__.lower()
    
    def set_hand_qpos(self,
                      qpos: torch.Tensor = None, 
                      pose: Union[Pose, torch.Tensor] = None,
                      finger_joints: torch.Tensor = None,
                      finger_joint_order: Literal["names", "tree"] = "names"):
        """
        Set the hand pose and finger joints.

        Finger joints are in "name" order by default.
        """
        if qpos is not None:
            self.agent.robot.set_qpos(qpos)
            return
        
        else:
            hand_qpos = self.agent.robot.get_qpos()

            if pose is not None:
                if isinstance(pose, Pose):
                    pos = pose.p
                    quat = pose.q
                    euler = R.from_quat(quat[:, [1,2,3,0]].cpu().numpy()).as_euler('XYZ', degrees=False)
                    euler = torch.tensor(euler, device=self.device)
                elif isinstance(pose, torch.Tensor):
                    pos_euler = mat_to_pos_euler(pose)
                    pos, euler = pos_euler[:, :3], pos_euler[:, 3:6]

                hand_qpos[:, :3] = pos
                hand_qpos[:, 3:6] = euler
            
            if finger_joints is not None:
                finger_joints = finger_joints.clone()
                if finger_joint_order == "names":
                    finger_joints = reorder_finger_config(
                        finger_joints, mode="names2tree", is_leap=self.is_leap)
                hand_qpos[:, 6:] = finger_joints

            self.agent.robot.set_qpos(hand_qpos)
