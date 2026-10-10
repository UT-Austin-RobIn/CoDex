import sapien.physx

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder


def get_custom_mesh_builder(
    scene: ManiSkillScene,
    mesh_path: str,
    density: float = 1000,
    add_collision: bool = True,
    add_visual: bool = True,
    friction: float = 0.7,
    scale: float = 1.0,
) -> ActorBuilder:
    """Build an ActorBuilder for a canonical scene mesh (stored metric, z-up).

    friction: the object's own material friction (contact coefficients are
    the average with the other surface; see pick_mesh). scale: size multiplier
    about the mesh origin (its centre).
    """
    builder = scene.create_actor_builder()

    if add_collision:
        material = sapien.physx.PhysxMaterial(
            static_friction=friction, dynamic_friction=friction, restitution=0.0)
        builder.add_multiple_convex_collisions_from_file(
            filename=mesh_path, scale=[scale] * 3, material=material, density=density)

    if add_visual:
        builder.add_visual_from_file(filename=str(mesh_path), scale=[scale] * 3)

    return builder
