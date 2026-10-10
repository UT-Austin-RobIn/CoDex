"""Table scene with a separate finite floor in every GPU sub-scene.

Adapts ManiSkill 3.0.1 TableSceneBuilder.build (Apache-2.0); CoDex changes
replace its shared collision plane with per-layer floor slabs. See
layered_table_scene.LICENSE for the upstream terms.
"""

import inspect
from pathlib import Path

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder.table import TableSceneBuilder


class LayeredTableSceneBuilder(TableSceneBuilder):
    """Keep ManiSkill's table and ground visual, with one finite floor per z layer.

    ManiSkill 3.0.1 builds only one plane collision for all parallel sub-scenes.
    That plane is below only the lowest layer when sub-scenes are spread over z.
    A wide, thin box at each layer catches objects ejected beyond their table.
    """

    FLOOR_HALF_WIDTH = 250.0
    FLOOR_HALF_THICKNESS = 0.05

    def build(self):
        # This mirrors ManiSkill 3.0.1's TableSceneBuilder.build. Its ground
        # collision is replaced while the table geometry and visual remain.
        builder = self.scene.create_actor_builder()
        table_model_file = str(Path(inspect.getfile(TableSceneBuilder)).parent / "assets" / "table.glb")
        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
            half_size=(2.418 / 2, 1.209 / 2, 0.9196429 / 2),
        )
        builder.add_visual_from_file(filename=table_model_file, scale=[1.75] * 3, pose=table_pose)
        builder.initial_pose = sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        self.table = builder.build_kinematic(name="table-workspace")

        aabb = np.array([[-0.7402168, -1.2148621, -0.91964257],
                         [0.4688596, 1.2030163, 3.5762787e-07]])
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]

        self.ground = build_ground(self.scene, floor_width=500, altitude=-self.table_height,
                                   add_collision=False)
        self.floors = []
        offsets = self.scene.scene_offsets_np
        for layer_number, z in enumerate(np.unique(offsets[:, 2])):
            scene_idx = int(np.flatnonzero(offsets[:, 2] == z)[0])
            offset = offsets[scene_idx]
            floor_builder = self.scene.create_actor_builder()
            floor_builder.set_scene_idxs([scene_idx])
            floor_builder.add_box_collision(
                half_size=(self.FLOOR_HALF_WIDTH, self.FLOOR_HALF_WIDTH, self.FLOOR_HALF_THICKNESS),
            )
            # ActorBuilder adds the chosen sub-scene offset to initial_pose.
            # Cancel x/y so each slab covers every cell in its z layer.
            floor_builder.initial_pose = sapien.Pose(p=[
                -float(offset[0]), -float(offset[1]), -self.table_height - self.FLOOR_HALF_THICKNESS,
            ])
            self.floors.append(floor_builder.build_static(name=f"ground-collision-layer-{layer_number}"))
        self.scene_objects = [self.table, self.ground, *self.floors]
