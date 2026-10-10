# Object meshes

Every scene needs a mesh of its functional object. Stage 0 builds it in three steps:

1. **Mesh.** Tripo (`TRIPO_API_KEY`) generates it from the crop of the functional object
   (`obj_0.png`), or you supply your own with `--mesh`.
2. **Registration.** FoundationPose registers the mesh against the RGB-D capture. The object height
   measured in the depth fixes the metric scale, and the registered pose fixes which way is up. The
   result is baked into `mesh/mesh.glb` (metres, +Z up, bottom at its lowest point) and the pose in
   the scene into `pose.txt`.
3. **Collision parts.** CoACD splits the mesh into the convex parts used for simulation
   (`mesh_coacd/`).

## Your own mesh

```bash
# a new scene from a capture, with your mesh instead of Tripo's
python scripts/0_create_scene.py --scene-name my_scene --image rgb.png --points pts.npy \
    --intrinsics K.npy --task "spray the plant" --mesh object.obj
# replace the mesh of an existing scene
python scripts/0_create_scene.py --output-dir out/scenes/my_scene --remesh --mesh object.glb
```

Any of `.glb`, `.gltf`, `.obj`, `.ply`, `.stl` or `.off` works, at any scale. One of the mesh's
x, y or z axes must point along the object's up direction (as with most CAD exports and scans,
Y-up or Z-up). Registration warns when the object is tilted away from every axis. The mesh must
match the object in the capture, since registration fits it to the observed points.

`--remesh` without `--mesh` generates a new mesh with Tripo. Keypoints, goal pose and grasps
belong to a mesh. Rebuilds finish registration and CoACD before replacing the published mesh;
old mesh-dependent outputs are archived under `<scene>/.history/`. Failed rebuilds leave them
intact. Training runs are kept, but evaluating an old policy against a remeshed scene changes
its inputs; use the archived original assets when comparing the original task.

In the app, *New scene → From your RGB-D capture* has the same mesh choice, and *Rebuild the mesh*
on the Scene tab replaces the mesh of the current scene.

## Outputs

```
out/scenes/<scene_name>/
    scene.pkl, scene_img.png        capture and segmentation
    obj_0.png                       functional object crop (Tripo's input)
    mesh/mesh.glb                   canonical mesh
    mesh/metadata.json              bounding box
    mesh_coacd/                     convex decomposition for simulation
    pose.txt                        canonical mesh in the camera frame
```

To recompute only the bounding box and the CoACD parts of the current mesh:
`python codex/scene/post_mesh_metadata.py --mesh-path out/scenes/<scene_name>/mesh/mesh.glb`.
