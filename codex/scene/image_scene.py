import os
from typing import Union, Optional, List, Tuple
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from codex.utils.utils_3d import transform_points
from codex.utils.utils_img import show_anns
from codex.utils.utils_path import PROJECT_ROOT_DIR
import argparse
from codex.scene.estimation.estimate_lang_seg import predict_mask, LangSAM

# Default directory for saving/loading scenes.
_DEFAULT_SAVE_DIR = os.path.join(PROJECT_ROOT_DIR, "out", "scenes")

class ImageScene:
    def __init__(
        self,
        image: Image.Image,
        points: Union[np.ndarray, str, None] = None,
        cam_intrinsics: Optional[np.ndarray] = None,
        cam_extrinsics: Optional[np.ndarray] = None
    ) -> None:
        """Initialize an ImageScene.
        
        Args:
            image (PIL.Image.Image): The full scene image.
            points (Union[np.ndarray, str, None], optional): Optional depth image or points. 
                Pixel-aligned shape (H, W, 3), camera coordinates in metres,
                or path to a npz containing those points. Defaults to None.
            cam_intrinsics (Optional[np.ndarray], optional): Camera intrinsics matrix. Defaults to None.
            cam_extrinsics (Optional[np.ndarray], optional): Camera extrinsics matrix. Defaults to None.
        """
        self.image: Image.Image = image
        if isinstance(points, str):
            self.load_points(points)
        else:
            self.points: np.ndarray = points               # Optional depth
        self.objects: list[ImageSceneObject] = []  # List of ImageSceneObject instances
        
        self.cam_intrinsics = cam_intrinsics
        self.cam_extrinsics = cam_extrinsics

    def load_points(self, points_path: str) -> None:
        """Load 3D points from a file.
        
        Args:
            points_path (str): Path to the file containing 3D points.
        """
        # Load the 3D points from the file.
        points = np.load(points_path)['points']
        print(f"Loaded 3D points from {points_path}")
        self.points = points

    def add_object(self, mask: np.ndarray, margin: float = 0.1, name: Optional[str] = None) -> "ImageSceneObject":
        """Adds an object to the scene given its binary mask.
        
        Args:
            mask (np.ndarray): Binary mask (with the same dimensions as the scene image).
            margin (float, optional): Fraction of the scene dimensions to use as margin. Defaults to 0.1.
            name (Optional[str], optional): Optional name for the object. Defaults to None.
        
        Returns:
            ImageSceneObject: The created scene object.
        """
        obj = ImageSceneObject(self, mask, margin, name)
        self.objects.append(obj)
        return obj

    def isolate_objects(self, objs: Union["ImageSceneObject", List["ImageSceneObject"], int, List[int]], bg_color: Tuple[int, int, int] = (255, 255, 255), inverse: bool = False) -> Image.Image:
        """Returns the full scene image with only the selected object's region(s) visible,
        and the rest filled with a background color.
        
        Args:
            objs (Union[ImageSceneObject, List[ImageSceneObject], int, List[int]]): Object(s) to isolate.
                If int is provided, the object at that index will be used.
            bg_color (Tuple[int, int, int], optional): Background color in RGB. Defaults to (255, 255, 255).
            inverse (bool, optional): Whether to invert the mask (hide the objects instead). Defaults to False.
        
        Returns:
            Image.Image: The resulting image.
        """
        # Allow a single object to be passed.
        if not isinstance(objs, list):
            objs = [objs]

        # Replace object indices with the actual objects.
        for i, obj in enumerate(objs):
            if isinstance(obj, int):
                objs[i] = self.objects[obj] 

        # Convert the scene image (a PIL Image) to a NumPy array for masking.
        image_np = np.array(self.image)
        combined_mask = np.zeros(image_np.shape[:2], dtype=bool)
        for obj in objs:
            combined_mask |= (obj.mask > 0)

        bg = np.full_like(image_np, bg_color)
        # If the scene image is colored, expand the mask along the channel dimension.
        if image_np.ndim == 3 and image_np.shape[2] > 1:
            combined_mask = combined_mask[..., None]
        if inverse:
            combined_mask = ~combined_mask
        result_np = np.where(combined_mask, image_np, bg)
        return Image.fromarray(result_np)
    
    def get_pcd(
        self,
        return_o3d: bool = True,
        exclude_objs: Optional[List[Union[int, "ImageSceneObject"]]] = None,
        transform: bool = True,
        reverse_selection: bool = False,
    ) -> Union[o3d.geometry.PointCloud, Tuple[np.ndarray, np.ndarray]]:
        """Get the point cloud representation of the scene.
        
        Args:
            return_o3d (bool, optional): If True, returns an Open3D PointCloud object. 
                If False, returns a tuple of points and colors arrays. Defaults to True.
            exclude_objs (List[Union[int, ImageSceneObject]], optional): List of objects to exclude 
                from the point cloud. Defaults to None.
            transform (bool, optional): Whether to apply camera extrinsics. Defaults to True.
            reverse_selection (bool, optional): If True, keeps ONLY the excluded objects. 
                Defaults to False.
                
        Returns:
            Union[o3d.geometry.PointCloud, Tuple[np.ndarray, np.ndarray]]: 
                The point cloud, or tuple of (points, colors).
        """
        assert self.points is not None, "No 3D points available for the scene."
        exclude_objs = exclude_objs or []
        # Assume self.points is of shape (H, W, 3); if not, adjust accordingly.
        assert self.points.ndim == 3
        H, W, _ = self.points.shape
        points_flat = self.points.reshape(-1, 3)
        colors_np = np.array(self.image) / 255.0
        colors_flat = colors_np.reshape(-1, 3)
        
        # Remove the nan points
        valid = ~np.isnan(points_flat).any(axis=-1)

        # Build a flat mask of everything to exclude
        exclude_obj_mask = np.zeros((H * W,), dtype=bool)
        for obj in exclude_objs:
            # allow either an index or an ImageSceneObject
            if isinstance(obj, ImageSceneObject):
                mask_flat = obj.mask.flatten()
            else:
                mask_flat = self.objects[obj].mask.flatten()
            exclude_obj_mask |= (mask_flat > 0)

        # if user wants *only* the excluded objects but didn't supply any, error out
        if reverse_selection and not exclude_objs:
            raise ValueError("reverse_selection=True requires a non‑empty exclude_objs list")

        # apply exclusion or reverse selection
        if reverse_selection:
            # keep only the excluded points
            valid &= exclude_obj_mask
        else:
            # drop those points under excluded objects
            valid &= ~exclude_obj_mask

        points_flat = points_flat[valid]
        colors_flat = colors_flat[valid]

        # Transform the points
        if transform:
            assert self.cam_extrinsics is not None, "No camera extrinsics available for the scene."
            points_flat = transform_points(points_flat, self.cam_extrinsics)

        if return_o3d:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_flat)
            pcd.colors = o3d.utility.Vector3dVector(colors_flat)
            return pcd
        else:
            return points_flat, colors_flat

    def visualize(self, show_masks: bool = False) -> None:
        """Visualizes the full scene image. Optionally overlays object masks.
        
        Args:
            show_masks (bool, optional): Whether to display the masks of the objects 
                in the scene. Defaults to False.
        """
        plt.imshow(np.array(self.image))
        if show_masks and self.objects:
            masks = [obj.mask for obj in self.objects]
            show_anns(masks)
        plt.axis("off")
        plt.show()

    def visualize_3d(self) -> None:
        """Visualizes the 3D point cloud of the full scene.
        Requires that self.points is not None.
        """
        if self.points is None:
            raise ValueError("No 3D points available for the scene.")
        
        # Assume self.points is of shape (H, W, 3); if not, adjust accordingly.
        geoms = [self.get_pcd(return_o3d=True)]
        orgin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        table_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        table_origin.translate([0.6, 0, 0])
        geoms.append(orgin)
        geoms.append(table_origin)
        if self.cam_extrinsics is not None:
            transformed_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            transformed_origin.transform(self.cam_extrinsics)
            geoms.append(transformed_origin)

        o3d.visualization.draw_geometries(geoms)

    def _crop_with_margin(self, mask: np.ndarray, margin: float) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Crops the scene image based on the mask and adds a margin.
        
        Args:
            mask (np.ndarray): Binary mask.
            margin (float): Fraction of the scene dimensions to use as margin.
        
        Returns:
            Tuple[Image.Image, Tuple[int, int, int, int]]: The cropped image and the 
                bounding box (x_min, y_min, x_max, y_max) in the scene.
        """
        mask = (mask > 0).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(ys) == 0 or len(xs) == 0:
            raise ValueError("Empty mask provided.")
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        # For a PIL Image, size returns (width, height)
        w, h = self.image.size
        x_margin = int(margin * w)
        y_margin = int(margin * h)
        x_min = max(0, x_min - x_margin)
        x_max = min(w, x_max + x_margin)
        y_min = max(0, y_min - y_margin)
        y_max = min(h, y_max + y_margin)
        cropped = self.image.crop((x_min, y_min, x_max, y_max))
        return cropped, (x_min, y_min, x_max, y_max)

    def save(self, filepath: Optional[str] = None, save_previews: bool = True) -> None:
        """Saves the scene (with its objects) to disk using pickle.
        
        Args:
            filepath (Optional[str], optional): Optional file path. If not provided, defaults to
                "<current_directory>/processed/scene.pkl". Defaults to None.
            save_previews (bool, optional): Whether to also save images/visuals for the scene. 
                Defaults to True.
        """
        if filepath is None:
            filename = "scene.pkl"
            filepath = os.path.join(_DEFAULT_SAVE_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Scene saved to {filepath}")

        # Save the scene and object image as well.
        if save_previews:
            image_path = filepath.replace(".pkl", ".png")
            self.image.save(image_path)
            for i, obj in enumerate(self.objects):
                obj_path = filepath.replace("scene.pkl", f"obj_{i}.png")
                obj.get_cropped().save(obj_path)
            print(f"Scene image and object images saved to {os.path.dirname(filepath)}")

    @classmethod
    def load(cls, filename: str) -> "ImageScene":
        """
        Loads a scene from a pickle file.
        
        Args:
            filename (str): Path to the pickle file.
        
        Returns:
            ImageScene: The loaded scene.
        """
        with open(filename, "rb") as f:
            scene = pickle.load(f)
        print(f"Scene loaded from {filename}")
        return scene


class ImageSceneObject:
    def __init__(self, scene: ImageScene, mask: np.ndarray, margin: float = 0.1, name: Optional[str] = None) -> None:
        """Initialize an ImageSceneObject.
        
        Args:
            scene (ImageScene): The parent scene.
            mask (np.ndarray): Binary mask for the object.
            margin (float, optional): Fraction of the scene dimensions to use as margin. Defaults to 0.1.
            name (Optional[str], optional): Optional name for the object. Defaults to None.
        """
        self.scene: ImageScene = scene
        self.mask: np.ndarray = mask
        self.margin: float = margin
        self.name: str = name
        # Crop the scene image using the parent's helper method.
        self.cropped, self.bbox = self.scene._crop_with_margin(mask, margin)

    def get_cropped(self) -> Image.Image:
        """Returns the cropped image of the object."""
        return self.cropped
    
    def get_pcd(
        self,
        use_cutout: bool = True,
        return_o3d: bool = True,
        transform: bool = True,
    ) -> Union[o3d.geometry.PointCloud, Tuple[np.ndarray, np.ndarray]]:
        """Get a point cloud representation for this object.

        Args:
            use_cutout: If True, use only points under the object mask. If False,
                use points from the object's bounding box.
            return_o3d: If True, return an Open3D PointCloud. If False, return
                raw numpy arrays.
            transform: If True, apply the scene camera extrinsics.

        Returns:
            pcd_or_arrays: Either an Open3D point cloud or a tuple of (points, colors).
        """
        if self.scene.points is None:
            raise ValueError("Scene does not have 3D points available.")

        if use_cutout:
            # Use only the 3D points where the object is present.
            points_selected = self.scene.points[self.mask > 0]
            colors_selected = np.array(self.scene.image)[self.mask > 0] / 255.0
        else:
            # The object's bounding box in the scene image.
            x_min, y_min, x_max, y_max = self.bbox

            # Extract the corresponding region from the scene's points.
            points_box = self.scene.points[y_min:y_max, x_min:x_max, :]
            colors_box = np.array(self.scene.image)[y_min:y_max, x_min:x_max, :] / 255.0

            # Use all points in the cropped region.
            points_selected = points_box.reshape(-1, 3)
            colors_selected = colors_box.reshape(-1, 3)
        
        # Remove the nan points
        valid = ~np.isnan(points_selected).any(axis=-1)
        points_selected = points_selected[valid]
        colors_selected = colors_selected[valid]

        # Transform the points
        if transform:
            assert self.scene.cam_extrinsics is not None, "No camera extrinsics available for the scene."
            points_selected = transform_points(points_selected, self.scene.cam_extrinsics)

        if return_o3d:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_selected)
            pcd.colors = o3d.utility.Vector3dVector(colors_selected)
            return pcd
        else:
            return points_selected, colors_selected

    def map_coord(self, coord, in_type: str = "perc", out_type: str = "perc") -> tuple[float, float]:
        """
        Maps a coordinate from the cropped (local) image to the full scene.
        
        Args:
            coord (tuple): The coordinate in the cropped image.
            in_type (str): Format of the input coordinate:
                           "perc" (0–100), "frac" (0–1), or "abs" (absolute pixels).
            out_type (str): Desired output format: "perc", "frac", or "abs".
        
        Returns:
            tuple: The coordinate in the scene in the specified format.
        """
        x_min, y_min, _, _ = self.bbox
        # For PIL Images, size returns (width, height)
        cropped_w, cropped_h = self.cropped.size
        scene_w, scene_h = self.scene.image.size

        if in_type == "perc":
            local_abs_x = (coord[0] / 100.) * cropped_w
            local_abs_y = (coord[1] / 100.) * cropped_h
        elif in_type == "frac":
            local_abs_x = coord[0] * cropped_w
            local_abs_y = coord[1] * cropped_h
        elif in_type == "abs":
            local_abs_x, local_abs_y = coord
        else:
            raise ValueError("in_type must be 'perc', 'frac', or 'abs'")

        scene_abs_x = local_abs_x + x_min
        scene_abs_y = local_abs_y + y_min

        if out_type == "abs":
            return (scene_abs_x, scene_abs_y)
        elif out_type == "frac":
            return (scene_abs_x / scene_w, scene_abs_y / scene_h)
        elif out_type == "perc":
            return (scene_abs_x / scene_w * 100, scene_abs_y / scene_h * 100)
        else:
            raise ValueError("out_type must be 'perc', 'frac', or 'abs'")
        
    def map_coords(self, coords, in_type: str = "perc", out_type: str = "perc") -> list[tuple[float, float]]:
        """
        Maps a list of coordinates from the cropped (local) image to the full scene.
        
        Args:
            coords (list of tuples): Coordinates in the cropped image.
            in_type (str): Format of the input coordinates:
                           "perc" (0–100), "frac" (0–1), or "abs" (absolute pixels).
            out_type (str): Desired output format: "perc", "frac", or "abs".
        
        Returns:
            list of tuples: Coordinates in the scene in the specified format.
        """
        return [self.map_coord(coord, in_type, out_type) for coord in coords]
    
    def visualize(self) -> None:
        """Visualizes the cropped object image."""
        plt.imshow(self.cropped)
        plt.axis("off")
        plt.show()
    
    def visualize_3d(self, use_cutout: bool = False) -> None:
        """
        Visualizes the 3D point cloud corresponding to this scene object.
        
        Args:
            use_cutout (bool): 
                - If False (default), visualizes the entire cropped region’s 3D points.
                - If True, visualizes only the masked (cutout) 3D points.
        
        Note:
            This method requires that the parent scene has 3D points (i.e. scene.points is not None).
        """
        pcd = self.get_pcd(use_cutout=use_cutout, return_o3d=True)
        o3d.visualization.draw_geometries([pcd])
        

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ImageScene demo.")
    parser.add_argument("--image", required=True, help="Path to scene image file")
    args = parser.parse_args()

    # Load a scene image using PIL.
    scene_img = Image.open(args.image)
    scene = ImageScene(scene_img)

    model = LangSAM(sam_type="sam2.1_hiera_large")
    masks = predict_mask(model, scene_img, "hot glue gun", visualize=False)
    obj1 = scene.add_object(masks[0], margin=0.1, name="hot_glue_gun")
    obj1.visualize()

    masks = predict_mask(model, scene_img, "wood blocks", visualize=False)
    obj2 = scene.add_object(masks[0], margin=0.1, name="wood_blocks")
    obj2.visualize()

    table_mask = predict_mask(model, scene_img, "table", visualize=False)
    table = scene.add_object(table_mask[0], margin=0.1, name="table")

    scene.isolate_objects([obj1, obj2]).show()
