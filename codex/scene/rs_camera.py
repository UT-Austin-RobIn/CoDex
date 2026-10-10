from typing import Dict

import cv2
import numpy as np
import open3d as o3d

try:
    import pyrealsense2 as rs  # optional: only needed for live camera capture
except ImportError:  # pragma: no cover
    rs = None
from PIL import Image
import argparse

from codex.utils.utils_3d import remove_outliers_2d, depth_to_point_cloud, points_to_pcd

def apply_square_center_crop(img):
    h, w, _ = img.shape

    ms = min(h, w)

    hs = (h-ms)//2
    ws = (w-ms)//2
    img = img[hs:hs + ms, ws: ws + ms]
    return img

def resize_img(img, size=(256, 256)):
    img = np.array(cv2.resize(img.astype(float), dsize=size))
    return img

img_proc_func = lambda x: resize_img(apply_square_center_crop(x))

class RealSenseCamera:
    def __init__(self, serial_number=None, *args, **kwargs) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 is required for camera capture; install it with: pip install pyrealsense2")
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial_number:
            config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        p = self.pipeline.start(config)
        depth_intr = p.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        rgb_intr = p.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._img_shape  = [rgb_intr.width, rgb_intr.height, 3]
        self._depth_shape = [depth_intr.width, depth_intr.height, 1]
        self.align = rs.align(rs.stream.color)

    def stop(self):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is required for camera capture; install it "
                "with: pip install pyrealsense2"
            )
        self.pipeline.stop()

    def get_img(self) -> np.ndarray:
        return self.get_camera_obs()['image']
    
    def get_depth(self) -> np.ndarray:
        return self.get_camera_obs()['depth']

    def get_camera_obs(self) -> Dict[str, np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asarray(depth_frame.get_data()).astype(int) if depth_frame else None
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asarray(color_frame.get_data()).astype(int) if color_frame else None
        return {
            'image': color_image,
            'depth': depth_image,
        }
    
    def get_intrinsics(self):
        intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        intrinsic_mat = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return intrinsic_mat
    
    def get_points(self, remove_outlier=True, observation=None) -> np.ndarray:
        cam_obs = self.get_camera_obs() if observation is None else observation
        color = cam_obs['image']
        depth = cam_obs['depth']

        intrinsic_mat = self.get_intrinsics()
        points, output_colors = depth_to_point_cloud(depth, intrinsic_mat, color, keep_shape=True)
        points = points / 1000.0 # Convert to meters

        if remove_outlier:
            points, mask = remove_outliers_2d(points, nb_neighbors=20, std_ratio=2.0, visualize=False)

        return points, output_colors
    
    def get_pcd(self, remove_outlier=True, visualize=False) -> o3d.geometry.PointCloud:
        """
        Generate a point cloud from the camera's RGB and depth data.
        
        Returns:
            o3d.geometry.PointCloud: Point cloud object containing 3D points with color information.
        """        
        points, colors = self.get_points(remove_outlier=remove_outlier)
        pcd = points_to_pcd(points, colors, visualize=visualize)
        return pcd
    
    @property
    def img_shape(self):
        return self._img_shape

    @property
    def depth_shape(self):
        return self._depth_shape
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a preview image from a RealSense camera.")
    parser.add_argument("--serial-number", default=None, help="Optional RealSense serial number to open.")
    parser.add_argument("--warmup-frames", type=int, default=20, help="Number of frames to discard before capture.")
    parser.add_argument("--save-path", default=None, help="Optional image output path. If omitted, only preview is shown.")
    args = parser.parse_args()

    connected_devices = list(rs.context().devices)
    if not connected_devices:
        print("No RealSense device connected; skipping camera capture demo.")
        raise SystemExit(0)

    serial_number = args.serial_number
    if serial_number is None:
        serial_number = connected_devices[0].get_info(rs.camera_info.serial_number)
        print(f"Using RealSense device {serial_number}")

    cam = RealSenseCamera(serial_number=serial_number)
    try:
        for _ in range(args.warmup_frames):
            img = cam.get_img()
            depth = cam.get_depth()

        pil_image = Image.fromarray(img.astype(np.uint8))
        pil_image.show()

        if args.save_path:
            pil_image.save(args.save_path)
            print(f"Saved preview image to {args.save_path}")
    finally:
        cam.stop()
