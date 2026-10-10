from pathlib import Path
from typing import Optional, Set

import gymnasium as gym
import numpy as np

from mani_skill.utils import common
from mani_skill.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)
from mani_skill.utils.wrappers.record import RecordEpisode

class CustomRecordEpisode(RecordEpisode):
    """
    Enhanced version of RecordEpisode with:
    1. Option to render only a subset of environments
    2. Option for parallel video output (one file per environment)
    3. Selective display of info keys on video
    """

    def __init__(
        self,
        env: gym.Env,
        max_render_envs: Optional[int] = None,
        parallel_videos: bool = False,
        info_keys: Optional[Set[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        
        # Maximum number of environments to render
        self.max_render_envs = min(max_render_envs or self.num_envs, self.num_envs)
        
        # Whether to output separate videos for each environment
        self.parallel_videos = parallel_videos
        
        # Info keys to display on video
        self.info_keys = info_keys
        
        # Setup for parallel videos if enabled
        if self.parallel_videos:
            self.render_images_per_env = [[] for _ in range(self.max_render_envs)]
            self.video_nrows = int(np.sqrt(self.max_render_envs))
        
    def capture_image(self, infos=None):
        """Capture rendered image(s) with optional info overlay"""
        img = self.env.render()
        img = common.to_numpy(img)
        
        if len(img.shape) == 3:
            img = img[None]
            
        # Limit to max_render_envs
        if len(img) > self.max_render_envs:
            img = img[:self.max_render_envs]
            
        if infos is not None and self.info_keys is not None:
            for i in range(len(img)):
                # Create filtered info dictionary with only specified keys
                filtered_info = {}
                for k in self.info_keys:
                    if k in infos:
                        v = infos[k]
                        # Handle batched values
                        if hasattr(v, "__len__") and not isinstance(v, (str, dict)):
                            if i < len(v):
                                filtered_info[k] = v[i]
                        else:
                            filtered_info[k] = v
                
                # Apply info overlay to image
                img[i] = put_info_on_image(img[i], filtered_info)
                
        # Format output based on rendering mode
        if len(img.shape) > 3:
            if len(img) == 1 or self.parallel_videos:
                if len(img) == 1:
                    img = img[0]
            else:
                img = tile_images(img, nrows=self.video_nrows)
                
        return img

    def step(self, action):
        if self.save_video and self._video_steps == 0:
            # Save the first frame of the video
            rendered_img = self.capture_image()
            
            if self.parallel_videos:
                # Handle parallel video storage
                if isinstance(rendered_img, np.ndarray) and len(rendered_img.shape) == 3:
                    self.render_images_per_env[0].append(rendered_img)
                else:
                    for i in range(min(len(rendered_img), len(self.render_images_per_env))):
                        self.render_images_per_env[i].append(rendered_img[i])
            else:
                self.render_images.append(rendered_img)
                
        obs, rew, terminated, truncated, info = super(RecordEpisode, self).step(action)

        if self.save_video:
            self._video_steps += 1
            
            if self.info_on_video:
                # Create info dict with reward and selected keys
                display_info = {}
                
                # Add reward to display info
                display_info["reward"] = common.to_numpy(rew)
                
                # Add other specified info keys if available
                if self.info_keys is not None:
                    for k in self.info_keys:
                        if k in info:
                            display_info[k] = info[k]
                
                image = self.capture_image(display_info)
            else:
                image = self.capture_image()

            if self.parallel_videos:
                # Store images for parallel videos
                if isinstance(image, np.ndarray) and len(image.shape) == 3:
                    self.render_images_per_env[0].append(image)
                else:
                    for i in range(min(len(image), len(self.render_images_per_env))):
                        self.render_images_per_env[i].append(image[i])
            else:
                self.render_images.append(image)
                
            if (
                self.max_steps_per_video is not None
                and self._video_steps >= self.max_steps_per_video
            ):
                self.flush_video()
                
        self._elapsed_record_steps += 1
        return obs, rew, terminated, truncated, info
    
    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
    ):
        """Flush videos with support for parallel output"""
        if not self.parallel_videos:
            # Use parent implementation for tiled videos
            super().flush_video(name, suffix, verbose, ignore_empty_transition, save)
            return
            
        # Process each environment separately for parallel videos
        for env_idx in range(self.max_render_envs):
            images = self.render_images_per_env[env_idx]
            
            if len(images) == 0 or (ignore_empty_transition and len(images) == 1):
                continue
                
            if save:
                self._video_id += 1
                if name is None:
                    video_name = f"{self._video_id}_env{env_idx}"
                    if suffix:
                        video_name += "_" + suffix
                    if self._avoid_overwriting_video:
                        while (
                            Path(self.output_dir)
                            / (video_name.replace(" ", "_").replace("\n", "_") + ".mp4")
                        ).exists():
                            self._video_id += 1
                            video_name = f"{self._video_id}_env{env_idx}"
                            if suffix:
                                video_name += "_" + suffix
                else:
                    video_name = f"{name}_env{env_idx}"
                    
                images_to_video(
                    images,
                    str(self.output_dir),
                    video_name=video_name,
                    fps=self.video_fps,
                    verbose=verbose,
                )
                
        # Reset video tracking
        self._video_steps = 0
        self.render_images_per_env = [[] for _ in range(self.max_render_envs)]
        self.render_images = []