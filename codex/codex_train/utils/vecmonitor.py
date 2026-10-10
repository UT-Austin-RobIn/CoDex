import time
from typing import Optional, Union

import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.monitor import ResultsWriter

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv


def select_index_from_dict(data: Union[dict, np.ndarray], i: int):
    if isinstance(data, np.ndarray):
        return data[i]
    elif isinstance(data, dict):
        out = dict()
        for k, v in data.items():
            out[k] = select_index_from_dict(v, i)
        return out
    else:
        return data[i]


class ManiSkillVecMonitor(ManiSkillSB3VectorEnv):
    """A wrapper that combines ManiSkillSB3VectorEnv and VecMonitor functionalities"""
    
    def __init__(self, env: BaseEnv, filename: Optional[str] = None, info_keywords: tuple = ()):
        super().__init__(env)
        self.info_keywords = info_keywords
        self.episode_count = 0
        self.t_start = time.time()
        
        env_id = None
        if hasattr(env, "spec") and env.spec is not None:
            env_id = env.spec.id
            
        self.results_writer: Optional[ResultsWriter] = None
        if filename:
            self.results_writer = ResultsWriter(
                filename, 
                header={"t_start": self.t_start, "env_id": str(env_id)}, 
                extra_keys=info_keywords
            )
    
    def step_wait(self) -> VecEnvStepReturn:
        vec_obs, rews, terminations, truncations, infos = self._env.step(
            self.last_actions
        )

        self.episode_returns += rews
        self.episode_lengths += 1
        dones = terminations | truncations

        # Convert observations, rewards, and dones to numpy arrays
        vec_obs = vec_obs.cpu().numpy()
        rews = rews.cpu().numpy()
        dones = dones.cpu().numpy()
        new_infos = []

        for env_idx in range(self.num_envs):
            info = {}
            
            # Add info keywords from the original environment
            for key in self.info_keywords:
                if key in infos:
                    if isinstance(infos[key], torch.Tensor):
                        info_item = infos[key][env_idx].cpu().detach()
                        if info_item.dim() == 0:
                            info[key] = info_item.item()
                        else:
                            info[key] = info_item.numpy()
                    else:
                        info[key] = infos[key][env_idx]
            
            # Add monitoring info
            info["TimeLimit.truncated"] = (truncations[env_idx] and not terminations[env_idx]).cpu().numpy()
            
            episode_return = self.episode_returns[env_idx].cpu().item()
            episode_length = self.episode_lengths[env_idx].cpu().item()
            episode_info = {
                "r": episode_return, 
                "l": episode_length, 
                "t": round(time.time() - self.t_start, 6)
            }
            
            # Add any extra info keywords to the episode info
            for key in self.info_keywords:
                if key in info:
                    episode_info[key] = info[key]
            
            info["episode"] = episode_info
            
            # Store terminal observations
            if dones[env_idx]:
                if "success" in infos:
                    info["is_success"] = infos["success"][env_idx].cpu().numpy()
                info["terminal_observation"] = select_index_from_dict(
                    vec_obs, env_idx
                )
                
                # Write episode info if we have a results writer
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                self.episode_count += 1
            
            new_infos.append(info)

        # Reset environments where episodes have ended
        if dones.any():
            reset_indices = np.where(dones)[0]
            new_obs = self._env.reset(options=dict(env_idx=reset_indices))[0]
            vec_obs[reset_indices] = new_obs[reset_indices].cpu().numpy()
            self.episode_returns[reset_indices] = 0
            self.episode_lengths[reset_indices] = 0

        return vec_obs, rews, dones, new_infos
    
    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return super().close()