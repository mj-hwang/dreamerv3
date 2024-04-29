# coding=utf-8
# This codebase is from "TD InfoNCE, Zheng et al."
# Copyright 2022 The Google Research Authors.
# Copyright 2023 Chongyi Zheng.

"""Utility for loading the OpenAI Gym Fetch robotics environments."""

import numpy as np

import gymnasium as gym
import gym.spaces

from gymnasium_robotics.envs.fetch import reach
from gymnasium_robotics.envs.fetch import push
from gymnasium_robotics.envs.fetch import pick_and_place
from gymnasium_robotics.envs.fetch import slide

import embodied

HEIGHT = 64
WIDTH = 64

class FetchReach(embodied.Env):
    """Wrapper for the FetchReach environment with image observations."""

    def __init__(self,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 keep_metrics=False):
        self.reward_mode = reward_mode
        self._dist = []
        self._dist_vec = []
        self._env = reach.MujocoFetchReachEnv(render_mode='rgb_array', height=HEIGHT, width=WIDTH)
        self.observation_space = gym.spaces.Box(
            low=np.full((64, 64, 3), 0),
            high=np.full((64, 64, 3), 255),
            dtype=np.uint8)
        self.action_space = self._env.action_space
        self._env.model.geom_rgba[1:5] = 0  # Hide the lasers

        self.keep_metrics = keep_metrics

        # self._viewer_setup()

    @property
    def obs_space(self):
        return{
            'image': embodied.Space(np.uint8, self.observation_space.shape),
            'goal_image': embodied.Space(np.uint8, self.observation_space.shape),
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'action': embodied.Space(np.float32, self.action_space.shape, low=-1, high=1),
            'reset': embodied.Space(bool),
        }

    def reset_metrics(self):
        self._dist_vec = []
        self._dist = []

    def _reset(self):
        if self._dist:  # if len(self._dist) > 0, ...
            self._dist_vec.append(self._dist)
        self._dist = []

        # generate the new goal image
        s = self._env.reset()[0]
        self._goal = s['desired_goal'].copy()

        for _ in range(10):
            hand = s['achieved_goal']
            obj = s['desired_goal']
            delta = obj - hand
            a = np.concatenate([np.clip(10 * delta, -1, 1), [0.0]])
            s = self._env.step(a)[0]

        self._goal_img = self.render()
        self._env.reset()

        s = self._env.reset()[0]
        return s

    def step(self, action):
        if action['reset'] or self._done:
            s = self._env.reset()[0]
        else:
            s = self._env.step(action)[0]

        dist = np.linalg.norm(s['achieved_goal'] - self._goal)
        self._dist.append(dist)
        self._done = False
        image = self.render()
        reward = float(dist < 0.05)

        if action['reset'] or self._done:
            return self._obs(image, self._goal_img, reward, is_first=True, is_last=self._done)
        else:
            return self._obs(image, self._goal_img, reward, is_last=self._done)

    def _obs(self, image, reward, is_first=False, is_last=False):
        return dict(
            image=image,
            goal_image=self._goal_image,
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_last,
        )
    
    def render(self):
        self._env.data.site_xpos[0] = 1_000_000
        img = self._env.render()
        return img

    def _viewer_setup(self):
        self._env._viewer_setup()
        self._env.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.5])
        self._env.viewer.cam.distance = 0.8
        self._env.viewer.cam.azimuth = 180
        self._env.viewer.cam.elevation = -30

    def close(self):
        return self._env.close()

# class FetchPush(embodied.Env):
#     """Wrapper for the FetchPush environment with image observations."""

#     def __init__(self, camera='camera2', start_at_obj=True, rand_y=False,
#                  reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
#                  ):
#         self.reward_mode = reward_mode
#         self._start_at_obj = start_at_obj
#         self._rand_y = rand_y
#         self._camera_name = camera
#         self._dist = []
#         self._dist_vec = []
#         self._env = push.MujocoFetchPushEnv(render_mode='rgb_array', height=HEIGHT, width=WIDTH)
#         self._old_observation_space = self.observation_space
#         self._new_observation_space = gym.spaces.Box(
#             low=np.full((64 * 64 * 6), 0),
#             high=np.full((64 * 64 * 6), 255),
#             dtype=np.uint8)
#         self.observation_space = self._new_observation_space
#         self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

#         self._viewer_setup()

#     def reset_metrics(self):
#         self._dist_vec = []
#         self._dist = []

#     def _move_hand_to_obj(self):
#         s = self._env._get_obs()
#         for _ in range(100):
#             hand = s['observation'][:3]
#             obj = s['achieved_goal'] + np.array([-0.02, 0.0, 0.0])
#             delta = obj - hand
#             if np.linalg.norm(delta) < 0.06:
#                 break
#             a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
#             s = self._env.step(a)[0]

#     def reset(self):
#         if self._dist:  # if len(self._dist) > 0 ...
#             self._dist_vec.append(self._dist)
#         self._dist = []

#         # generate the new goal image
#         self.observation_space = self._old_observation_space
#         s = self._env.reset()
#         self.observation_space = self._new_observation_space
#         # Randomize object position
#         for _ in range(8):
#             self._env.step(np.array([-1.0, 0.0, 0.0, 0.0]))
#         object_qpos = self.sim.data.get_joint_qpos('object0:joint')
#         if not self._rand_y:
#             object_qpos[1] = 0.75
#         self.sim.data.set_joint_qpos('object0:joint', object_qpos)
#         self._move_hand_to_obj()
#         self._goal_img = self.render()
#         block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3]
#         if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
#             print('Bad reset, recursing.')
#             return self.reset()
#         self._goal = block_xyz[:2].copy()

#         self.observation_space = self._old_observation_space
#         s = self._env.reset()
#         self.observation_space = self._new_observation_space
#         for _ in range(8):
#             self._env.step(np.array([-1.0, 0.0, 0.0, 0.0]))
#         object_qpos = self.sim.data.get_joint_qpos('object0:joint')
#         object_qpos[:2] = np.array([1.15, 0.75])
#         self.sim.data.set_joint_qpos('object0:joint', object_qpos)
#         if self._start_at_obj:
#             self._move_hand_to_obj()
#         else:
#             for _ in range(5):
#                 self._env.step(self.action_space.sample())

#         block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
#         img = self.render()
#         dist = np.linalg.norm(block_xyz[:2] - self._goal)
#         self._dist.append(dist)
#         if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
#             print('Bad reset, recursing.')
#             return self.reset()
#         return np.concatenate([img, self._goal_img])

#     def step(self, action):
#         s = self._env.step(action)[0]
#         block_xy = self.sim.data.get_joint_qpos('object0:joint')[:2]
#         dist = np.linalg.norm(block_xy - self._goal)
#         self._dist.append(dist)
#         done = False
#         is_success = float(dist < 0.05)  # Taken from the original task code.
#         img = self.render()
#         info = dict(
#             is_success=is_success,
#         )
#         r = get_reward(dist / 0.05, self.reward_mode)
#         return np.concatenate([img, self._goal_img]), r, done, info

#     def observation(self, observation):
#         self.sim.data.site_xpos[0] = 1_000_000
#         img = self.render()
#         return img.flatten()

#     def _viewer_setup(self):
#         self._env._viewer_setup()
#         if self._env._camera_name == 'camera1':
#             self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
#             self.viewer.cam.distance = 0.9
#             self.viewer.cam.azimuth = 180
#             self.viewer.cam.elevation = -40
#         elif self._env._camera_name == 'camera2':
#             self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
#             self.viewer.cam.distance = 0.65
#             self.viewer.cam.azimuth = 90
#             self.viewer.cam.elevation = -40
#         else:
#             raise NotImplementedError

#     def compute_reward(self, achieved_goal, goal, info):
#         # just image comparison
#         assert achieved_goal.shape == goal.shape, (achieved_goal.shape, goal.shape)
#         is_success = (achieved_goal == goal).all(axis=-1)
#         if self.reward_mode == 'positive':
#             r = is_success
#         else:
#             assert self.reward_mode == 'negative'
#             r = is_success - 1
#         return r


# class FetchSlide(embodied.Env):
#     """Wrapper for the FetchSlide environment with image observations."""

#     def __init__(self, camera='camera2', reward_mode='positive'):
#         self.reward_mode = reward_mode
#         self._camera_name = camera
#         self._dist = []
#         self._dist_vec = []
#         self._env = slide.MujocoFetchSlideEnv(render_mode='rgb_array', height=HEIGHT, width=WIDTH)
#         self._old_observation_space = self.observation_space
#         self._new_observation_space = gym.spaces.Box(
#             low=np.full((64 * 64 * 6), 0),
#             high=np.full((64 * 64 * 6), 255),
#             dtype=np.uint8)
#         self.observation_space = self._new_observation_space
#         self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

#     def reset_metrics(self):
#         self._dist_vec = []
#         self._dist = []

#     def _move_hand_to_obj(self):
#         s = self._env._get_obs()
#         for _ in range(100):
#             hand = s['observation'][:3]
#             obj = s['achieved_goal'] + np.array([-0.02, 0.0, 0.0])
#             delta = obj - hand
#             if np.linalg.norm(delta) < 0.06:
#                 break
#             a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
#             s = self._env.step(a)[0]

#     def _raise_hand(self):
#         s = self._env._get_obs()
#         for _ in range(100):
#             hand = s['observation'][:3]
#             target = hand + np.array([0.0, 0.0, 0.05])
#             delta = target - hand
#             if np.linalg.norm(delta) < 0.02:
#                 break
#             a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
#             s = self._env.step(a)[0]

#     def reset(self):
#         if self._dist:  # if len(self._dist) > 0 ...
#             self._dist_vec.append(self._dist)
#         self._dist = []

#         # generate the new goal image
#         self.observation_space = self._old_observation_space
#         s = self._env.reset()
#         self.observation_space = self._new_observation_space
#         object_qpos = self.sim.data.get_joint_qpos('object0:joint')
#         object_qpos[:3] = self.goal
#         self.sim.data.set_joint_qpos('object0:joint', object_qpos)
#         self._raise_hand()
#         self._goal_img = self.observation(s)
#         block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3]
#         self._goal = block_xyz[:2].copy()
#         old_goal = self.goal.copy()

#         self.observation_space = self._old_observation_space
#         s = self._env.reset()
#         self.observation_space = self._new_observation_space
#         self._move_hand_to_obj()
#         self.goal = old_goal  # set to the same goal as the goal image

#         block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
#         img = self.observation(s)
#         dist = np.linalg.norm(block_xyz[:2] - self._goal)
#         self._dist.append(dist)

#         return np.concatenate([img, self._goal_img])

#     def step(self, action):
#         s = self._env.step(action)[0]
#         block_xy = self.sim.data.get_joint_qpos('object0:joint')[:2]
#         dist = np.linalg.norm(block_xy - self._goal)
#         self._dist.append(dist)
#         done = False
#         is_success = float(dist < 0.05)  # Taken from the original task code.
#         img = self.observation(s)
#         info = dict(
#             is_success=is_success,
#         )
#         r = get_reward(dist / 0.05, self.reward_mode)
#         return np.concatenate([img, self._goal_img]), r, done, info

#     def observation(self, observation):
#         self.sim.data.site_xpos[0] = 1_000_000
#         img = self.render()
#         return img.flatten()

#     def render(self):
#         if not self._done:
#             self._prev_image = self._env.observations()['RGB_INTERLEAVED']
#         return self._prev_image

#     def _viewer_setup(self):
#         self._env._viewer_setup()
#         if self._camera_name == 'camera1':
#             self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
#             self.viewer.cam.distance = 0.9
#             self.viewer.cam.azimuth = 180
#             self.viewer.cam.elevation = -40
#         elif self._camera_name == 'camera2':
#             self.viewer.cam.lookat[Ellipsis] = np.array([1.35, 0.8, 0.4])
#             self.viewer.cam.distance = 1.75
#             self.viewer.cam.azimuth = 90
#             self.viewer.cam.elevation = -40
#         else:
#             raise NotImplementedError

#     def compute_reward(self, achieved_goal, goal, info):
#         # just image comparison
#         assert achieved_goal.shape == goal.shape, (achieved_goal.shape, goal.shape)
#         is_success = (achieved_goal == goal).all(axis=-1)
#         if self.reward_mode == 'positive':
#             r = is_success
#         else:
#             assert self.reward_mode == 'negative'
#             r = is_success - 1
#         return r
