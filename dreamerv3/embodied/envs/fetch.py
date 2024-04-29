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
        self._observation_space = gym.spaces.Box(
            low=np.full((HEIGHT, WIDTH, 3), 0),
            high=np.full((HEIGHT, WIDTH, 3), 255),
            dtype=np.uint8)
        self._action_space = self._env.action_space
        self._env.model.geom_rgba[1:5] = 0  # Hide the lasers

        self._done = True
        self.keep_metrics = keep_metrics
        # self._viewer_setup()

    @property
    def obs_space(self):
        return{
            'image': embodied.Space(np.uint8, self._observation_space.shape),
            'goal_image': embodied.Space(np.uint8, self._observation_space.shape),
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'action': embodied.Space(np.float32, self._action_space.shape, low=-1, high=1),
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

        for _ in range(15):
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
            is_first = True
            s = self._reset()
        else:
            is_first = False
            s = self._env.step(action)[0]

        dist = np.linalg.norm(s['achieved_goal'] - self._goal)
        self._dist.append(dist)
        image = self.render()
        reward = float(dist < 0.05)
        self._done = dist < 0.05

        return self._obs(
            image,
            self._goal_img,
            reward,
            is_first=is_first,
            is_last=self._done,
        )

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
    
class FetchPush(embodied.Env):
    """Wrapped version of the FetchPush environment with image observations."""

    def __init__(self,
                 reward_mode='positive',  # positive: 0 or 1; negative: -1 or 0
                 keep_metrics=False):
        self.reward_mode = reward_mode
        self._dist = []
        self._dist_vec = []
        self._env = push.MujocoFetchPushEnv(render_mode='rgb_array', height=HEIGHT, width=WIDTH)
        self._observation_space = gym.spaces.Box(
            low=np.full((HEIGHT, WIDTH, 3), 0),
            high=np.full((HEIGHT, WIDTH, 3), 255),
            dtype=np.uint8)
        self._action_space = self._env.action_space
        self._env.model.geom_rgba[1:5] = 0  # Hide the lasers

        self._done = True
        self.keep_metrics = keep_metrics
        # self._viewer_setup()

    @property
    def obs_space(self):
        return{
            'image': embodied.Space(np.uint8, self._observation_space.shape),
            'goal_image': embodied.Space(np.uint8, self._observation_space.shape),
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'action': embodied.Space(np.float32, self._action_space.shape, low=-1, high=1),
            'reset': embodied.Space(bool),
        }

    def reset_metrics(self):
        self._dist_vec = []
        self._dist = []

    def _move_hand_to_obj(self):
        pass

    def _reset(self):
        if self._dist:  # if len(self._dist) > 0, ...
            self._dist_vec.append(self._dist)
        self._dist = []

        # generate the new goal image
        s = self._env.reset()[0]
        self._goal = s['desired_goal'].copy()

        for _ in range(15):
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
            is_first = True
            s = self._reset()
        else:
            is_first = False
            s = self._env.step(action)[0]

        dist = np.linalg.norm(s['achieved_goal'] - self._goal)
        self._dist.append(dist)
        image = self.render()
        reward = float(dist < 0.05)
        self._done = dist < 0.05

        return self._obs(
            image,
            reward,
            is_first=is_first,
            is_last=self._done,
        )

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
