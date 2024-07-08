import cv2
import gymnasium as gym
import collections
import numpy as np

class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(FireResetEnv, self).__init__(env)

        try:
            assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        except AssertionError:
            raise Exception('ACTION 1 NOT FIRE. FireResetEnv Wrapper.')
        
        try:
            assert len(env.unwrapped.get_action_meanings()) >= 3
        except AssertionError:
            raise Exception('len(actions) < 3. FireResetEnv Wrapper.')

        def reset(self):
            '''
            i do not understand what happening here
            '''
            self.env.reset()
            obs, _, terminated, truncated, info = self.env.step(1)
            if terminated or truncated:
                self.env.reset()
            obs, _, terminated, truncated, info = self.env.step(2)
            if terminated or truncated:
                self.env.reset()
            return obs, info

        def step(self, ac):
            return self.env.step(ac)
    

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if (terminated or truncated):
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self):
        self._obs_buffer.clear()
        obs, info = self.env.reset()
        self._obs_buffer.append(obs)
        return obs, info 
    