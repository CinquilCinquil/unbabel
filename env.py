import gymnasium
from gymnasium import spaces

import numpy as np
from typing import Union
from utils import AgentAction
from game_env import GameEnv

from stable_baselines3.common.type_aliases import GymResetReturn, GymStepReturn

class CustomEnv(gymnasium.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, env : GameEnv):
        super(CustomEnv, self).__init__()

        Speech = spaces.Box(low=-1, high=1, shape=(1, 120), dtype=np.float32)

        #TODO: make agent see which piece he has and make drop action

        # Which action | dx | dy | target agent | Speech
        self.action_space = spaces.Dict({
            "action" : spaces.Discrete(8),
            "dx" : spaces.Discrete(3),
            "dy" : spaces.Discrete(3),
            "agent" : spaces.Discrete(env.n_agents),
            "speech" : Speech
        })

        obs_dict = {
            "eyes" : spaces.Box(low=0, high=1, shape=(5, 5, 6), dtype=np.float32),
            "offer" : spaces.Box(low=0, high=1, shape=(1, 4), dtype=np.float32),
            "desired_piece" : spaces.Box(low=0, high=1, shape=(1, 2), dtype=np.float32)
        }
        for i in range(5):
            obs_dict["speech " + str(i + 1)] = Speech
        
        self.observation_space = spaces.Dict(obs_dict)
        
        self.env = env

    def step(
            self, action: AgentAction
        ) -> GymStepReturn:
        
        obs, reward, terminated, truncated = self.env.step(action)

        return obs, reward, terminated, truncated, {}

    def reset(
            self, seed: int = None, options: dict = None
        ) -> GymResetReturn:
        obs = self.env.reset()
        return (obs, {})

    def render(self, mode='human'):
        ...
