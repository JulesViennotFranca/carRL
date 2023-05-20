import gymnasium as gym
import random
import numpy as np
import time

import gamestate
import player
import machineActions
import config
import geometry

class carEnv(gym.Env):
    def __init__(self, rewards, max_steps):
        self.rewards = rewards
        self.max_steps = max_steps

    def reset(self):
        track_width = config.track_width # max(20, int(random.gauss(50, 10)))
        track_size = config.track_size # [max(500, int(random.gauss(2000, 500))) for _ in range(2)]
        self.player = player.TrainingPlayer()
        self.game = gamestate.GameBase(self.player, track_width, track_size)
        self.game.start()

        self.step_count = 0
        self.step_checkpoint_count = 0

        obs = machineActions.observe(self.game)
        return np.copy(obs) 
    
    def step(self, action):
        acc, turn = machineActions.act(action)
        self.player.set_action(acc, turn)

        for _ in range(config.machine_action_spand):
            self.game.update()
        
        obs = machineActions.observe(self.game)
        
        self.step_checkpoint_count += 1
        self.step_count += 1
        if self.game.is_game_over:
            reward = self.rewards["loose"] 
            done = True
        elif self.step_count > self.max_steps or self.step_checkpoint_count > self.max_checkpoint_count:
            reward = self.rewards["time_out"]
            done = True
        elif self.game.is_win:
            reward = self.rewards["win"] 
            done = True
        elif self.game.pass_checkpoint:
            reward = self.rewards["checkpoint"] 
            self.step_checkpoint_count = 0
            done = False
        else:
            reward = self.rewards["survive"]
            done = False

        return np.copy(obs), reward, done, {}