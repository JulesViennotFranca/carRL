import gymnasium as gym
import random
import numpy as np
import time

import gamestate
import player
import machineActions
import config
import geometry

class Env(gym.Env):
    def __init__(self, rewards, max_steps):
        self.rewards = rewards
        self.max_steps = max_steps
        self.action_spand = config.machine_action_spand_start

    def reset(self):
        track_width = max(20, int(random.gauss(50, 10)))
        track_size = [max(500, int(random.gauss(2000, 500))) for _ in range(2)]
        self.player = player.Training()
        self.game = gamestate.GameBase(self.player, track_width, track_size)
        self.game.start()

        if self.action_spand > config.machine_action_spand_end:
            self.action_spand *= config.machine_action_spand_decay

        self.step_count = 0
        self.checkpoint_step_count = 0
        self.max_checkpoint_steps = round(7 * track_width / 3)

        obs = machineActions.observe(self.game)
        return np.copy(obs) 
    
    def step(self, action):
        acc, turn = machineActions.act(action)
        self.player.set_action(acc, turn)

        reward = 0
        for _ in range(round(self.action_spand)):
            self.game.update()
        
            obs = machineActions.observe(self.game)
            
            self.checkpoint_step_count += 1
            self.step_count += 1
            if self.game.is_game_over:
                reward += self.rewards["loose"] 
                return np.copy(obs), reward, True, {}
            elif self.step_count > self.max_steps or self.checkpoint_step_count > self.max_checkpoint_steps:
                reward += self.rewards["time_out"]
                return np.copy(obs), reward, True, {}
            elif self.game.is_win:
                reward += self.rewards["win"] 
                return np.copy(obs), reward, True, {}
            elif self.game.pass_checkpoint:
                reward += self.rewards["checkpoint"] 
                self.checkpoint_step_count = 0
            else:
                reward += self.rewards["survive"] / self.max_checkpoint_steps

        return np.copy(obs), reward, False, {}