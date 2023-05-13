import gymnasium as gym
import random
import numpy as np
import time

import gamestate
import player
import config
import interactions
import geometry

def action_to_physics(action):
    acc = action // 3 - 1
    if acc < 0:
        acc *= config.acceleration_back_coef
    elif acc > 0:
        acc *= config.acceleration_front_coef
    turn = (action % 3 - 1) * config.turn_coef
    return acc, turn

class carEnv(gym.Env):
    def __init__(self, rewards, max_steps):
        self.rewards = rewards
        self.max_steps = max_steps

    def reset(self):
        self.survive_reward = self.rewards["survive"]
        self.checkpoint_reward = self.rewards["checkpoint"]
        self.win_reward = self.rewards["win"]
        self.loose_reward = self.rewards["loose"]
        self.gamma = 1 - self.survive_reward / self.checkpoint_reward

        track_width = config.track_width # max(20, int(random.gauss(50, 10)))
        track_size = config.track_size # [max(500, int(random.gauss(2000, 500))) for _ in range(2)]
        self.player = player.TrainingPlayer()
        self.game = gamestate.GameBase(self.player, track_width, track_size)
        self.game.start()
        
        self.d = interactions.dist_to_next_checkpoint(self.game.car.pos, self.game.track, self.game.checkpoint)

        self.step_count = 0
        self.step_checkpoint_count = 0

        self.obs = player.get_first_obs(self.game.car, self.game.track)
        return np.copy(self.obs) 
    
    def step(self, action):
        # t = time.time()
        acc, turn = action_to_physics(action)
        self.player.set_action(acc, turn)

        for _ in range(config.machine_action_spand):
            self.game.update()
        # nt = time.time()
        # print("game update time", nt - t)
        # t = nt

        # print("get obs")
        # t = time.time()
        self.obs = player.get_next_obs(self.obs, self.game.car, self.game.track)
        # nt = time.time()
        # print("get obs time", nt - t)

        # print("c:", self.game.checkpoint)

        self.step_count += 1
        if self.game.is_game_over or self.step_count > self.max_steps:
            reward = self.loose_reward # / self.step_count
            done = True
        else:
            if self.game.is_win:
                reward = self.win_reward # / self.step_checkpoint_count
                done = True
            elif self.game.pass_checkpoint:
                reward = self.checkpoint_reward # / self.step_checkpoint_count
                done = False
                # self.step_checkpoint_count = 0
                # self.survive_reward = self.rewards["survive"]
                self.d = interactions.dist_to_next_checkpoint(self.game.car.pos, self.game.track, self.game.checkpoint)
                # print("d:", self.d)
            else:
                dtnc = interactions.dist_to_next_checkpoint(self.game.car.pos, self.game.track, self.game.checkpoint)
                # print("dtnc:", dtnc)
                acc, _ = action_to_physics(action)
                reward = self.survive_reward + acc
                # self.survive_reward *= self.gamma
                # self.step_checkpoint_count += abs(reward)
                done = False

        # print("r:", reward)

        return np.copy(self.obs), reward, done, {}