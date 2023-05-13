import torch
import numpy as np
import pygame
import matplotlib.pyplot as plt
import time

import carenv
import DQN
import config
import gamestate
import event
import player

def render(agent):
    mplayer = player.MachinePlayer(agent)
    game = gamestate.Game(mplayer, config.track_width, config.track_size)
    mplayer.set_car_track(game.car, game.track)
    game.start()
    event.reset()
    while not (event.closed() or game.is_game_over or game.is_win or event.quit()):
        event.update()
        game.update()
    pygame.quit()

def train_agent(agent, episodes, rewards, max_steps):
    env = carenv.carEnv(rewards, max_steps)
    total_steps = 0
    last_total_steps = 0
    rewards = []
    steps = []
    update_steps = episodes // 100
    for i in range(episodes):
        obs = env.reset()
        obs = np.reshape(obs, [1, config.machine_nbr_search_dir])
        obs = torch.tensor(obs).float()
        done = False
        episode_reward = 0
        while not done:
            # print(total_steps)
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            next_obs = np.reshape(next_obs, [1, config.machine_nbr_search_dir])
            next_obs = torch.tensor(next_obs).float()
            agent.update_replay((obs, action, reward, next_obs, done))
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            if total_steps % 4 == 0:
                agent.learn()
        print(f"•  Episode {i+1} Reward: {episode_reward} Total Stepy: {total_steps} Steps: {total_steps - last_total_steps}")
        rewards.append(episode_reward)
        steps.append(total_steps - last_total_steps)
        last_total_steps = total_steps
        if i % update_steps == update_steps - 1:
            print(f"-> Pourcentage {100 * i / episodes} Rewards mean: {sum(rewards[-update_steps:]) / len(rewards[-update_steps:])} Steps mean: {sum(steps[-update_steps:]) / len(steps[-update_steps:])} Epsilon: {agent.eps}")
        if i % config.training_render_step == config.training_render_step - 1:
            render(agent)
    env.close()

if __name__ == "__main__":
    agent = DQN.DQNAgent(config.machine_nbr_search_dir, config.dqn_hidden_sizes, 9, config.dqn_lr, config.dqn_batchsize, config.dqn_capacity, config.dqn_gamma, config.dqn_eps_start, config.dqn_eps_end, config.dqn_eps_decay)
    train_agent(agent, config.training_episodes, config.training_rewards, config.training_max_steps)