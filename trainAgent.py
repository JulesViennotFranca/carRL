import torch
import numpy as np
import pygame
import matplotlib.pyplot as plt
import time

import environment
import DQN
import config
import gamestate
import event
import player

def render(agent):
    eps_tamp = agent.eps
    agent.eps = 0
    mplayer = player.Machine(agent)
    game = gamestate.Game(mplayer, config.track_width, config.track_size)
    game.start()
    mplayer.set_game(game)
    event.reset()
    while not (event.closed() or game.is_game_over or game.is_win or event.quit()):
        event.update()
        game.update()
    pygame.quit()
    agent.eps = eps_tamp

def train_agent(agent, episodes, rewards, max_steps):
    agent.load(config.training_path)
    env = environment.Env(rewards, max_steps)
    total_steps = 0
    last_total_steps = 0
    rewards = []
    steps = []
    for i in range(episodes):
        obs = env.reset()
        obs = np.reshape(obs, [1, config.machine_obs_size])
        obs = torch.tensor(obs).float()
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs, eps=True)
            next_obs, reward, done, _ = env.step(action)
            next_obs = np.reshape(next_obs, [1, config.machine_obs_size])
            next_obs = torch.tensor(next_obs).float()
            agent.update_replay((obs, action, reward, next_obs, done))
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            if total_steps % config.training_learning_step == 0:
                agent.learn()
        print(f"Episode {i+1} Reward: {round(episode_reward, 1)} Total Stepy: {total_steps} Steps: {total_steps - last_total_steps}")
        rewards.append(episode_reward)
        steps.append(total_steps - last_total_steps)
        last_total_steps = total_steps
        if i % config.training_info_step == config.training_info_step - 1:
            print(f"-> Pourcentage {100 * (i+1) / episodes} Rewards mean: {round(sum(rewards[-config.training_info_step:]) / len(rewards[-config.training_info_step:]), 4)} Steps mean: {sum(steps[-config.training_info_step:]) / len(steps[-config.training_info_step:])} Epsilon: {round(agent.eps, 3)} Action Spand: {round(env.action_spand)}")
        if i % config.training_render_step == config.training_render_step - 1:
            render(agent)
    env.close()
    agent.save(config.training_path)

if __name__ == "__main__":
    agent = DQN.Agent(config.machine_obs_size, config.dqn_hidden_sizes, 9, config.dqn_lr, config.dqn_batchsize, config.dqn_capacity, config.dqn_gamma, config.dqn_eps_start, config.dqn_eps_end, config.dqn_eps_decay)
    train_agent(agent, config.training_episodes, config.training_rewards_info, config.training_max_steps)