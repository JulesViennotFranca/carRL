import numpy as np
import pygame 


# General variables
resolution = [1000, 500]
screen_mid = np.array(resolution) / 2
background_color = (255, 234, 193)
window_caption = "Racing car"
fps_limit = 60


# Menu variables
title = "Racing car"
title_size = 40
buttons = ["Start", "Exit"]
buttons_size = 20
text_color = (34, 44, 83)
text_selected_color = (0, 72, 144)
game_over_background_color = (0, 0, 0)
game_over_text = "Game over"
game_over_text_color = (255, 255, 255)
win_background_color = (255, 255, 255)
win_text = "Win"
win_text_color = (0, 0, 0)
margin = 20
spacing = 10
start_button = 1
exit_button = 2

def get_default_font(size):
    font_default = pygame.font.get_default_font()
    return pygame.font.Font(font_default, size)

# Car variables
start_position = [500, 250]
start_direction = 0

car_length = 3.5
car_width = 2
car_height = 1.5
car_mass = 1500
tire_radius = 0.35
tire_width = 0.4
car_scaling = 5
car_front_surface = 0.81 * car_width * car_height
car_color = (191, 0, 58)
tire_color = (0, 0, 0)

acceleration_front_coef = 0.5
acceleration_back_coef = 0.3
turn_coef = - 1 / 18
max_turn = np.pi / 3

# Physic variables
gravity = 9.81
rho_air = 1.2
Cx = 0.34
friction_coef = 0.05

# Track variables
track_size = [1500, 1500]
track_width = 45
track_color = (125, 125, 125)
track_nbr_points = 150
track_search_flattest_zone_size = 3

# DQN variables
dqn_hidden_sizes = [128, 128, 128, 128, 128]
dqn_lr = 0.0001
dqn_batchsize = 128
dqn_capacity = 10000
dqn_gamma = 0.99
dqn_eps_start = 0.9
dqn_eps_end = 0.05
dqn_eps_decay = 0.997

# Machine variables
machine_search_dirs = np.linspace(- np.pi, np.pi, 28, False)
machine_search_dirs = np.array(machine_search_dirs)
machine_know_self_vel = True
machine_know_track_dir = True
machine_obs_size = len(machine_search_dirs) + machine_know_self_vel + machine_know_track_dir
machine_dp = 2
machine_action_spand_start = 10
machine_action_spand_decay = 0.999
machine_action_spand_end = 1

# Training variables
training_episodes = 2000
training_render_step = 50
training_info_step = 20
training_learning_step = 6
training_rewards_info = {"loose": 0, "time_out": -5, "win": 14, "checkpoint": 14, "survive": - 4}
training_max_steps = 1000
training_path = "agents/decaying_action_spand/first_one.pt"
