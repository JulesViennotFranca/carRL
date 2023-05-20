import numpy as np

import config 
import geometry 

def search(pos, closest_point, search_vec, step_dist, limit_step_dist, track):
    dist = - step_dist
    while track.point_is_on_track(closest_point, pos + search_vec * (dist + step_dist)):
        dist += step_dist
        closest_point = track.next_closest_track_point(closest_point, pos + search_vec * (dist + step_dist))
    while step_dist > limit_step_dist:
        if track.point_is_on_track(closest_point, pos + search_vec * (dist + step_dist)):
            dist += step_dist
        step_dist /= 2
        closest_point = track.next_closest_track_point(closest_point, pos + search_vec * (dist + step_dist))
    return dist

def captor(game):
    search_dirs = config.machine_search_dirs + game.car.dir
    return np.array([geometry.angle_to_vector(search_dir) for search_dir in search_dirs])

def observe(game):
    search_vecs = captor(game)
    closest_point = game.closest_car_point
    obs = []
    for search_vec in search_vecs:
        d = search(game.car.pos, closest_point, search_vec, game.track.width, config.machine_dp, game.track)
        obs.append(d)
    if config.machine_know_self_vel:
        obs.append(game.car.vel)
    if config.machine_know_track_dir:
        track_dir = game.track.checkpoints[game.checkpoint][1]
        track_dir = (track_dir - game.car.dir) % (2 * np.pi)
        obs.append(track_dir)
    return np.array(obs)

def act(action):
    acc = action // 3 - 1
    if acc < 0:
        acc *= config.acceleration_back_coef
    elif acc > 0:
        acc *= config.acceleration_front_coef
    turn = (action % 3 - 1) * config.turn_coef
    return acc, turn