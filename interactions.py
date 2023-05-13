import numpy as np

import geometry
import config

def point_on_track(point, track, mode="normal"):
    if mode == "normal":
        focus_track = track.opt_track
    elif mode == "car":
        focus_track = track.limit_track
    dist_to_track = map(lambda p: geometry.get_norm(p - point), focus_track)
    return len(focus_track) == 0 or min(dist_to_track) <= track.width

def checkpoint_passed(point, track, checkpoint):
    if checkpoint < len(track.checkpoints):
        dir = track.checkpoints[checkpoint][1]
        v = point - track.checkpoints[checkpoint][0]
        vd = geometry.apply_rotation(- dir, v)
        diff_dir = geometry.get_angle(vd)
        max_vel = config.acceleration_front_coef * (1 - config.friction_coef) / config.friction_coef
        if geometry.get_norm(v) <= max(3 * track.width, max_vel) and abs(diff_dir) < np.pi / 2:
            return checkpoint + 1
    return checkpoint

def dist_to_next_checkpoint(point, track, checkpoint):
    if checkpoint < len(track.checkpoints):
        cp = track.checkpoints[checkpoint][0]
        return geometry.get_norm(cp - point)
    else:
        return -1
    
def point_cross_finish(point, track):
    finish_dir = track.start_dir
    v = point - track.start_pos
    vd = geometry.apply_rotation(- finish_dir, v)
    diff_dir = geometry.get_angle(vd)
    max_vel = config.acceleration_front_coef * (1 - config.friction_coef) / config.friction_coef
    return geometry.get_norm(v) <= max(3 * track.width, max_vel) and abs(diff_dir) < np.pi / 2