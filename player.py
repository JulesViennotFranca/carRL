import event
import numpy as np

import geometry
import interactions
import config
import carenv

class Player():
    def __init__(self):
        pass

    def update(self):
        pass

class TrainingPlayer(Player):
    def __init__(self):
        super().__init__()

    def set_action(self, acc, turn):
        self.acc = acc 
        self.turn = turn

    def update(self):
        return self.acc, self.turn

class HumanPlayer(Player):
    def __init__(self):
        super().__init__()
    
    def update(self):
        return event.acceleration(), event.turn()
    
def get_first_obs(car, track):
    search_dirs = [car.dir - np.pi / 2 + i * np.pi / (config.machine_nbr_search_dir - 3) for i in range(config.machine_nbr_search_dir - 3)]
    search_dirs.append(car.dir - 3 * np.pi / 2)
    search_dirs.append(car.dir + np.pi)
    search_dirs.append(car.dir + 3 * np.pi / 2) 
    search_vecs = np.array([geometry.angle_to_vector(search_dir) for search_dir in search_dirs])
    obs = []
    for search_vec in search_vecs:
        d = track.width
        increase = track.width // 2
        while increase > config.machine_dp:
            if interactions.point_on_track(car.pos + d * search_vec, track):
                d += increase
            else:
                d -= increase
            increase //= 2
        obs.append(d)
    # obs.append(car.vel)
    return np.array(obs)

def get_next_obs(obs, car, track):
    search_dirs = [car.dir - np.pi / 2 + i * np.pi / (config.machine_nbr_search_dir - 3) for i in range(config.machine_nbr_search_dir - 3)]
    search_dirs.append(car.dir - 3 * np.pi / 2)
    search_dirs.append(car.dir + np.pi)
    search_dirs.append(car.dir + 3 * np.pi / 2)
    search_vecs = np.array([geometry.angle_to_vector(search_dir) for search_dir in search_dirs])
    obs = []
    for search_vec in search_vecs:
        d = track.width
        increase = track.width // 2
        while increase > config.machine_dp:
            if interactions.point_on_track(car.pos + d * search_vec, track):
                d += increase
            else:
                d -= increase
            increase //= 2
        obs.append(d)
    # obs.append(car.vel)
    return np.array(obs)

class MachinePlayer(Player):
    def __init__(self, engine):
        super(MachinePlayer, self).__init__()
        self.engine = engine
        self.action_counter = config.machine_action_spand

    def set_car_track(self, car, track):
        self.car = car
        self.track = track
        self.obs = get_first_obs(self.car, self.track)
    
    def update(self):
        if self.action_counter == config.machine_action_spand:
            self.obs = get_next_obs(self.obs, self.car, self.track)
            self.act = self.engine.act(self.obs)
            self.action_counter = 0
        self.action_counter += 1
        return carenv.action_to_physics(self.act)