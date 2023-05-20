import pygame
import numpy as np

import track
import car
import config 
import event
import graphics
import machineActions

class GameBase():
    def __init__(self, player, track_width, track_size):
        self.player = player
        self.checkpoint = 1
        self.obs = np.zeros(config.machine_obs_size)
        self.track = track.TrackBase(track_width, track_size, self, self.obs)
        self.car = car.CarBase(self.player)

    def start(self):
        self.reset_state()
        self.reset_track()
        self.reset_car()

    def reset_state(self):
        self.is_game_over = False
        self.is_win = False
        self.pass_checkpoint = False

    def reset_track(self):
        self.checkpoint = 1
        self.obs = np.zeros(config.machine_obs_size)
        self.track.reset(self, self.obs)

    def reset_car(self):
        self.car.reset(self.track.start_pos, self.track.start_dir)
        self.closest_car_point = 0

    def update(self):
        self.closest_car_point = self.track.next_closest_track_point(self.closest_car_point, self.car.pos)
        self.obs = machineActions.observe(self)
        self.track.update(self.car.pos, self, self.obs)
        self.car.update()

        next_checkpoint = self.track.checkpoint_passed(self.car.pos, self.checkpoint)
        self.pass_checkpoint = self.checkpoint != next_checkpoint
        self.checkpoint = next_checkpoint

        self.collision()
        self.finish()

    def collision(self):
        self.is_game_over = not self.track.point_is_on_track(self.closest_car_point, self.car.pos)

    def finish(self):
        see_all_checkpoints = self.checkpoint == 0
        self.is_win = see_all_checkpoints
        
        
class Game(GameBase):
    def __init__(self, player, track_width, track_size):
        pygame.init()
        pygame.display.set_caption(config.window_caption)
        event.set_allowed_events()
        self.canvas = graphics.Canvas()
        self.fps_clock = pygame.time.Clock()

        super().__init__(player, track_width, track_size)
        self.track = track.TrackSprite(track_width, track_size, self, self.obs)
        self.car = car.CarSprite(player)

    def fps(self):
        return self.fps_clock.get_fps()

    def mark_one_frame(self):
        self.fps_clock.tick(config.fps_limit)

    def start(self):
        self.reset_state()
        self.reset_track()
        self.reset_car()

    def reset_state(self):
        super().reset_state()
        self.all_sprites = pygame.sprite.OrderedUpdates()
        self.canvas.reset()

    def reset_track(self):
        super().reset_track()
        self.all_sprites.add(self.track)

    def reset_car(self):
        super().reset_car()
        self.all_sprites.add(self.car)

    def update(self, update=True):
        super().update()
        self.all_sprites.clear(self.canvas.surface, self.canvas.background)
        self.all_sprites.draw(self.canvas.surface)
        if update:
            pygame.display.flip()
        self.mark_one_frame()
        