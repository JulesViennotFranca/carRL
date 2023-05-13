import pygame

import track
import car
import config 
import event
import graphics
import interactions 

class GameBase():
    def __init__(self, player, track_width, track_size):
        self.player = player
        self.track = track.TrackBase(track_width, track_size)
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
        self.track.reset()

    def reset_car(self):
        self.car.reset(self.track.start_pos, self.track.start_dir)

    def update(self):
        self.track.update(self.car.pos)
        self.car.update()

        next_checkpoint = interactions.checkpoint_passed(self.car.pos, self.track, self.checkpoint)
        self.pass_checkpoint = self.checkpoint != next_checkpoint
        self.checkpoint = next_checkpoint

        self.collision()
        self.finish()

    def collision(self):
        self.is_game_over = not interactions.point_on_track(self.car.pos, self.track, mode="car")

    def finish(self):
        see_all_checkpoints = self.checkpoint == len(self.track.checkpoints)
        self.is_win = interactions.point_cross_finish(self.car.pos, self.track) and see_all_checkpoints
        
        
class Game(GameBase):
    def __init__(self, player, track_width, track_size):
        pygame.init()
        pygame.display.set_caption(config.window_caption)
        event.set_allowed_events()
        self.canvas = graphics.Canvas()
        self.fps_clock = pygame.time.Clock()

        super().__init__(player, track_width, track_size)
        self.track = track.TrackSprite(track_width, track_size)
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
        