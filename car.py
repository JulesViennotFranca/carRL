import numpy as np
import pygame 

import geometry
import config

class CarBase():
    def __init__(self, player):
        self.friction = config.friction_coef

        self.player = player

        self.reset(np.zeros(2, dtype = float), 0)

    def reset(self, position, direction):
        self.pos = np.array(position, dtype = float) 
        self.vel = 0
        self.dir = direction

    def update(self):
        acceleration, turn = self.player.update()
        self.vel += acceleration
        self.dir += turn 
        self.vel *= (1 - self.friction)
        self.pos += self.vel * geometry.angle_to_vector(self.dir)

class CarSprite(pygame.sprite.Sprite, CarBase):
    def __init__(self, player):
        pygame.sprite.Sprite.__init__(self)

        self.length = config.car_length
        self.width = config.car_width
        self.height = config.car_height
        self.color = config.car_color

        self.tire_radius = config.tire_radius
        self.tire_width = config.tire_width
        self.tire_color = config.tire_color

        CarBase.__init__(self, player)

    def reset(self, position, direction):
        CarBase.reset(self, position, direction)

        self.update_sprite()

    def update_sprite(self):
        dimension_radius = (np.sqrt(self.length ** 2 + self.width ** 2) + np.sqrt(self.tire_radius ** 2 + self.tire_width ** 2)) * config.car_scaling
        sprite_dimension = np.repeat(dimension_radius, 2)
        self.image = pygame.Surface(sprite_dimension).convert_alpha()
        self.image.fill((255, 255, 255, 0))

        tire_coordinate = np.array([[np.array([2 * self.tire_radius, self.tire_width]) * np.array([j, i * j]) * config.car_scaling * 0.5 for i in range(-1, 2, 2)] for j in range(-1, 2, 2)])
        tire_coordinate = np.array([[geometry.apply_rotation(self.dir, tire_coordinate[i][j]) for j in range(2)] for i in range(2)])
        tire_coordinate = np.reshape(tire_coordinate, (4, 2))
        tire_coordinate = list(tire_coordinate)

        car_coordinate = np.array([[np.array([self.length, self.width]) * np.array([j, i * j]) * config.car_scaling * 0.5 for i in range(-1, 2, 2)] for j in range(-1, 2, 2)])
        car_coordinate = np.array([[sprite_dimension * 0.5 + geometry.apply_rotation(self.dir, car_coordinate[i][j]) for j in range(2)] for i in range(2)])
        car_coordinate = np.reshape(car_coordinate, (4, 2))
        car_coordinate = list(car_coordinate)
        
        for cc in car_coordinate:
            one_tire_coordinate = [cc + tc for tc in tire_coordinate]
            pygame.draw.polygon(self.image, self.tire_color, one_tire_coordinate)
        
        pygame.draw.polygon(self.image, self.color, car_coordinate)

        front_coordinate = np.array([[np.array([self.length * 0.2, self.width * 0.8]) * np.array([j, i * j]) * config.car_scaling * 0.5 for i in range(-1, 2, 2)] for j in range(-1, 2, 2)])
        front_coordinate = np.array([[sprite_dimension * 0.5 + geometry.apply_rotation(self.dir, np.array([self.length * 0.8, 0]) + front_coordinate[i][j]) for j in range(2)] for i in range(2)])
        front_coordinate = np.reshape(front_coordinate, (4, 2))
        front_coordinate = list(front_coordinate)

        pygame.draw.polygon(self.image, (200, 200, 200), front_coordinate)

        self.rect = self.image.get_rect()
        self.top_left = config.screen_mid - dimension_radius
        self.rect.center = config.screen_mid.tolist()

    def update(self):
        CarBase.update(self)
        self.update_sprite()
    