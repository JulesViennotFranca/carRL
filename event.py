import pygame 

import config

class GameEvent():
    def __init__(self, type, data):
        self.type = type
        self.data = data

def set_allowed_events():
    pygame.event.set_allowed([pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT])

events = {}

def reset():
    events["closed"] = False
    events["quit"] = False
    events["turn"] = 0
    events["acceleration"] = 0
    events["clicked"] = pygame.mouse.get_pressed()[0]

def update():
    events["clicked"] = pygame.mouse.get_pressed()[0]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            events["closed"] = True 
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                events["quit"] = True 
            if event.key == pygame.K_t:
                events["turn"] += config.turn_coef
            if event.key == pygame.K_s:
                events["turn"] -= config.turn_coef
            if event.key == pygame.K_v:
                events["acceleration"] += config.acceleration_front_coef
            if event.key == pygame.K_r:
                events["acceleration"] -= config.acceleration_back_coef
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_t:
                events["turn"] -= config.turn_coef
            if event.key == pygame.K_s:
                events["turn"] += config.turn_coef
            if event.key == pygame.K_v:
                events["acceleration"] -= config.acceleration_front_coef
            if event.key == pygame.K_r:
                events["acceleration"] += config.acceleration_back_coef

def closed():
    return events["closed"]

def quit():
    return events["quit"]

def turn():
    return events["turn"]

def acceleration():
    return events["acceleration"]

def clicked():
    return events["clicked"]