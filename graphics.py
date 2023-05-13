import numpy as np
import pygame

import config
import event


class Canvas:
    def __init__(self):
        self.surface = pygame.display.set_mode(config.resolution)
        self.background = pygame.Surface(self.surface.get_size())
        self.background = self.background.convert()
        self.reset()

    def reset(self):
        self.background.fill(config.background_color)
        self.surface.blit(self.background, (0, 0))


def create_buttons(text, text_font, text_color_normal, text_color_on_hover):
    button_size = np.array([text_font[num].size(text[num]) for num in range(len(text))])

    buttons = [
        [text_font[num].render(text[num], False, text_color_normal[num]),
         text_font[num].render(text[num], False, text_color_on_hover[num])]
        for num in range(len(text))]

    screen_mid = config.resolution[0] / 2
    change_in_y = (config.resolution[1] -
                   config.margin * 2) / (len(buttons))
    screen_button_middles = np.stack((np.repeat([screen_mid], len(buttons)),
                                      np.arange(len(buttons)) * change_in_y), axis=1)

    text_starting_place = screen_button_middles + [-0.5, 0.5] * button_size
    text_ending_place = text_starting_place + button_size

    return buttons, button_size, text_starting_place, text_ending_place


def draw_main_menu(game_state):
    buttons, button_size, text_starting_place, text_ending_place = create_buttons(
        [config.title] + config.buttons,
        [config.get_default_font(config.title_size)] + [
            config.get_default_font(config.buttons_size)] * 3,
        [config.text_color] * 4,
        [config.text_color] + [config.text_selected_color] * 3)
    draw_rects(button_size, buttons, game_state, text_starting_place, emit=[0])
    button_clicked = iterate_until_button_press(buttons, game_state, text_ending_place, text_starting_place)

    return button_clicked


def iterate_until_button_press(buttons, game_state, text_ending_place, text_starting_place):
    button_clicked = 0
    while not button_clicked:
        pygame.display.update()
        event.update()
        for num in range(1, len(buttons)):
            if np.all((np.less(text_starting_place[num] - config.spacing, np.array(pygame.mouse.get_pos())),
                       np.greater(text_ending_place[num] + config.spacing, np.array(pygame.mouse.get_pos())))):
                if event.clicked():
                    button_clicked = num
                else:
                    game_state.canvas.surface.blit(
                        buttons[num][1], text_starting_place[num])
            else:
                game_state.canvas.surface.blit(
                    buttons[num][0], text_starting_place[num])
        if event.closed() or event.quit():
            button_clicked = len(buttons)-1
    return button_clicked


def draw_rects(button_size, buttons, game_state, text_starting_place, emit=list()):
    for num in range(len(buttons)):
        game_state.canvas.surface.blit(buttons[num][0], text_starting_place[num])
        if not num in emit:
            pygame.draw.rect(game_state.canvas.surface, config.text_color,
                             np.concatenate((text_starting_place[num] -
                                             config.spacing, button_size[num] +
                                             config.spacing * 2)), 1)

def game_over(game_state):
    font = config.get_default_font(config.title_size)
    rendered_text = font.render(config.game_over_text, False, config.game_over_text_color)
    game_state.canvas.background.fill(config.game_over_background_color)
    game_state.canvas.surface.blit(game_state.canvas.background, (0, 0))
    game_state.canvas.surface.blit(rendered_text, (np.array(config.resolution) - font.size(config.game_over_text)) / 2)
    pygame.display.flip()
    pygame.event.clear()

    exit = False 
    while not exit:
        event.update()
        exit = event.closed() or event.quit()

def win(game_state):
    font = config.get_default_font(config.title_size)
    rendered_text = font.render(config.win_text, False, config.win_text_color)
    game_state.canvas.background.fill(config.win_background_color)
    game_state.canvas.surface.blit(game_state.canvas.background, (0, 0))
    game_state.canvas.surface.blit(rendered_text, (np.array(config.resolution) - font.size(config.win_text)) / 2)
    pygame.display.flip()
    pygame.event.clear()

    exit = False 
    while not exit:
        event.update()
        exit = event.closed() or event.quit()
