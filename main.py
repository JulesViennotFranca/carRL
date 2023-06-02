import pygame

import event
import gamestate
import graphics
import config
import player

closed = False
while not closed:
    game = gamestate.Game(player.Human(), config.track_width, config.track_size)
    event.reset()
    button_pressed = graphics.draw_main_menu(game)

    if button_pressed == config.start_button:
        game.start()
        event.update()
        while not (event.closed() or game.is_game_over or game.is_win or event.quit()):
            event.update()
            game.update()
        
        if game.is_win:
            graphics.win(game)

        if game.is_game_over:
            graphics.game_over(game)
        
        closed = event.closed()

    if button_pressed == config.exit_button:
        closed = True

pygame.quit()