import pygame

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

WIDTH = 20
HEIGHT = 20
MARGIN = 5

class ExampleSetup():
    from environment.example import Actions

    colors = {
        '#': WHITE,          # Impassable wall
        '0': GREEN,          # Player 1
        '1': BLUE,           # Player 2
        ' ': BLACK,          # Background
        'missing': RED,
        'background': BLACK
    }

    action_map = {
        pygame.K_RIGHT: Actions.RIGHT,
        pygame.K_LEFT: Actions.LEFT,
        pygame.K_UP: Actions.STAY,
    }

class PrisonSetup():
    from environment.prison import Actions

    colors = {
        '#': WHITE,          # Impassable wall
        '0': GREEN,          # Player 1
        '1': BLUE,           # Player 2
        ' ': BLACK,          # Background
        'C': GREEN,
        'D': RED,
        'missing': RED,
        'background': BLACK
    }

    action_map = {
        pygame.K_RIGHT: Actions.RIGHT,
        pygame.K_LEFT: Actions.LEFT,
        pygame.K_UP: Actions.STAY,
        pygame.K_SPACE: Actions.PUNISH
    }
