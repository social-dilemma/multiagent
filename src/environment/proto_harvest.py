"""
proto_harvest.py
Simon Mendelsohn

A game faintly reminiscent of Harvest.

The player goes around the board, collecting apples.  New apples spawn with some probability depending on how many nearby apples there are (more nearby applies --> increased chance of getting more apples).  

Keys: left, right, up, down - move the harvester.

"python proto_harvest.py 0" will use the first map, and "python proto_harvest.py 1" will use the second.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

import sys

from pycolab import ascii_art
from pycolab import cropping
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites
import random

MAZES_ART = [
    # Full of apples
    ['##############################',
     '#                            #',
     '#   @   @   @   @   @   @    #',
     '#    @   @   @   @   @   @   #',
     '#     @   @   @   @   @   @  #',
     '#  @   @   @   @   @   @     #',
     '#   @   @   @   @   @   @    #',
     '#    @   @   @   @   @   @   #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#   @   @   @   @   @   @    #',
     '#    @   @   @   @   @   @   #',
     '#     @   @   @   @   @   @  #',
     '#  @   @   @   @   @   @     #',
     '#   @   @   @   @   @   @    #',
     '#    @   @   @   @   @   @   #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#   @   @   @   @   @   @    #',
     '#    @   @   @   @   @   @   #',
     '#     @   @   @   @   @   @  #',
     '#  @   @   @   @   @   @     #',
     '#   @   @   @   @   @   @    #',
     '#    @   @   @   @   @   @   #',
     '#              P             #',
     '##############################'],
    
    # few apples
    ['##############################',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#     @   @   @   @   @   @  #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#                            #',
     '#              P             #',
     '##############################'],
]

COLOUR_FG = {' ': (0, 0, 0),        # Default black background
             '@': (999, 862, 110),  # Shimmering applies
             '#': (764, 0, 999),    # Fence around field
             'P': (0, 999, 999)}    # Player

COLOUR_BG = {'@': (0, 0, 0)}

STARTER_OFFSET = [(10, 0), (10, 0)]    # For levels 0, 1

def make_game(level):
  return ascii_art.ascii_art_to_game(
      MAZES_ART[level], what_lies_beneath=' ',
      sprites={
          'P': PlayerSprite},
      drapes={
          '@': CashDrape},
      update_schedule=['P', '@'],
      z_order='@P')

def make_croppers(level):

  return [
      # The player view.
      cropping.ScrollingCropper(rows=10, cols=30, to_track=['P'],
                                initial_offset=STARTER_OFFSET[level]),
  ]

class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player, the harvester."""

  def __init__(self, corner, position, character):
    """Constructor: just tells `MazeWalker` we can't walk through walls."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#')

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, things, layers  # Unused

    if actions == 0:    # go upward?
      self._north(board, the_plot)
    elif actions == 1:  # go downward?
      self._south(board, the_plot)
    elif actions == 2:  # go leftward?
      self._west(board, the_plot)
    elif actions == 3:  # go rightward?
      self._east(board, the_plot)
    elif actions == 4:  # stay put? (Not strictly necessary.)
      self._stay(board, the_plot)
    if actions == 5:    # just quit?
      the_plot.terminate_episode()


class CashDrape(plab_things.Drape):
  """A `Drape` handling all of the apples.

  This Drape detects when a player traverses a apples, removing the apple and
  crediting the player for the collection. Terminates if all apples are gone.  It also makes new apples with some probability if there are nearby apples.
  """

  def update(self, actions, board, layers, backdrop, things, the_plot):
    # If the player has reached an apple, credit one reward and remove the apple
    # from the scrolling pattern. If the player has obtained all apples, quit!
    player_pattern_position = things['P'].position

    if self.curtain[player_pattern_position]:
      the_plot.log('Coin collected at {}!'.format(player_pattern_position))
      the_plot.add_reward(100)
      self.curtain[player_pattern_position] = False
      
      chance = 0
      if not self.curtain.any(): the_plot.terminate_episode()

    for i in range(1, len(self.curtain) - 1):
      for j in range(1, len(self.curtain[i]) - 1):
        chance = 0.001 * (self.curtain[i-1][j-1] + self.curtain[i-1][j] + self.curtain[i-1][j+1] + self.curtain[i][j-1] + self.curtain[i][j+1] + self.curtain[i+1][j-1] + self.curtain[i+1][j] + self.curtain[i+1][j+1])
        if chance > random.uniform(0, 1) and self.curtain[i][j] == False:
          self.curtain[i][j] = True

def main(argv=()):
  level = int(argv[1]) if len(argv) > 1 else 0

  game = make_game(level)
  # Build the croppers we'll use to scroll around in it, etc.
  croppers = make_croppers(level)

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                       curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                       -1: 4,
                       'q': 5, 'Q': 5},
      delay=100, colour_fg=COLOUR_FG, colour_bg=COLOUR_BG,
      croppers=croppers)

  # Let the game begin!
  ui.play(game)


if __name__ == '__main__':
  main(sys.argv)

