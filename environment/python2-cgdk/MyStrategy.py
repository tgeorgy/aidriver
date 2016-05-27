from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World

import numpy
import requests

class MyStrategy:
    def move(self, me, world, game, move):
        """
        @type me: Car
        @type world: World
        @type game: Game
        @type move: Move
        """
        
        resp = requests.post('http://127.0.0.1:5010',data='test=1').json()
        move.engine_power = resp['engine']
        """
        move.engine_power = 1.0
        move.throw_projectile = True
        move.spill_oil = True

        if world.tick > game.initial_freeze_duration_ticks:
            move.use_nitro = True
        """