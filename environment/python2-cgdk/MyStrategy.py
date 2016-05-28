from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World

import numpy as np
import requests

class MyStrategy:
    map = []

    def init_map(world, game):
        tile_size = float(game.track_tile_size)
        pixel_per_tile = 80
        
        border_size = 80.
        border_radius = 80.
        
        border_size_ = int(border_size*pixel_per_tile/tile_size)
        border_radius_ = int(border_radius*pixel_per_tile/tile_size)
        
        empty_corner = np.zeros(
            (pixel_per_tile/2,pixel_per_tile/2),
            dtype=np.int8)
        border = empty_corner.copy()
        outer_circle = empty_corner.copy()
        inner_circle = empty_corner.copy()
        
        border[0:border_size_,:] = 1
        
        circle = lambda y,x: (y**2+x**2) < border_radius_**2
        for i in range(pixel_per_tile/2):
            for j in range(pixel_per_tile/2):
                if circle(i,j):
                    inner_circle[i,j] = 1
        
        outer_circle[:border_size_,:] = 1
        outer_circle[:,:border_size_] = 1
        outer_circle[:border_size_*2-1,:border_size_*2-1] = 1
        center = (border_size_+border_radius_)
        center = (center,center)
        
        circle2 = lambda y,x: circle(y-center[0],x-center[1])
        for i in range(pixel_per_tile/2):
            for j in range(pixel_per_tile/2):
                if circle2(i,j):
                    outer_circle[i,j] = 0
        
        # Pipe tile
        P = np.concatenate((border,border),axis=1)
        P = np.concatenate((P,np.rot90(P,2)),axis=0)
        
        # Corner tile
        C = np.concatenate((outer_circle,border),axis=1)
        tmp = np.concatenate((border.T,np.rot90(inner_circle,2)),axis=1)
        C = np.concatenate((C,tmp),axis=0)
        
        # T-shape tile
        T = np.concatenate((np.rot90(inner_circle,1),
                            np.rot90(inner_circle,2)),axis=1)
        tmp = np.concatenate((border,border),axis=1)
        T = np.concatenate((tmp,T),axis=0)
        
        # Crossroad tile
        X = np.concatenate((inner_circle,np.rot90(inner_circle,3)),axis=1)
        X = np.concatenate((X,np.rot90(X,2)),axis=0)
        
        # Empty tile
        E = np.ones((pixel_per_tile,pixel_per_tile), dtype=np.int8)
        
        tile_dict = {
            0:E,
            1:P.T,
            2:P,
            3:C,
            4:np.rot90(C,3),
            5:np.rot90(C,1),
            6:np.rot90(C,2),
            7:np.rot90(T,3),
            8:np.rot90(T,1),
            9:np.rot90(T,2),
            10:T,
            11:X
        }
        
        map = np.zeros((world.height*pixel_per_tile,
                    world.width*pixel_per_tile),dtype=np.int8)
        
        for i,col in enumerate(world.tiles_x_y):
            for j,tile in enumerate(col):
                map[j*pixel_per_tile:(j+1)*pixel_per_tile,
                    i*pixel_per_tile:(i+1)*pixel_per_tile] = tile_dict[tile]
    
        return map
    
    def move(self, me, world, game, move):
        """
        @type me: Car
        @type world: World
        @type game: Game
        @type move: Move
        """
        
        if world.tick == 0:
            self.map = init_map(world, game)
        
        resp = requests.post('http://127.0.0.1:5010',data='test=1').json()
        move.engine_power = resp['engine']
        """
        move.engine_power = 1.0
        move.throw_projectile = True
        move.spill_oil = True
        if world.tick > game.initial_freeze_duration_ticks:
            move.use_nitro = True
        """
