from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World

import numpy as np
#from scipy.ndimage.interpolation import rotate
import cv2
import requests
from matplotlib import pyplot

class MyStrategy:
    map = []
    x_ = 0
    y_ = 0
    pixel_per_tile = 80
    save_map_every = 50
    img_acc = 0
    score_ = 0

    def init_map(self, world, game):
        tile_size = float(game.track_tile_size)
        pixel_per_tile = self.pixel_per_tile

        border_size = game.track_tile_margin
        border_radius = game.track_tile_margin

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
            0:E,                1:P.T,              2:P,
            3:C,                4:np.rot90(C,3),    5:np.rot90(C,1),
            6:np.rot90(C,2),    7:np.rot90(T,3),    8:np.rot90(T,1),
            9:np.rot90(T,2),    10:T,               11:X
        }

        map = np.zeros((world.height*pixel_per_tile,
                    world.width*pixel_per_tile),dtype=np.int8)

        for i,col in enumerate(world.tiles_x_y):
            for j,tile in enumerate(col):
                map[j*pixel_per_tile:(j+1)*pixel_per_tile,
                    i*pixel_per_tile:(i+1)*pixel_per_tile] = tile_dict[tile]

        return map

    def get_direction(self, me, world, game, n_next_pnt=3):
        size = self.pixel_per_tile

        start = me.next_waypoint_index
        next_pnt = [world.waypoints[start+i] for i in range(n_next_pnt)]
        next_pnt_pos = []
        for x,y in next_pnt:
            x_ = int(size*(x+0.5))
            y_ = int(size*(y+0.5))
            next_pnt_pos.append((x_,y_))

        direction_map = np.zeros(self.map.shape, dtype=np.int8)

        prev_pnt = (int(self.x_),int(self.y_))
        for i, pnt in enumerate(next_pnt_pos):
            cv2.line(direction_map, prev_pnt, pnt, len(next_pnt_pos)-i)
            prev_pnt = pnt

        return direction_map

    def move(self, me, world, game, move):
        """
        @type me: Car
        @type world: World
        @type game: Game
        @type move: Move
        """

        if world.tick == 0:
            self.map = self.init_map(world, game)
            rows,cols = self.map.shape[:2]

            # opposite notation high x width
            h = me.width*self.pixel_per_tile/game.track_tile_size
            w = me.height*self.pixel_per_tile/game.track_tile_size
            d1 = (rows/2+w/2, cols/2+h/2)
            d2 = (rows/2-w/2, cols/2-h/2)
            d1 = tuple(np.int32(np.round(d1)))
            d2 = tuple(np.int32(np.round(d2)))
            self.car_img = np.zeros((rows,cols,1))
            # thickness=-1 for filled rect
            cv2.rectangle(self.car_img, d1, d2, 1, thickness=-1)
        else:
            rows,cols = self.map.shape[:2]

        self.x_ = me.x*self.pixel_per_tile/game.track_tile_size
        self.y_ = me.y*self.pixel_per_tile/game.track_tile_size

        alpha = np.pi/2-me.angle

        direction_map = self.get_direction(me, world, game)

        pre_state = np.concatenate((self.map[:,:,np.newaxis],
                                direction_map[:,:,np.newaxis]), axis=2)

        transform_mat = cv2.getRotationMatrix2D((self.x_,self.y_),-alpha/np.pi*180+180,1)

        transform_mat[0,2] += cols/2-self.x_ # Adding offsets to hold car position in a center
        transform_mat[1,2] += rows/2-self.y_

        state = cv2.warpAffine(pre_state, transform_mat, (cols,rows),
            flags=0) # flags = 0 for CV_INTER_NN

        state = np.concatenate((state,self.car_img), axis=2)
        state = state[:-rows/4,cols/8:-cols/8,:]

        player = world.get_my_player()
        if self.score_ != player.score:
            reward = player.score - self.score_
        else:
            reward = -1

        #if world.tick % self.save_map_every == 0:
        #    pyplot.imsave('map_'+str(world.tick)+'.png', state)
        #resp = requests.post('http://127.0.0.1:5010',data='test=1').json()
        #move.engine_power = resp['engine']

        if resp['action'] == 0:
            # No action
            move.engine_power = 0.
            move.wheel_turn = 0.
        elif resp['action'] == 1:
            # Forward
            move.engine_power = 1.
            move.wheel_turn = 0.
        elif resp['action'] == 2:
            # Right
            move.engine_power = 0.
            move.wheel_turn = 1.
        elif resp['action'] == 3:
            # Left
            move.engine_power = 0.
            move.wheel_turn = -1.
        elif resp['action'] == 4:
            # Back
            move.engine_power = -1.
            move.wheel_turn = 0.
        elif resp['action'] == 5:
            # Brake
            move.brake = True
            move.engine_power = 0.
            move.wheel_turn = 0.
        elif resp['action'] == 6:
            # Forward + Right
            move.engine_power = 1.
            move.wheel_turn = 1.
        elif resp['action'] == 7:
            # Forward + Left
            move.engine_power = 1.
            move.wheel_turn = -1.
        elif resp['action'] == 8:
            # Back + Right
            move.engine_power = -1.
            move.wheel_turn = 1.
        elif resp['action'] == 9:
            # Back + Left
            move.engine_power = -1.
            move.wheel_turn = -1.
        elif resp['action'] == 10:
            # Brake + Right
            move.brake = True
            move.engine_power = 0.
            move.wheel_turn = 0.
        elif resp['action'] == 11:
            # Brake + Left
            move.brake = True
            move.engine_power = 0.
            move.wheel_turn = 0.

        """
        move.engine_power = 1.0
        move.throw_projectile = True
        move.spill_oil = True
        if world.tick > game.initial_freeze_duration_ticks:
            move.use_nitro = True
        """
