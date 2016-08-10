from model.Car import Car
from model.Game import Game
from model.Move import Move
from model.World import World

import numpy as np
#from scipy.ndimage.interpolation import rotate
import cv2
import requests
#from matplotlib import pyplot
#import time

class MyStrategy:
    map_ = []
    x_ = 0
    y_ = 0
    pixel_per_tile = 20
    save_map_every = 50
    pass_every = 5

    img_acc = 0
    nwpi = 1
    finished = False
    action_ = 0
    dist_ = 0
    health_ = 1.0

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

        map_ = np.zeros((world.height*pixel_per_tile,
                    world.width*pixel_per_tile),dtype=np.int8)

        for i,col in enumerate(world.tiles_x_y):
            for j,tile in enumerate(col):
                map_[j*pixel_per_tile:(j+1)*pixel_per_tile,
                    i*pixel_per_tile:(i+1)*pixel_per_tile] = tile_dict[tile]

        for i in range(map_.shape[0]):
            for j in range(map_.shape[1]):
                if (i+j)%10 == 0 and map_[i,j] == 1:
                    map_[i,j] = 0
                if (i-j+1)%10 == 0 and map_[i,j] == 1:
                    map_[i,j] = 0

        return map_

    def get_direction(self, map_, me, world, game, n_next_pnt=1):
        size = self.pixel_per_tile

        start = me.next_waypoint_index
        n_next_pnt = min(n_next_pnt, len(world.waypoints)-start)
        next_pnt = [world.waypoints[start+i] for i in range(n_next_pnt)]
        next_pnt_pos = []
        for x,y in next_pnt:
            x_ = int(size*(x+0.5))
            y_ = int(size*(y+0.5))
            next_pnt_pos.append((x_,y_))

        prev_pnt = (int(self.x_),int(self.y_))
        for i, pnt in enumerate(next_pnt_pos):
            cv2.line(map_, prev_pnt, pnt, 2)
            prev_pnt = pnt

        # target tile highlight
        x0 = int(size*(next_pnt[0][0]))
        y0 = int(size*(next_pnt[0][1]))
        x1 = int(size*(next_pnt[0][0]+1))
        y1 = int(size*(next_pnt[0][1]+1))
        cv2.rectangle(map_, (x0,y0), (x1,y1), 2, thickness=1)

        return map_

    def move(self, me, world, game, move):
        """
        @type me: Car
        @type world: World
        @type game: Game
        @type move: Move
        """
        if world.tick == 0:
            self.map_ = self.init_map(world, game)
            rows,cols = self.map_.shape[:2]

            # opposite notation high x width
            h = me.width*self.pixel_per_tile/game.track_tile_size
            w = me.height*self.pixel_per_tile/game.track_tile_size
            d1 = (rows/2+w/2, cols/2+h/2)
            d2 = (rows/2-w/2, cols/2-h/2)
            d1 = tuple(np.int32(d1))
            d2 = tuple(np.int32(d2))
            # thickness=-1 for filled rect
            self.draw_car = lambda m: cv2.rectangle(m, d1, d2, 3, thickness=-1)
        else:
            rows,cols = self.map_.shape[:2]

        if world.tick < game.initial_freeze_duration_ticks or self.finished:

            return
        elif (world.tick % self.pass_every == 0 or
                me.finished_track or 
                world.tick == game.tick_count-1):

            self.x_ = me.x*self.pixel_per_tile/game.track_tile_size
            self.y_ = me.y*self.pixel_per_tile/game.track_tile_size

            alpha = np.pi/2-me.angle

            map_ = self.map_.__copy__()
            map_ = self.get_direction(map_, me, world, game)

            transform_mat = cv2.getRotationMatrix2D((self.x_,self.y_),-alpha/np.pi*180+180,1)
            transform_mat[0,2] += cols/2-self.x_ # Adding offsets to hold car position in a center
            transform_mat[1,2] += rows/2-self.y_

            state = cv2.warpAffine(np.float32(map_), transform_mat, (cols,rows),
                flags=9, borderValue=1) # flags = 0 for CV_INTER_NN
            self.draw_car(state)
            pixel_per_tile = self.pixel_per_tile
            state = state[rows/2-pixel_per_tile*1.75:rows/2+pixel_per_tile*0.25,
                          cols/2-pixel_per_tile:cols/2+pixel_per_tile]
            state = np.uint8(state*32)

            player = world.get_my_player()
            if self.nwpi != me.next_waypoint_index:
                reward = 10
                self.nwpi = me.next_waypoint_index
            else:
                reward = -5

            nwpx = (me.next_waypoint_x + 0.5) * game.track_tile_size
            nwpy = (me.next_waypoint_y + 0.5) * game.track_tile_size
            dist = me.get_distance_to(nwpx, nwpy)
            if dist < self.dist_-5:
                #reward += (self.dist_ - dist) / game.track_tile_size
                #reward += 5
                reward_add = (self.dist_ - dist) / 10.
                reward_add = min(reward_add, 10)
                reward += int(reward_add)
            if self.health_ > me.durability:
                reward = reward-2
            self.health_ = me.durability

            self.dist_ = dist

            # Save data transfer costs (check later)
            #state = np.uint8(state).reshape(-1)
            #n = len(state)
            #state = (state[:n/4] + state[n/4:n/2]*4 +
            #         state[n/2:3*n/4]*16 + state[3*n/4:]*64)

            if me.finished_track or world.tick == game.tick_count-1:
                terminate = 1
                self.finished = True
            else:
                terminate = 0

            state_s = state.tostring()
            reward_s = np.int32(reward).tostring()
            terminate_s = np.int8(terminate).tostring()

            #if world.tick % self.save_map_every == 0:
            #    pyplot.imsave('map_'+str(world.tick)+'.png', state)

            #with open('out.txt', 'w') as logfile:
            #    logfile.write(terminate_s+reward_s+state_s)
            headers = {'content-type':'application/json'}
            resp = requests.post('http://127.0.0.1:5010',
                                 data=terminate_s+reward_s+state_s,
                                 headers=headers).json()
            #resp = {'action':6}
            self.action_ = resp['action']

        if self.action_ == 0:
            # No action
            move.engine_power = 0.
            move.wheel_turn = 0.
        elif self.action_ == 1:
            # Forward
            move.engine_power = 1.
            move.wheel_turn = 0.
        elif self.action_ == 2:
            # Right
            move.engine_power = 0.
            move.wheel_turn = 1.
        elif self.action_ == 3:
            # Left
            move.engine_power = 0.
            move.wheel_turn = -1.
        elif self.action_ == 4:
            # Back
            move.engine_power = -1.
            move.wheel_turn = 0.
        elif self.action_ == 5:
            # Brake
            move.brake = True
            move.engine_power = 0.
            move.wheel_turn = 0.
        elif self.action_ == 6:
            # Forward + Right
            move.engine_power = 1.
            move.wheel_turn = 1.
        elif self.action_ == 7:
            # Forward + Left
            move.engine_power = 1.
            move.wheel_turn = -1.
        elif self.action_ == 8:
            # Back + Right
            move.engine_power = -1.
            move.wheel_turn = 1.
        elif self.action_ == 9:
            # Back + Left
            move.engine_power = -1.
            move.wheel_turn = -1.
        elif self.action_ == 10:
            # Brake + Right
            move.brake = True
            move.engine_power = 0.
            move.wheel_turn = 0.
        elif self.action_ == 11:
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
