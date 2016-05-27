from math import *
from model.Car import Car
from model.World import World


class PlayerContext:
    def __init__(self, cars, world):
        self.cars = cars
        self.world = world