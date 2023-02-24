import math

import numpy as np
from matplotlib import pyplot as plt


class helper_degree(object):
    def __init__(self, degree_rotate, x_rsu, y_rsu, x_v, y_v):
        self.degree_rotate = degree_rotate
        self.x_rsu = x_rsu
        self.y_rsu = y_rsu
        self.x_v = x_v
        self.y_v = y_v

        # plt.rcParams["figure.figsize"] = [7.00, 3.50]
        # plt.rcParams["figure.autolayout"] = True
        self.newX = 0 * math.cos(math.radians(360 - degree_rotate)) - 1 * math.sin(math.radians(360 - degree_rotate))
        self.newY = 0 * math.sin(math.radians(360 - degree_rotate)) + 1 * math.cos(math.radians(360 - degree_rotate))
        # data = np.array([[newX, newY]])
        # plt.quiver( data[:, 0], data[:, 1], color=['black'], scale=15)
        # plt.show()
        # plt.rcParams["figure.figsize"] = [7.00, 3.50]
        # plt.rcParams["figure.autolayout"] = True
        self.x_Z = x_rsu - x_v
        self.y_z = y_rsu - y_v
        self.deg = math.degrees(angle([self.newX, self.newY], [self.x_Z, self.y_z]))
    # data1 = np.array([[newX, newY],[x_Z, y_z]])
    # plt.quiver( data1[:, 0], data1[:, 1], color=['black','red'], scale=15)
    # plt.show()


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

