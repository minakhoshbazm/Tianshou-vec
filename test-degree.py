import math

import numpy as np
from matplotlib import pyplot as plt


class helper_degree(object):
    def __init__(self, degree_rotate, x_rsu, y_rsu, x_v, y_v):
        self.degree_rotate = -44.63
        self.x_rsu = 705
        self.y_rsu = 810
        self.x_v = 1270.66
        self.y_v =722.81
def p():
        degree_rotate = 44.63
        x_rsu = 705
        y_rsu = 810
        x_v = 1270.66
        y_v = 722.81
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        newX = 0 * math.cos(math.radians(360 - degree_rotate)) - 1 * math.sin(math.radians(360 - degree_rotate))
        newY = 0 * math.sin(math.radians(360 - degree_rotate)) + 1 * math.cos(math.radians(360 - degree_rotate))
        data = np.array([[newX, newY]])
        plt.quiver( data[:, 0], data[:, 1], color=['black'], scale=15)
        plt.show()
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        x_Z =  x_rsu -  x_v
        y_z =  y_rsu -  y_v
        deg = math.degrees(angle([newX, newY], [x_Z, y_z]))
        data1 = np.array([[newX, newY],[x_Z, y_z]])
        plt.quiver( data1[:, 0], data1[:, 1], color=['black','red'], scale=15)
        plt.show()


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
if __name__ == '__main__':
    p()