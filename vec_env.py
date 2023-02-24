import gym
import numpy as np
import random

from scipy.spatial.ckdtree import ordered_pairs

from rotate import helper_degree
from utils import *
from helper import helper


class vec_env(Network):
    def __init__(self):
        
        self.df = open('plots/action.txt', 'w')
        self.df1 = open('plots/energy.txt', 'w')
        self.df2 = open('plots/times.txt', 'w')
        self.df3 = open('plots/penalty.txt', 'w')
        self.df4 = open('plots/cost.txt', 'w')
        self.energy = 0
        self.time = 0
        self.cost = 0
        self.penalty = 0
        self.energies = []
        self.actions = []
        array_bound_low = []
        array_bound_high = []
        self.network = helper.make_network(helper)
        for vehicle in self.network.moving_vehicles:
            array_bound_low.append(0)
            array_bound_high.append(1)
            for mec in self.network.MECs:
                array_bound_low.append(0)
                array_bound_high.append(1)
        for vehicle in self.network.moving_vehicles:
            for mec in self.network.MECs:
                array_bound_low.append(0)
                array_bound_high.append(1)
        array_bound_low = np.array(array_bound_low, dtype=np.float32)
        array_bound_high = np.array(array_bound_high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.network.dic_map_action))
        self.observation_space = gym.spaces.Box(low=array_bound_low, high=array_bound_high, dtype=np.float32)
    
    def reset(self):
        self.network = helper.update_env(helper, self.network)
        self.time_slot = 185
        state = []
        for vehicle in self.network.moving_vehicles:
            x, y, vehicle_speed, vehicle_angle = helper.read_coordinate_vehicle(helper, vehicle)
            vehicle.x = x
            vehicle.y = y
            vehicle.vehicle_speed = vehicle_speed
            vehicle.vehicle_angle = vehicle_angle
        for vehicle in self.network.moving_vehicles:
            for task_index in range(1):
                c = self.network.task_cpu
                a = random.uniform(self.network.min_data_size, self.network.max_data_size)  # random between 1 - 2 MB
                task = Task(self.network.task_id_counter, a, c, self.network.task_deadline,
                            vehicle.id,
                            (185 - self.time_slot) * self.network.time_slot_length)
                state.append((a - self.network.min_data_size) / self.network.max_data_size)
                # state.append((task.c - self.network.min_task_cpu) / self.network.max_task_cpu)
                # state.append(task.deadline / self.network.task_deadline)
                vehicle.task_queue.append(task)
                self.network.task_id_counter = self.network.task_id_counter + 1
            self.network.unprocess_tasks.append(vehicle.task_queue.pop(0))
            for mec in self.network.MECs:
                state.append(helper.compute_distance(helper, mec, vehicle) / 978)
            
            for mec in self.network.MECs:
                helper_degre = helper_degree(vehicle.vehicle_angle, mec.x, mec.y,
                                             vehicle.x, vehicle.y)
                self.network.angles.append(helper_degre.deg)
                state.append(helper_degre.deg / 179)
        self.state = tuple(state)
        return np.array(self.state)
    
    def step(self, action):
        states = []
        reward, penalty, total_delay, cost, energy, org_time, org_energy = helper.reward_computation_new_(helper,
                                                                                                          self.network,
                                                                                                          action)
        self.energies.append(org_energy)
        self.actions.append(action)
        self.df1.write(str(org_energy))
        self.df1.write('\n')
        self.df2.write(str(org_time))
        self.df2.write('\n')
        self.df3.write(str(penalty))
        self.df3.write('\n')
        self.df4.write(str(cost))
        self.df4.write('\n')
        self.energy = self.energy + org_energy
        self.time = self.time + org_time
        self.cost = self.cost + cost
        self.penalty = self.penalty + penalty
        for vehicle in self.network.moving_vehicles:
            x, y, vehicle_speed, vehicle_angle = helper.read_coordinate_vehicle(helper, vehicle)
            vehicle.x = x
            vehicle.y = y
            vehicle.vehicle_speed = vehicle_speed
            vehicle.vehicle_angle = vehicle_angle
        # states.append(total_delay)
        # states.append(penalty)
        for vehicle in self.network.moving_vehicles:
            for task_index in range(1):
                c = self.network.task_cpu
                a = random.uniform(self.network.min_data_size, self.network.max_data_size)  # random between 1 - 2 MB
                task = Task(self.network.task_id_counter, a, c, self.network.task_deadline, vehicle.id,
                            (185 - self.time_slot) * self.network.time_slot_length)
                states.append((a - self.network.min_data_size) / self.network.max_data_size)
                # states.append((task.c - self.network.min_task_cpu) / self.network.max_task_cpu)
                # states.append(task.deadline / self.network.task_deadline)
                vehicle.task_queue.append(task)
                self.network.task_id_counter = self.network.task_id_counter + 1
            self.network.unprocess_tasks.append(vehicle.task_queue.pop(0))
            
            for mec in self.network.MECs:
                states.append(helper.compute_distance(helper, mec, vehicle) / 978)
            
            for mec in self.network.MECs:
                helper_degre = helper_degree(vehicle.vehicle_angle, mec.x, mec.y,
                                             vehicle.x, vehicle.y)
                self.network.angles.append(helper_degre.deg)
                states.append(helper_degre.deg / 179)
        
        self.time_slot = self.time_slot - 1
        if self.time_slot <= 0:
            done = True
            self.df.write(str(self.actions))
            print(self.time, self.energy, self.penalty, self.cost)
            self.energy = 0
            self.time = 0
            self.cost = 0
            self.penalty = 0
        else:
            done = False
        info = {}
        info["energy"] = org_energy
        self.state = tuple(states)
        
        #return np.array(self.state), reward, done, info, penalty, total_delay, cost, energy, org_time, org_energy
        return np.array(self.state), reward, done, info
    
    def seed(self, seed):
        np.random.seed(seed)
