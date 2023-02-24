import random
import pandas as pd
import itertools
from utils import *


class helper:
    def make_network(self):
        RSU_list = []
        BS_list = []
        moving_vehicles_list = []
        parked_vehicles_list = []
        MECs = []
        dic_map_action = {}
        
        action_list = [p for p in itertools.product([0, 1, 2, 3, 4, 5], repeat=3)]
        # action_list = [p for p in itertools.product([0,1,2], repeat=3)]
        for action in range(len(action_list)):
            dic_map_action[action] = action_list[action]
        
        # vehicles list
        vehicle_1 = Vehicle(0, 0, 'BusFlowWestEast0.0', 20 * pow(10, 9), 1, 1, 100)
        moving_vehicles_list.append(vehicle_1)
        vehicle_2 = Vehicle(0, 0, 'VehicleFlowWestToEast.0', 20 * pow(10, 9), 1, 2, 100)
        moving_vehicles_list.append(vehicle_2)
        vehicle_3 = Vehicle(0, 0, 'VehicleFlowWestToEast_0.0', 20 * pow(10, 9), 1, 3, 100)
        moving_vehicles_list.append(vehicle_3)
        
        # RSUs
        rsu_1 = RSU(705, 807, mec_capable=100 * pow(10, 9), coverage_radius=125)
        RSU_list.append(rsu_1)
        MECs.append(rsu_1)
        
        rsu_2 = RSU(900, 710, mec_capable=100 * pow(10, 9), coverage_radius=125)
        RSU_list.append(rsu_2)
        MECs.append(rsu_2)
        
        rsu_3 = RSU(1150, 620, mec_capable=100 * pow(10, 9), coverage_radius=125)
        # rsu_3 = RSU(1205, 600, mec_capable=130 * pow(10, 9), coverage_radius=125)
        
        RSU_list.append(rsu_3)
        MECs.append(rsu_3)
        
        rsu_4 = RSU(1400, 590, mec_capable=100 * pow(10, 9), coverage_radius=125)
        RSU_list.append(rsu_4)
        MECs.append(rsu_4)
        
        rsu_5 = RSU(1400, 483, mec_capable=100 * pow(10, 9), coverage_radius=125)
        RSU_list.append(rsu_5)
        MECs.append(rsu_5)
        
        # rsu_6 = RSU(1513, 443, mec_capable=pow(10, 9), coverage_radius=125)
        # RSU_list.append(rsu_6)
        # MECs.append(rsu_6)
        
        # BS
        
        # BS center
        # bs_1 = BS(1200, 592, mec_capable = pow(10, 12), coverage_radius = 300)
        # BS_list.append(bs_1)
        # MECs.append(bs_1)
        
        # BS last
        # bs_2 = BS(1500, 510, mec_capable=pow(10, 12), coverage_radius=300)
        # BS_list.append(bs_2)
        # MECs.append(bs_2)
        
        # BS first
        # bs_2 = BS(614, 1010, mec_capable=pow(10, 12), coverage_radius=300)
        # BS_list.append(bs_2)
        # MECs.append(bs_2)
        
        self.data = pd.read_csv("vanet-trace-creteil-20130924-0700-0900.csv", sep=';')
        vehicle_1.data = self.data[self.data['vehicle_id'] == vehicle_1.name]
        vehicle_2.data = self.data[self.data['vehicle_id'] == vehicle_2.name]
        vehicle_3.data = self.data[self.data['vehicle_id'] == vehicle_3.name]
        
        return Network(dic_map_action, RSU_list, moving_vehicles_list, parked_vehicles_list, BS_list, MECs)
    
    def read_coordinate_vehicle(self, vehicle):
        
        data_vehicle = vehicle.data[vehicle.data['timestep_time'] == vehicle.index_coordinates]
        vehicle_speed = data_vehicle['vehicle_speed'].item()
        x = data_vehicle['vehicle_x'].item()
        y = data_vehicle['vehicle_y'].item()
        vehicle_angle = data_vehicle['vehicle_angle'].item()
        vehicle.index_coordinates = vehicle.index_coordinates + 1
        return x, y, vehicle_speed, vehicle_angle
    
    def update_env(self, network):
        network.task_id_counter = 0
        network.unprocess_tasks.clear()
        network.number_of_penalty_task_locall = 0
        network.number_of_penalty_task_remote = 0
        network.number_all_penalties = 0
        network.deadine_penalties = 0
        network.distance_penalties = 0
        network.penalty_local = 0
        network.number_of_penalty_task_Q = 0
        network.locall_actions = 0
        
        for vehicle in network.moving_vehicles:
            vehicle.task_queue.clear()
            vehicle.index_coordinates = 0
        
        for vehicle in network.parked_vehicles:
            vehicle.task_queue.clear()
            vehicle.index_coordinates = 0
        
        for rsu in network.RSUs:
            rsu.task_queue.clear()
            rsu.index_coordinates = 0
        
        for bs in network.BS:
            bs.task_queue.clear()
            bs.index_coordinates = 0
        
        return network
    
    def compute_distance(self, mec, vehicle):
        return math.sqrt(pow(vehicle.x - mec.x, 2) + pow(vehicle.y - mec.y, 2))
    
    def normalize(self, network, time, energy):
        normalize_time = (time - network.min_time) / (network.max_time - network.min_time)
        normalize_energy = (energy - network.min_energy) / (network.max_energy - network.min_energy)
        return normalize_time, normalize_energy
    
    def reward_computation_new_(self, network, action):
        acition_list = list(network.dic_map_action[action])
        time = 0
        penalty = 0
        energy = 0
        cost = 0
        cost_plot = 0
        org_time = 0
        org_energy = 0
        reward = 0
        missed_tasks_number = 0
        penalty_count = 0
        for index in range(len(acition_list)):
            penalty_deadline = 0
            penalty_coverage = 0
            penalty_deadline_locall = 0
            if acition_list[index] == 0:
                # local time & energy
                network.locall_actions = network.locall_actions + 1
                current_local_time = network.compute_local_computation_time(network.unprocess_tasks[index],
                                                                            network.vehicle_capability)
                if (current_local_time > network.unprocess_tasks[index].deadline):
                    # penalty_deadline_locall = network.penaltycost_local
                    # penalty = penalty + penalty_deadline_locall
                    missed_tasks_number = missed_tasks_number + 1
                    penalty_count = penalty_count + 1
                    network.penalty_local = network.penalty_local + 1
                current_local_energy = network.compute_local_computation_energy(current_local_time)
                normalized_time, normalized_energy = self.normalize(self, network, current_local_time,
                                                                    current_local_energy)
                
                time = time + normalized_time
                org_time = org_time + current_local_time
                energy = energy + normalized_energy
                org_energy = org_energy + current_local_energy
                cost = cost + network.alfa * (normalized_time) + network.beta * (
                    normalized_energy) + penalty_deadline_locall
                cost_plot = cost_plot + network.alfa * (normalized_time) + network.beta * (
                    normalized_energy)
                network.cost_local.append(
                    network.alfa * (normalized_time) + network.beta * (normalized_energy) + penalty_deadline_locall)
            else:
                # remote energy & time for communication
                distance = network.compute_distance(network.MECs[acition_list[index] - 1],
                                                    network.moving_vehicles[index])
                network.distances.append(distance)
                tr = network.compute_transmition_rate(distance)
                current_remote_time_comm = network.compute_Remote_communication_time(network.unprocess_tasks[index], tr)
                network.transmision_rate.append(tr)
                # remote energy & time for computation
                current_remote_time_comp = network.compute_Remote_computation_time(network.unprocess_tasks[index],
                                                                                   network.MECs[
                                                                                       acition_list[index] - 1])
                each_task_time = current_remote_time_comm + current_remote_time_comp
                if (each_task_time > network.unprocess_tasks[index].deadline):
                    # penalty_deadline = network.penaltycost
                    # penalty = penalty + penalty_deadline
                    missed_tasks_number = missed_tasks_number + 1
                    penalty_count = penalty_count + 1
                    network.number_of_penalty_task_remote = network.number_of_penalty_task_remote + 1
                    network.number_all_penalties = network.number_all_penalties + 1
                    network.deadine_penalties = network.deadine_penalties + 1
                
                if distance > network.MECs[acition_list[index] - 1].coverage_radius:
                    penalty_coverage = network.penaltycost
                    network.number_all_penalties = network.number_all_penalties + 1
                    network.distance_penalties = network.distance_penalties + 1
                    # penalty = penalty + penalty_coverage
                current_remote_energy_comm = network.compute_Remote_communication_energy(current_remote_time_comm)
                current_remote_energy_comp = network.compute_Remote_computation_energy(current_remote_time_comp)
                normalized_time_comp, normalized_energy_comp = self.normalize(self, network,
                                                                              current_remote_time_comp,
                                                                              current_remote_energy_comp)
                normalized_time_comm, normalized_energy_comm = self.normalize(self, network,
                                                                              current_remote_time_comm,
                                                                              current_remote_energy_comm)
                network.cost_offlaod.append(
                    network.alfa * (normalized_time_comp + normalized_time_comm) + network.beta * (
                            normalized_energy_comp + normalized_energy_comm) + network.gama * (
                            penalty_coverage + penalty_deadline))
                cost = cost + network.alfa * (normalized_time_comp + normalized_time_comm) + network.beta * (
                        normalized_energy_comp + normalized_energy_comm) + network.gama * (
                               penalty_coverage + penalty_deadline)
                cost_plot = cost_plot + network.alfa * (normalized_time_comp + normalized_time_comm) + network.beta * (
                        normalized_energy_comp + normalized_energy_comm)
                time = time + normalized_time_comp + normalized_time_comm
                org_time = org_time + current_remote_time_comp + current_remote_time_comm
                energy = energy + normalized_energy_comp + normalized_energy_comm
                org_energy = org_energy + current_remote_energy_comm + current_remote_energy_comp
        
        network.unprocess_tasks.clear()
        # network.penalties.append(penalty)
        network.times.append(org_time)
        network.energies.append(org_energy)
        if cost < 0:
            a = 0
        if (cost) != 0:
            reward = (3 - missed_tasks_number) / (cost)
        return reward, penalty_count, time, cost, energy, org_time, org_energy




























