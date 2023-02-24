import math


class Network(object):
    def __init__(self, dic_map_action={}, RSU_list=[], moving_vehicles_list=[], parked_vehicles_list=[], BS_list=[],
                 MECs=[]):
        self.bandwidth = 2000000  # 2 Mhz
        # self.bandwidth = 35000000
        self.channel_fading_coefficient = 1
        self.Gaussian_white_noise_power = pow(10, -14)  # -110 dbm to wat
        self.path_loss_exponent = 2
        self.vehicle_capability = 20 * pow(10, 9)  # Ghz
        #self.Rsu_capability = 100 * pow(10, 9)
        self.Rsu_capability = 100 * pow(10, 9)

        self.local_power_co = 0.2  # local computing power  v * c (c of task)
        self.local_power = pow(10, -31) * pow(self.vehicle_capability, 3)  # local computing power k * power (f , 3)
        self.dic_map_action = dic_map_action
        self.RSUs = RSU_list
        self.moving_vehicles = moving_vehicles_list
        self.parked_vehicles = parked_vehicles_list
        self.BS = BS_list
        self.computation_nodes_number = len(MECs) + len(moving_vehicles_list)
        self.unprocess_tasks = []
        self.MECs = MECs
        self.ptr = 1.3  # Vehicle transmitting power 24dbm
        self.local_power_idle = 0.2  # power which consumed when vehicle is idle
        self.time_slot_length = 0.1
        self.task_id_counter = 0
        self.times = []
        self.energies = []
        self.penalties = []
        self.transmision_rate = []
        self.number_of_penalty_task_locall = 0
        self.number_of_penalty_task_remote = 0
        self.number_of_penalty_task_Q = 0
        self.time_normal = []
        self.energy_normal = []
        self.distances = []
        self.path_1_x = []
        self.path_2_x = []
        self.path_3_x = []
        self.path_1_y = []
        self.path_2_y = []
        self.path_3_y = []
        #self.max_data_size = 8 * 8 * 1000000
        #self.min_data_size = 1 * 8 * 1000000
        self.max_data_size = 8 * 8 * 1000000
        self.min_data_size = 1 * 8 * 1000000
        # self.task_cpu = 600
        #self.task_cpu = 600
        self.task_cpu = 600

        self.min_unit_tr = 27019488
        self.max_unit_tr = 47328904  # tr for bandwidth 1mhz & 110 dbm

        # self.min_unit_tr = 23698136
        # self.max_unit_tr = 37692786  # tr for bandwidth 1mhz & 100 dbm
        self.angles = []
        self.min_local_t = (self.min_data_size * self.task_cpu) / self.vehicle_capability
        self.min_comp_t = (self.min_data_size * self.task_cpu) / self.Rsu_capability
        self.min_comm_t = (self.min_data_size) / (self.max_unit_tr * (self.bandwidth / 1000000))
        self.min_local_e = self.min_local_t * self.local_power
        self.min_comp_e = self.min_comp_t * self.local_power_idle
        self.min_comm_e = self.min_comm_t * self.ptr
        self.max_local_t = (self.max_data_size * self.task_cpu) / self.vehicle_capability
        self.max_comp_t = (self.max_data_size * self.task_cpu) / self.Rsu_capability
        self.max_comm_t = (self.max_data_size) / (self.min_unit_tr * (self.bandwidth / 1000000))
        self.max_local_e = self.max_local_t * self.local_power
        self.max_comp_e = self.max_comp_t * self.local_power_idle
        self.max_comm_e = self.max_comm_t * self.ptr
        self.min_time = min(self.min_local_t, self.min_comp_t, self.min_comm_t)
        self.max_time = max(self.max_local_t, self.max_comp_t, self.max_comm_t)
        self.min_energy = min(self.min_local_e, self.min_comp_e, self.min_comm_e)
        self.max_energy = max(self.max_local_e, self.max_comp_e, self.max_comm_e)
        max_local_e = self.max_local_e
        min_local_e = self.min_local_e
        max_local_t = self.max_local_t
        min_local_t = self.min_local_t
        max_off_e = self.max_comm_e + self.max_comp_e
        min_off_e = self.min_comm_e + self.min_comp_e
        max_off_t = self.max_comm_t + self.max_comp_t
        min_off_t = self.min_comm_t + self.min_comp_t
        # print(max_local_e, min_local_e, max_local_t, min_local_t, max_off_e, min_off_e, max_off_t, min_off_t)
        # self.task_deadline = 1.7
        self.task_deadline = 1.3
        self.number_all_penalties = 0
        self.deadine_penalties = 0
        self.distance_penalties = 0
        self.penalty_local = 0
        self.alfa = 0.5
        self.beta = 0.5
        self.gama = 1
        self.penaltycost = self.compute_penalty_cost()
        self.penaltycost_local = self.compute_penalty_cost_local()
        
        self.cost_local = []
        self.cost_offlaod = []
        self.locall_actions = 0
    
    def compute_penalty_cost(self):
        normalize_energy_comm = (self.max_comm_e - self.min_energy) / (self.max_energy - self.min_energy)
        normalize_energy_comp = (self.max_comp_e - self.min_energy) / (self.max_energy - self.min_energy)
        normalize_time_comm = (self.max_comm_t - self.min_time) / (self.max_time - self.min_time)
        normalize_time_comp = (self.max_comp_t - self.min_time) / (self.max_time - self.min_time)
        return self.alfa * (normalize_time_comm + normalize_time_comp) + self.beta * (
                normalize_energy_comm + normalize_energy_comp)
    
    def compute_penalty_cost_local(self):
        normalize_energy = (self.max_local_e - self.min_energy) / (self.max_energy - self.min_energy)
        normalize_time = (self.max_local_t - self.min_time) / (self.max_time - self.min_time)
        return self.alfa * (normalize_time) + self.beta * (normalize_energy)
    
    def compute_distance(self, mec, vehicle):
        return math.sqrt(pow(vehicle.x - mec.x, 2) + pow(vehicle.y - mec.y, 2))
    
    def compute_transmition_rate(self, distance):
        return self.bandwidth * math.log2(1 + ((pow(abs(self.channel_fading_coefficient), 2) * self.ptr) / (
                self.Gaussian_white_noise_power * pow(distance, self.path_loss_exponent))))
    
    def compute_local_computation_time(self, Task, v_capable):
        return ((Task.c * Task.a) / v_capable)
    
    # def compute_local_computation_energy(self, task, local_power_co):
    # return task.c * local_power_co * task.a
    
    def compute_local_computation_energy(self, local_computation_time):
        return self.local_power * local_computation_time
    
    def compute_Remote_computation_time(self, Task, MEC):
        return (Task.c * Task.a) / MEC.mec_capable
    
    def compute_Remote_communication_time(self, Task, transmition_rate):
        return Task.a / transmition_rate
    
    def compute_communication_time_queue(self, MEC, curent_time_slot, Task):
        waiting_time = 0
        for task in MEC.task_queue:
            waiting_time = waiting_time + (Task.c / MEC.mec_capable)
        # A = (((185 - curent_time_slot) * self.time_slot_length) + waiting_time) - Task.t_req
        A = waiting_time
        return A
    
    def compute_Remote_communication_energy(self, remote_communication_time):
        return remote_communication_time * self.ptr
    
    def compute_Remote_computation_energy(self, remote_computation_time):
        return remote_computation_time * self.local_power_idle


class Vehicle(object):
    def __init__(self, x, y, name, v_capable, status, id, coverage_radius):
        self.x = x
        self.y = y
        self.name = name
        self.status = status
        self.index_coordinates = 0
        self.vehicle_angle = 0
        self.vehicle_speed = 0
        self.data = 0
        self.v_capable = v_capable
        self.mec_capable = v_capable
        self.status = status  # moving vehicles = 1   parked vehicles = 0
        self.id = id
        self.task_queue = []
        self.coverage_radius = coverage_radius


class RSU(object):
    def __init__(self, x, y, mec_capable, coverage_radius):
        self.x = x
        self.y = y
        self.mec_capable = mec_capable
        self.coverage_radius = coverage_radius
        self.task_queue = []
        self.index_coordinates = 0


class BS(object):
    def __init__(self, x, y, mec_capable, coverage_radius):
        self.x = x
        self.y = y
        self.mec_capable = mec_capable
        self.coverage_radius = coverage_radius
        self.task_queue = []
        self.index_coordinates = 0


class Task(object):
    def __init__(self, id, a, c, deadline, vehicle_id, t_req):
        self.id = id
        self.a = a
        self.c = c
        self.vehicle_id = vehicle_id
        self.deadline = deadline
        self.t_req = t_req
        self.comm_r_checked = False



