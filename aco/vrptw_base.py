import numpy as np
import copy


class Node:
    def __init__(
        self,
        id: int,
        x: float,
        y: float,
        demand: float,
        ready_time: float,
        due_time: float,
        service_time: float,
    ):
        super()
        self.id = id

        if id == 0:
            self.is_depot = True
        else:
            self.is_depot = False

        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time


class VrptwGraph:
    def __init__(self, file_path, rho=0.1):
        super()
        # node_num node number
        # node_dist_mat distance between nodes (matrix)
        # pheromone_mat Information density on the path between nodes
        (
            self.node_num,
            self.nodes,
            self.node_dist_mat,
            self.vehicle_num,
            self.vehicle_capacity,
        ) = self.create_from_file(file_path)
        # rho pheromone evaporation rate
        self.rho = rho
        # Create a pheromone matrix

        (
            self.nnh_travel_path,
            self.init_pheromone_val,
            _,
        ) = self.nearest_neighbor_heuristic()
        self.init_pheromone_val = 1 / (self.init_pheromone_val * self.node_num)

        self.pheromone_mat = (
            np.ones((self.node_num, self.node_num)) * self.init_pheromone_val
        )
        # heuristic information matrix
        self.heuristic_info_mat = 1 / self.node_dist_mat

    def copy(self, init_pheromone_val):
        new_graph = copy.deepcopy(self)

        # Pheromones
        new_graph.init_pheromone_val = init_pheromone_val
        new_graph.pheromone_mat = (
            np.ones((new_graph.node_num, new_graph.node_num)) * init_pheromone_val
        )

        return new_graph

    def create_from_file(self, file_path):
        # Read the location of service points and customers from the file
        node_list = []
        with open(file_path, "rt") as f:
            count = 1
            for line in f:
                if count == 5:
                    vehicle_num, vehicle_capacity = line.split()
                    vehicle_num = int(vehicle_num)
                    vehicle_capacity = int(vehicle_capacity)
                elif count >= 10:
                    node_list.append(line.split())
                count += 1
        node_num = len(node_list)
        nodes = list(
            Node(
                int(item[0]),
                float(item[1]),
                float(item[2]),
                float(item[3]),
                float(item[4]),
                float(item[5]),
                float(item[6]),
            )
            for item in node_list
        )

        # Create distance matrix
        node_dist_mat = np.zeros((node_num, node_num))
        for i in range(node_num):
            node_a = nodes[i]
            node_dist_mat[i][i] = 1e-8
            for j in range(i + 1, node_num):
                node_b = nodes[j]
                node_dist_mat[i][j] = VrptwGraph.calculate_dist(node_a, node_b)
                node_dist_mat[j][i] = node_dist_mat[i][j]

        return node_num, nodes, node_dist_mat, vehicle_num, vehicle_capacity

    @staticmethod
    def calculate_dist(node_a, node_b):
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))

    def local_update_pheromone(self, start_ind, end_ind):
        self.pheromone_mat[start_ind][end_ind] = (1 - self.rho) * self.pheromone_mat[
            start_ind
        ][end_ind] + self.rho * self.init_pheromone_val

    def global_update_pheromone(self, best_path, best_path_distance):
        """
        Update pheromone matrix
        :return:
        """
        self.pheromone_mat = (1 - self.rho) * self.pheromone_mat

        current_ind = best_path[0]
        for next_ind in best_path[1:]:
            self.pheromone_mat[current_ind][next_ind] += self.rho / best_path_distance
            current_ind = next_ind

    def nearest_neighbor_heuristic(self, max_vehicle_num=None):
        index_to_visit = list(range(1, self.node_num))
        current_index = 0
        current_load = 0
        current_time = 0
        travel_distance = 0
        travel_path = [0]

        if max_vehicle_num is None:
            max_vehicle_num = self.node_num

        while len(index_to_visit) > 0 and max_vehicle_num > 0:
            nearest_next_index = self._cal_nearest_next_index(
                index_to_visit, current_index, current_load, current_time
            )

            if nearest_next_index is None:
                travel_distance += self.node_dist_mat[current_index][0]

                current_load = 0
                current_time = 0
                travel_path.append(0)
                current_index = 0

                max_vehicle_num -= 1
            else:
                current_load += self.nodes[nearest_next_index].demand

                dist = self.node_dist_mat[current_index][nearest_next_index]
                wait_time = max(
                    self.nodes[nearest_next_index].ready_time - current_time - dist, 0
                )
                service_time = self.nodes[nearest_next_index].service_time

                current_time += dist + wait_time + service_time
                index_to_visit.remove(nearest_next_index)

                travel_distance += self.node_dist_mat[current_index][nearest_next_index]
                travel_path.append(nearest_next_index)
                current_index = nearest_next_index
        # Finally return to the depot
        travel_distance += self.node_dist_mat[current_index][0]
        travel_path.append(0)

        vehicle_num = travel_path.count(0) - 1
        return travel_path, travel_distance, vehicle_num

    def _cal_nearest_next_index(
        self, index_to_visit, current_index, current_load, current_time
    ):
        """
        Find the nearest reachable next_index
        :param index_to_visit:
        :return:
        """
        nearest_ind = None
        nearest_distance = None

        for next_index in index_to_visit:
            if current_load + self.nodes[next_index].demand > self.vehicle_capacity:
                continue

            dist = self.node_dist_mat[current_index][next_index]
            wait_time = max(self.nodes[next_index].ready_time - current_time - dist, 0)
            service_time = self.nodes[next_index].service_time
            # Check whether you can return to the service shop after visiting a certain passenger
            if (
                current_time
                + dist
                + wait_time
                + service_time
                + self.node_dist_mat[next_index][0]
                > self.nodes[0].due_time
            ):
                continue

            # Cannot serve passengers outside due time
            if current_time + dist > self.nodes[next_index].due_time:
                continue

            if (
                nearest_distance is None
                or self.node_dist_mat[current_index][next_index] < nearest_distance
            ):
                nearest_distance = self.node_dist_mat[current_index][next_index]
                nearest_ind = next_index

        return nearest_ind


class PathMessage:
    def __init__(self, path, distance):
        if path is not None:
            self.path = copy.deepcopy(path)
            self.distance = copy.deepcopy(distance)
            self.used_vehicle_num = self.path.count(0) - 1
        else:
            self.path = None
            self.distance = None
            self.used_vehicle_num = None

    def get_path_info(self):
        return self.path, self.distance, self.used_vehicle_num
