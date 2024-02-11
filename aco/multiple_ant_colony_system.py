import numpy as np
import random
from aco.vprtw_aco_figure import VrptwAcoFigure
from aco.vrptw_base import VrptwGraph, PathMessage
from aco.ant import Ant
from threading import Thread, Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import copy
import time
from multiprocessing import Process, Manager
from multiprocessing import Queue as MPQueue


class MultipleAntColonySystem:
    def __init__(
        self,
        graph: VrptwGraph,
        ants_num=10,
        beta=1,
        q0=0.1,
        whether_or_not_to_show_figure=True,
        runtime_in_minutes=5,
    ):
        super()
        # The location and service time information of graph nodes
        self.graph = graph
        # ants_num number of ants
        self.ants_num = ants_num
        # vehicle_capacity represents the maximum load of each vehicle
        self.max_load = graph.vehicle_capacity
        # beta heuristic information importance
        self.beta = beta
        # q0 represents the probability of directly selecting the next point with the highest probability
        self.q0 = q0
        manager = Manager()
        # self.best_path_distance = None
        self.best_path_distance = manager.Value("d", 0.0)  # Shared double
        # best path
        # self.best_path = None
        self.best_path = manager.list()  # shared list
        self.best_vehicle_num = None

        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure
        self.runtime_in_minutes = runtime_in_minutes

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        """
        Roulette
        :param index_to_visit: a list of N index (list or tuple)
        :param transition_prob:
        :return: selected index
        """
        # calculate N and max fitness value
        N = len(index_to_visit)

        # normalize
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob / sum_tran_prob

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]

    @staticmethod
    def new_active_ant(
        ant: Ant,
        vehicle_num: int,
        local_search: bool,
        IN: np.numarray,
        q0: float,
        beta: int,
        stop_event: Event,
    ):
        """
       Explore the map according to the specified vehicle_num. The vehicle num used cannot be more than the specified number. Both acs_time and acs_vehicle will use this method.
        For acs_time, you need to visit all nodes (the path is feasible), and try to find a path with a shorter travel distance.
        For acs_vehicle, the vehicle num used will be one less than the number of vehicles used by the currently found best path. To use fewer vehicles, try to visit the nodes. If all nodes are visited (the path is feasible), macs will be notified:param ant:
        :param vehicle_num:
        :param local_search:
        :param IN:
        :param q0:
        :param beta:
        :param stop_event:
        :return:
        """
        # print('[new_active_ant]: start, start_index %d' % ant.travel_path[0])

        # In new_active_ant, up to vehicle_num vehicles can be used, that is, it can contain at most vehicle_num+1 depot nodes. Since one starting node is used, only vehicle depots are left.
        unused_depot_count = vehicle_num

        # If there are still unvisited nodes, you can return to the depot
        while not ant.index_to_visit_empty() and unused_depot_count > 0:
            if stop_event.is_set():
                # print('[new_active_ant]: receive stop event')
                return

            # Calculate all next nodes that meet load and other constraints
            next_index_meet_constrains = ant.cal_next_index_meet_constrains()

            # If there is no next node that satisfies the restriction, return to the depot.
            if len(next_index_meet_constrains) == 0:
                ant.move_to_next_index(0)
                unused_depot_count -= 1
                continue

            # Start calculating the next node that meets the constraints and select the probability of each node
            length = len(next_index_meet_constrains)
            ready_time = np.zeros(length)
            due_time = np.zeros(length)

            for i in range(length):
                ready_time[i] = ant.graph.nodes[
                    next_index_meet_constrains[i]
                ].ready_time
                due_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].due_time

            delivery_time = np.maximum(
                ant.vehicle_travel_time
                + ant.graph.node_dist_mat[ant.current_index][
                    next_index_meet_constrains
                ],
                ready_time,
            )
            delta_time = delivery_time - ant.vehicle_travel_time
            distance = delta_time * (due_time - ant.vehicle_travel_time)

            distance = np.maximum(1.0, distance - IN[next_index_meet_constrains])
            closeness = 1 / distance

            transition_prob = ant.graph.pheromone_mat[ant.current_index][
                next_index_meet_constrains
            ] * np.power(closeness, beta)
            transition_prob = transition_prob / np.sum(transition_prob)

            # Directly select the node with the largest closeness according to probability
            if np.random.rand() < q0:
                max_prob_index = np.argmax(transition_prob)
                next_index = next_index_meet_constrains[max_prob_index]
            else:
                # Use the roulette algorithm
                next_index = MultipleAntColonySystem.stochastic_accept(
                    next_index_meet_constrains, transition_prob
                )

            # Update pheromone matrix
            ant.graph.local_update_pheromone(ant.current_index, next_index)
            ant.move_to_next_index(next_index)

        # If you have visited all the points, you need to return to the depot.
        if ant.index_to_visit_empty():
            ant.graph.local_update_pheromone(ant.current_index, 0)
            ant.move_to_next_index(0)

        # Insert unvisited points to ensure that the path is feasible
        ant.insertion_procedure(stop_event)

        # ant.index_to_visit_empty()==True means feasible
        if local_search is True and ant.index_to_visit_empty():
            ant.local_search_procedure(stop_event)

    @staticmethod
    def acs_time(
        new_graph: VrptwGraph,
        vehicle_num: int,
        ants_num: int,
        q0: float,
        beta: int,
        global_path_queue: Queue,
        path_found_queue: Queue,
        stop_event: Event,
    ):
        """
        For acs_time, you need to visit all nodes (the path is feasible), and try to find a path with a shorter travel distance.
        :param new_graph:
        :param vehicle_num:
        :param ants_num:
        :param q0:
        :param beta:
        :param global_path_queue:
        :param path_found_queue:
        :param stop_event:
        :return:
        """

        # Up to vehicle_num vehicles can be used, that is, among the depots containing at most vehicle_num+1 in the path, find the shortest path.
        # vehicle_num is set to be consistent with the current best_path
        # print("[acs_time]: start, vehicle_num %d" % vehicle_num)
        # Initialize the pheromone matrix
        global_best_path = None
        global_best_distance = None
        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        while True:
            # print("[acs_time]: new iteration")

            if stop_event.is_set():
                # print("[acs_time]: receive stop event")
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(
                    MultipleAntColonySystem.new_active_ant,
                    ant,
                    vehicle_num,
                    True,
                    np.zeros(new_graph.node_num),
                    q0,
                    beta,
                    stop_event,
                )
                ants_thread.append(thread)
                ants.append(ant)

            # You can use the result method here to wait for the thread to finish running
            for thread in ants_thread:
                thread.result()

            ant_best_travel_distance = None
            ant_best_path = None
            # Determine whether the path found by the ant is feasible and better than the global path
            for ant in ants:
                if stop_event.is_set():
                    # print("[acs_time]: receive stop event")
                    return

                # Get the current best path
                if not global_path_queue.empty():
                    info = global_path_queue.get()
                    while not global_path_queue.empty():
                        info = global_path_queue.get()
                    # print("[acs_time]: receive global path info")
                    (
                        global_best_path,
                        global_best_distance,
                        global_used_vehicle_num,
                    ) = info.get_path_info()

                # The shortest path calculated by path ants
                if ant.index_to_visit_empty() and (
                    ant_best_travel_distance is None
                    or ant.total_travel_distance < ant_best_travel_distance
                ):
                    ant_best_travel_distance = ant.total_travel_distance
                    ant_best_path = ant.travel_path

            # Global updates of pheromones are performed here
            new_graph.global_update_pheromone(global_best_path, global_best_distance)

            # Send the calculated current best path to macs
            if (
                ant_best_travel_distance is not None
                and ant_best_travel_distance < global_best_distance
            ):
                # print(
                #     "[acs_time]: ants' local search found a improved feasible path, send path info to macs"
                # )
                path_found_queue.put(
                    PathMessage(ant_best_path, ant_best_travel_distance)
                )

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    @staticmethod
    def acs_vehicle(
        new_graph: VrptwGraph,
        vehicle_num: int,
        ants_num: int,
        q0: float,
        beta: int,
        global_path_queue: Queue,
        path_found_queue: Queue,
        stop_event: Event,
    ):
        """
        For acs_vehicle, the vehicle num used will be one less than the number of vehicles used by the currently found best path. To use fewer vehicles, try to visit the nodes. If all nodes are visited (the path is feasible), macs will be notified
        :param new_graph:
        :param vehicle_num:
        :param ants_num:
        :param q0:
        :param beta:
        :param global_path_queue:
        :param path_found_queue:
        :param stop_event:
        :return:
        """
        # vehicle_num is set to one less than the current best_path
        # print("[acs_vehicle]: start, vehicle_num %d" % vehicle_num)
        global_best_path = None
        global_best_distance = None

        # Initialize path and distance using nearest_neighbor_heuristic algorithm
        current_path, current_path_distance, _ = new_graph.nearest_neighbor_heuristic(
            max_vehicle_num=vehicle_num
        )

        # Find unvisited nodes in the current path
        current_index_to_visit = list(range(new_graph.node_num))
        for ind in set(current_path):
            current_index_to_visit.remove(ind)

        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        IN = np.zeros(new_graph.node_num)
        while True:
            # print("[acs_vehicle]: new iteration")

            if stop_event.is_set():
                # print("[acs_vehicle]: receive stop event")
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(
                    MultipleAntColonySystem.new_active_ant,
                    ant,
                    vehicle_num,
                    False,
                    IN,
                    q0,
                    beta,
                    stop_event,
                )

                ants_thread.append(thread)
                ants.append(ant)

            # You can use the result method here to wait for the thread to finish running
            for thread in ants_thread:
                thread.result()

            for ant in ants:
                if stop_event.is_set():
                    # print("[acs_vehicle]: receive stop event")
                    return

                IN[ant.index_to_visit] = IN[ant.index_to_visit] + 1

                # The path found by the ant is compared with the current_path to see whether vehicle_num vehicles can be used to access more nodes.
                if len(ant.index_to_visit) < len(current_index_to_visit):
                    current_path = copy.deepcopy(ant.travel_path)
                    current_index_to_visit = copy.deepcopy(ant.index_to_visit)
                    current_path_distance = ant.total_travel_distance
                    # and set IN to 0
                    IN = np.zeros(new_graph.node_num)

                    # If this path is feasible, it must be sent to macs_vrptw.
                    if ant.index_to_visit_empty():
                        # print(
                        #     "[acs_vehicle]: found a feasible path, send path info to macs"
                        # )
                        path_found_queue.put(
                            PathMessage(ant.travel_path, ant.total_travel_distance)
                        )

            # Update the pheromone in new_graph, global
            new_graph.global_update_pheromone(current_path, current_path_distance)

            if not global_path_queue.empty():
                info = global_path_queue.get()
                while not global_path_queue.empty():
                    info = global_path_queue.get()
                # print("[acs_vehicle]: receive global path info")
                (
                    global_best_path,
                    global_best_distance,
                    global_used_vehicle_num,
                ) = info.get_path_info()

            new_graph.global_update_pheromone(global_best_path, global_best_distance)

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    def run_multiple_ant_colony_system(self, file_to_write_path=None):
        """
        Start another thread to run multiple_ant_colony_system, and use the main thread to draw
        :return:
        """
        path_queue_for_figure = MPQueue()
        multiple_ant_colony_system_thread = Process(
            target=self._multiple_ant_colony_system,
            args=(
                path_queue_for_figure,
                file_to_write_path,
            ),
        )
        multiple_ant_colony_system_thread.start()

        # Whether to display figure
        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph.nodes, path_queue_for_figure)
            figure.run()
        multiple_ant_colony_system_thread.join()
        # print("Finished:", self.best_path)
        # print("Finished distance:", self.best_path_distance)
        # print("Finished from manager:", self.best_path[:])

    def _multiple_ant_colony_system(
        self, path_queue_for_figure: MPQueue, file_to_write_path=None
    ):
        """
        Call acs_time and acs_vehicle to explore paths
        :param path_queue_for_figure:
        :return:
        """
        if file_to_write_path is not None:
            file_to_write = open(file_to_write_path, "w")
        else:
            file_to_write = None

        start_time_total = time.time()

        # Two queues are needed here, time_what_to_do and vehicle_what_to_do, to tell the two threads acs_time and acs_vehicle what the current best path is, or to stop them from calculating.
        global_path_to_acs_time = Queue()
        global_path_to_acs_vehicle = Queue()

        # Another queue, path_found_queue, is a feasible path calculated by receiving acs_time and acs_vehicle that is better than the best path.
        path_found_queue = Queue()

        # Initialize using nearest neighbor algorithm
        (
            self.best_path[:],
            self.best_path_distance.value,
            self.best_vehicle_num,
        ) = self.graph.nearest_neighbor_heuristic()
        path_queue_for_figure.put(
            PathMessage(self.best_path, self.best_path_distance.value)
        )

        while True:
            # print("[multiple_ant_colony_system]: new iteration")
            start_time_found_improved_solution = time.time()

            # The current best path information is placed in the queue to inform acs_time and acs_vehicle what the current best_path is.
            global_path_to_acs_vehicle.put(
                PathMessage(self.best_path, self.best_path_distance.value)
            )
            global_path_to_acs_time.put(
                PathMessage(self.best_path, self.best_path_distance.value)
            )

            stop_event = Event()

            # acs_vehicle, try to explore with self.best_vehicle_num-1 vehicles and visit more nodes
            graph_for_acs_vehicle = self.graph.copy(self.graph.init_pheromone_val)
            acs_vehicle_thread = Thread(
                target=MultipleAntColonySystem.acs_vehicle,
                args=(
                    graph_for_acs_vehicle,
                    self.best_vehicle_num - 1,
                    self.ants_num,
                    self.q0,
                    self.beta,
                    global_path_to_acs_vehicle,
                    path_found_queue,
                    stop_event,
                ),
            )

            # acs_time tries to explore with self.best_vehicle_num vehicles to find a shorter path
            graph_for_acs_time = self.graph.copy(self.graph.init_pheromone_val)
            acs_time_thread = Thread(
                target=MultipleAntColonySystem.acs_time,
                args=(
                    graph_for_acs_time,
                    self.best_vehicle_num,
                    self.ants_num,
                    self.q0,
                    self.beta,
                    global_path_to_acs_time,
                    path_found_queue,
                    stop_event,
                ),
            )

            # Start acs_vehicle_thread and acs_time_thread. When they find a feasible and better path than the best path, they will be sent to macs.
            # print("[macs]: start acs_vehicle and acs_time")
            acs_vehicle_thread.start()
            acs_time_thread.start()

            best_vehicle_num = self.best_vehicle_num

            while acs_vehicle_thread.is_alive() and acs_time_thread.is_alive():
                # If no better results are found within the specified time, exit the program
                # given_time = 5
                if (
                    time.time() - start_time_found_improved_solution
                    > 60 * self.runtime_in_minutes
                ):
                    stop_event.set()
                    # self.print_and_write_in_file(file_to_write, "*" * 50)
                    # self.print_and_write_in_file(
                    #     file_to_write,
                    #     "time is up: cannot find a better solution in given time(%d minutes)"
                    #     % self.runtime_in_minutes,
                    # )
                    # self.print_and_write_in_file(
                    #     file_to_write,
                    #     "it takes %0.3f second from multiple_ant_colony_system running"
                    #     % (time.time() - start_time_total),
                    # )
                    # self.print_and_write_in_file(
                    #     file_to_write, "the best path have found is:"
                    # )
                    # self.print_and_write_in_file(file_to_write, self.best_path)
                    # self.print_and_write_in_file(
                    #     file_to_write,
                    #     "best path distance is %f, best vehicle_num is %d"
                    #     % (self.best_path_distance.value, self.best_vehicle_num),
                    # )
                    # self.print_and_write_in_file(file_to_write, "*" * 50)

                    # Pass in None as the end flag
                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(PathMessage(None, None))

                    if file_to_write is not None:
                        file_to_write.flush()
                        file_to_write.close()
                    return

                if path_found_queue.empty():
                    continue

                path_info = path_found_queue.get()
                # print("[macs]: receive found path info")
                (
                    found_path,
                    found_path_distance,
                    found_path_used_vehicle_num,
                ) = path_info.get_path_info()
                while not path_found_queue.empty():
                    path, distance, vehicle_num = path_found_queue.get().get_path_info()

                    if distance < found_path_distance:
                        found_path, found_path_distance, found_path_used_vehicle_num = (
                            path,
                            distance,
                            vehicle_num,
                        )

                    if vehicle_num < found_path_used_vehicle_num:
                        found_path, found_path_distance, found_path_used_vehicle_num = (
                            path,
                            distance,
                            vehicle_num,
                        )

                # If the distance of the found path (which is feasible) is shorter, update the current best path information
                if found_path_distance < self.best_path_distance.value:
                    # Search for better results, update start_time
                    start_time_found_improved_solution = time.time()

                    # self.print_and_write_in_file(file_to_write, "*" * 50)
                    # self.print_and_write_in_file(
                    #     file_to_write,
                    #     "[macs]: distance of found path (%f) better than best path's (%f)"
                    #     % (found_path_distance, self.best_path_distance.value),
                    # )
                    # self.print_and_write_in_file(
                    #     file_to_write,
                    #     "it takes %0.3f second from multiple_ant_colony_system running"
                    #     % (time.time() - start_time_total),
                    # )
                    # self.print_and_write_in_file(file_to_write, "*" * 50)
                    if file_to_write is not None:
                        file_to_write.flush()

                    self.best_path[:] = found_path
                    # print("best1: ", self.best_path)
                    # print("found1: ", found_path)
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance.value = found_path_distance

                    # If graphics need to be drawn, the best path to be found is sent to the drawing program
                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(
                            PathMessage(self.best_path, self.best_path_distance.value)
                        )

                    # Notify acs_vehicle and acs_time threads of the currently found best_path and best_path_distance
                    global_path_to_acs_vehicle.put(
                        PathMessage(self.best_path, self.best_path_distance.value)
                    )
                    global_path_to_acs_time.put(
                        PathMessage(self.best_path, self.best_path_distance.value)
                    )

                # If the path found by these two threads uses fewer vehicles, stop these two threads and start the next iteration.
                # Send stop information to acs_time and acs_vehicle
                if found_path_used_vehicle_num < best_vehicle_num:
                    # Search for better results, update start_time
                    start_time_found_improved_solution = time.time()
                    # self.print_and_write_in_file(file_to_write, "*" * 50)
                    # self.print_and_write_in_file(
                    #     file_to_write,
                    #     "[macs]: vehicle num of found path (%d) better than best path's (%d), found path distance is %f"
                    #     % (
                    #         found_path_used_vehicle_num,
                    #         best_vehicle_num,
                    #         found_path_distance,
                    #     ),
                    # )
                    # self.print_and_write_in_file(
                    #     file_to_write,
                    #     "it takes %0.3f second multiple_ant_colony_system running"
                    #     % (time.time() - start_time_total),
                    # )
                    # self.print_and_write_in_file(file_to_write, "*" * 50)
                    if file_to_write is not None:
                        file_to_write.flush()

                    self.best_path[:] = found_path
                    # print("best2: ", self.best_path)
                    # print("found2: ", found_path)
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance.value = found_path_distance

                    if self.whether_or_not_to_show_figure:
                        path_queue_for_figure.put(
                            PathMessage(self.best_path, self.best_path_distance.value)
                        )

                    # Stop the acs_time and acs_vehicle threads
                    # print("[macs]: send stop info to acs_time and acs_vehicle")
                    # Notify acs_vehicle and acs_time threads of the currently found best_path and best_path_distance
                    stop_event.set()

    def get_best_route(self):
        if not self.best_path:
            return []
        routes = []
        route = []
        for node in self.best_path:
            if node != 0:
                route.append(node)
            else:
                if route:
                    routes.append(route)
                    route = []
        return routes

    @staticmethod
    def print_and_write_in_file(file_to_write=None, message="default message"):
        if file_to_write is None:
            print(message)
        else:
            print(message)
            file_to_write.write(str(message) + "\n")
