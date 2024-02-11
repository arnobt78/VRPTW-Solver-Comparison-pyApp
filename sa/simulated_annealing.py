import random
from copy import deepcopy
from math import exp
import time

counter = 0


def objective_function(num_vhcls, total_distance):
    global counter
    counter += 1
    return num_vhcls * total_distance


def sa_algorithm(
    instance,
    temp_start=350,
    update_temp=lambda t: 0.9999 * t,
    stop_criterion=lambda t: t <= 0.01,
):
    curr_solution = incumb_solution = deepcopy(instance)
    # print("Inside sa: ", instance.get_total_distance_and_vehicles())
    curr_dist, curr_vhcls = (
        incumb_dist,
        incumb_vhcls,
    ) = curr_solution.get_total_distance_and_vehicles()

    temp = temp_start

    start = time.time()
    afterOneMin, cOneMin = None, 0
    afterFiveMin, cFiveMin = None, 0
    total = None
    incumb_vhcls_o = objective_function(incumb_vhcls, incumb_dist)

    while not stop_criterion(temp):
        # print(temp, incumb_vhcls, incumb_dist)
        neighbour = deepcopy(curr_solution)
        neighbour.generate_random_neighbour()

        neighbour_dist, neighbour_vhcls = neighbour.get_total_distance_and_vehicles()

        curr_sol_o = objective_function(curr_vhcls, curr_dist)
        neigh_sol_o = objective_function(neighbour_vhcls, neighbour_dist)

        if neigh_sol_o < curr_sol_o or random.random() < exp(
            -(abs(curr_sol_o - neigh_sol_o)) / temp
        ):
            curr_solution = neighbour
            curr_dist, curr_vhcls = neighbour_dist, neighbour_vhcls

            if objective_function(curr_vhcls, curr_dist) < incumb_vhcls_o:
                incumb_solution = curr_solution
                incumb_dist, incumb_vhcls = curr_dist, curr_vhcls
                incumb_vhcls_o = objective_function(incumb_vhcls, incumb_dist)

        temp = update_temp(temp)
        if not afterOneMin and time.time() - start >= 60:
            afterOneMin, cOneMin = deepcopy(incumb_solution), counter
        if not afterFiveMin and time.time() - start >= 5 * 60:
            afterFiveMin, cFiveMin = deepcopy(incumb_solution), counter

    if not afterOneMin:
        afterOneMin = incumb_solution
    if not afterFiveMin:
        afterFiveMin = incumb_solution

    # print("Algo Time:", time.time() - start)
    return [
        (afterOneMin, cOneMin),
        (afterFiveMin, cFiveMin),
        (incumb_solution, counter),
    ]
