from sa.instance_loader import load_from_file
from sa.simulated_annealing import sa_algorithm

INIT_TEMP = 700
UPDATE_TEMP = lambda t: 0.9999 * t
STOP_CRITERIA = lambda t: t <= 0.01


def solve_using_sa(input_path):
    instance = load_from_file(input_path)
    instance.find_initial_solution()

    results = sa_algorithm(instance, INIT_TEMP, UPDATE_TEMP, STOP_CRITERIA)

    routes = results[2][0].get_solution()
    cost = results[2][0].get_total_distance()

    print(f"SA cost: {cost}")
    print("SA solution:")
    for i, route in enumerate(routes, start=1):
        print(f"Route #{i}: {' '.join(str(node) for node in route)}")

    return routes, round(cost, 1)
