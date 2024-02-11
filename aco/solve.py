from aco.vrptw_base import VrptwGraph
from aco.multiple_ant_colony_system import MultipleAntColonySystem


ants_num = 30
q0 = 0.9
beta = 0.9
rho = 0.1
show_figure = False
runtime_in_minutes = 5


def solve_with_aco(input_path):
    graph = VrptwGraph(input_path, rho)
    macs = MultipleAntColonySystem(
        graph,
        ants_num=ants_num,
        beta=beta,
        q0=q0,
        whether_or_not_to_show_figure=show_figure,
        runtime_in_minutes=runtime_in_minutes,
    )
    macs.run_multiple_ant_colony_system()
    print("ACO cost:", macs.best_path_distance.value)
    routes = get_best_route_from_path(macs.best_path)
    print("ACO solution:")
    for i, route in enumerate(routes, start=1):
        print(f"Route #{i}: {' '.join(str(node) for node in route)}")
    print()

    return routes, round(macs.best_path_distance.value, 1)


def get_best_route_from_path(best_path):
    if not best_path:
        return []
    routes = []
    route = []
    for node in best_path:
        if node != 0:
            route.append(node)
        else:
            if route:
                routes.append(route)
                route = []
    return routes
