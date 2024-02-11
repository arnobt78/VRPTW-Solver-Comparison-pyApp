from pyvrp import Model, read
from pyvrp.stop import MaxIterations, MaxRuntime

ITERATIONS = 10


def solve_with_hgs(input_path, runtime):
    INSTANCE = read(input_path, instance_format="solomon", round_func="trunc1")
    model = Model.from_data(INSTANCE)
    result = model.solve(stop=MaxRuntime(runtime), seed=0)
    # result = model.solve(stop=MaxIterations(ITERATIONS), seed=0)

    print("HGS cost:", result.cost() / 10)
    print("HGS solution:")
    print(result.best)
    # create list of routes for result.best
    routes = []
    for route in result.best.get_routes():
        routes.append(route.visits())

    return routes, round(result.cost() / 10, 1)
