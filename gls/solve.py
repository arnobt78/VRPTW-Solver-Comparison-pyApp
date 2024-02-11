from gls.base_solver import Solver
from gls.instance_loader import load_instance


time_precision_scaler = 10


def solve_with_gls(input_path, runtime):
    settings = {}
    settings["time_limit"] = runtime

    data = load_instance(input_path, time_precision_scaler)
    solver = Solver(data, time_precision_scaler)
    solver.create_model()
    solver.solve_model(settings)
    routes = solver.get_solution()
    print("GLS cost:", f"{solver.get_solution_travel_time():.1f}")
    print("GLS solution:")
    for i, route in enumerate(routes, start=1):
        print(f"Route #{i}: {' '.join(str(node) for node in route)}")
    print()

    return routes, round(solver.get_solution_travel_time(), 1)
