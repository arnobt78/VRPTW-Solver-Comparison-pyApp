from pydantic import BaseModel


class ProblemInstance(BaseModel):
    time_matrix: list
    time_windows: list
    demands: list
    depot: int
    num_vehicles: int
    vehicle_capacities: list
    service_times: list
