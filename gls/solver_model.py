from pydantic import BaseModel


class SolverSetting(BaseModel):
    time_limit: int
