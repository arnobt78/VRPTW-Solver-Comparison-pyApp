import time
from matplotlib import pyplot as plt
from tabulate import tabulate
from aco.solve import solve_with_aco
from bks import bks_solution
from hgs.solve import solve_with_hgs
from gls.solve import solve_with_gls
from plot import plot_my_solution
from sa.solve import solve_using_sa
from pyvrp import read

dataset = "r211"
INPUT_PATH = f"data/{dataset}.txt"
BKS_PATH = f"data/{dataset}.sol"
RUNTIME = 120  # seconds

INSTANCE = read(INPUT_PATH, instance_format="solomon", round_func="trunc1")

result = {
    "bks": {},
    "hgs": {},
    "gls": {},
    "aco": {},
    "sa": {},
}
print("Running Algorithms on dataset:", dataset)
result["bks"]["routes"], result["bks"]["cost"] = bks_solution(BKS_PATH)

start = time.time()
result["hgs"]["routes"], result["hgs"]["cost"] = solve_with_hgs(INPUT_PATH, RUNTIME)
result["hgs"]["runtime"] = time.time() - start

start = time.time()
result["gls"]["routes"], result["gls"]["cost"] = solve_with_gls(INPUT_PATH, RUNTIME)
result["gls"]["runtime"] = time.time() - start

start = time.time()
result["aco"]["routes"], result["aco"]["cost"] = solve_with_aco(INPUT_PATH)
result["aco"]["runtime"] = time.time() - start

start = time.time()
result["sa"]["routes"], result["sa"]["cost"] = solve_using_sa(INPUT_PATH)
result["sa"]["runtime"] = time.time() - start

_, ax = plt.subplots(figsize=(10, 10))
plot_my_solution(result["hgs"], INSTANCE, ax=ax, dataset=dataset, algo="HGS")

_, ax = plt.subplots(figsize=(10, 10))
plot_my_solution(result["gls"], INSTANCE, ax=ax, dataset=dataset, algo="GLS")

_, ax = plt.subplots(figsize=(10, 10))
plot_my_solution(result["aco"], INSTANCE, ax=ax, dataset=dataset, algo="ACO")

_, ax = plt.subplots(figsize=(10, 10))
plot_my_solution(result["sa"], INSTANCE, ax=ax, dataset=dataset, algo="SA")

gap = lambda bks_cost, algo_cost: round(100 * (algo_cost - bks_cost) / bks_cost, 2)
header = ["Algorithms", "No. of Routes", "Costs", "Gap(%)", "Runtime(seconds)"]
rows = [
    ["BKS", len(result["bks"]["routes"]), result["bks"]["cost"], "-", "-"],
    [
        "HGS",
        len(result["hgs"]["routes"]),
        result["hgs"]["cost"],
        gap(result["bks"]["cost"], result["hgs"]["cost"]),
        result["hgs"]["runtime"],
    ],
    [
        "GLS",
        len(result["gls"]["routes"]),
        result["gls"]["cost"],
        gap(result["bks"]["cost"], result["gls"]["cost"]),
        result["gls"]["runtime"],
    ],
    [
        "ACO",
        len(result["aco"]["routes"]),
        result["aco"]["cost"],
        gap(result["bks"]["cost"], result["aco"]["cost"]),
        result["aco"]["runtime"],
    ],
    [
        "SA",
        len(result["sa"]["routes"]),
        result["sa"]["cost"],
        gap(result["bks"]["cost"], result["sa"]["cost"]),
        result["sa"]["runtime"],
    ],
]
print("Algorithm results on dataset:", dataset)
tabulate(rows, header, tablefmt="html")