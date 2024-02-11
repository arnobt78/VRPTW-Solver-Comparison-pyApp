from vrplib import read_solution

def bks_solution(bks_path):
    BKS = read_solution(bks_path)
    BKS_ROUTES = BKS["routes"]
    BKS_COST = BKS["cost"]
    print("BKS cost:", BKS_COST)
    print("BKS solution:")
    for i, route in enumerate(BKS_ROUTES, start=1):
        print(f"Route #{i}: {' '.join(str(node) for node in route)}")
    print()

    return BKS_ROUTES, BKS_COST
