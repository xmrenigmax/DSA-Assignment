import json

from classes.models import Customer, VRPInstance


def load_test_case(filepath: str):
    try:
        with open(filepath, "r") as fh:
            data = json.load(fh)

        demands = data["demands"]
        customers = [
            Customer(customer_id=i, demand=demands[i])
            for i in range(len(demands))
        ]

        return VRPInstance(
            customers=customers,
            distance_matrix=data["distance_matrix"],
            vehicle_capacity=data["vehicle_capacity"],
            num_vehicles=data["num_vehicles"],
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as error:
        print(f"[load_test_case] Failed to load '{filepath}': {error}")
        return None


def validate_solution(
    solution: dict, demands: list, vehicle_capacity: int, num_customers: int
) -> dict:
    errors = []
    routes = solution.get("routes", [])
    visited = set()

    for idx, route in enumerate(routes):
        if route[0] != 0 or route[-1] != 0:
            errors.append(f"Route {idx}: must start and end at depot (0).")

        route_demand = 0
        for node in route[1:-1]:
            if node == 0:
                errors.append(f"Route {idx}: depot appears mid-route.")
                continue
            if node in visited:
                errors.append(f"Route {idx}: customer {node} visited more than once.")
            visited.add(node)
            route_demand += demands[node]

        if route_demand > vehicle_capacity:
            errors.append(f"Route {idx}: demand {route_demand} exceeds capacity.")

    missing = set(range(1, num_customers + 1)) - visited
    if missing:
        errors.append(f"Unvisited customers: {sorted(missing)}.")

    return {"valid": len(errors) == 0, "errors": errors}


def print_solution(solution: dict, label: str = "Solution") -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    routes = solution.get("routes", [])
    for i, route in enumerate(routes):
        print(f"  Route {i + 1:>2}: {' -> '.join(str(n) for n in route)}")
    print(f"  {'─' * 40}")
    print(f"  Total distance : {solution.get('total_distance', 0):.4f}")
    print(f"  Number of routes: {len(routes)}")
    print(f"{'=' * 60}\n")