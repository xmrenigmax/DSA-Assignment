"""
clarke_operators.py - Modular operators for the Clarke-Wright Savings algorithm

Contains the mathematical operations, routing merges, and calculations
decoupled from the main procedural loop.
"""

def compute_savings(distance_matrix: list) -> list:
    """
    Time: O(n^2), Space: O(n^2) - Computes and sorts savings.
    Formula: s(i, j) = d(0, i) + d(0, j) - d(i, j)
    """
    number_of_nodes = len(distance_matrix)
    savings = []

    for i in range(1, number_of_nodes):
        for j in range(i + 1, number_of_nodes):
            saving_value = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((saving_value, i, j))

    savings.sort(key=lambda entry: entry[0], reverse=True)
    return savings


def initialize_routes(number_of_nodes: int, demands: list) -> tuple:
    """
    Time: O(n), Space: O(n) - Initialises individual routes for all customers.
    """
    routes = [[i] for i in range(1, number_of_nodes)]
    route_of = {i: routes[i - 1] for i in range(1, number_of_nodes)}
    route_demand = {id(r): demands[r[0]] for r in routes}

    return routes, route_of, route_demand


def merge_routes(
    customer_i: int,
    customer_j: int,
    routes: list,
    route_of: dict,
    route_demand: dict,
    vehicle_capacity: int,
) -> None:
    """
    Time: O(1) amortized, Space: O(1) - Evaluates constraints and executes a valid route merge.
    """
    route_i = route_of.get(customer_i)
    route_j = route_of.get(customer_j)

    if route_i is None or route_j is None or route_i is route_j:
        return

    i_at_end = route_i[-1] == customer_i
    j_at_start = route_j[0] == customer_j

    if not (i_at_end and j_at_start):
        j_at_end = route_j[-1] == customer_j
        i_at_start = route_i[0] == customer_i
        if j_at_end and i_at_start:
            route_i, route_j = route_j, route_i
            customer_i, customer_j = customer_j, customer_i
        else:
            return

    combined_demand = route_demand[id(route_i)] + route_demand[id(route_j)]
    if combined_demand > vehicle_capacity:
        return

    route_i.extend(route_j)

    for customer in route_j:
        route_of[customer] = route_i

    routes.remove(route_j)
    route_demand[id(route_i)] = combined_demand


def calculate_total_distance(routes: list, distance_matrix: list) -> float:
    """
    Time: O(n), Space: O(1) - Iterates through formatted routes to sum edge weights.
    """
    total_distance = 0.0

    for route in routes:
        for index in range(len(route) - 1):
            total_distance += distance_matrix[route[index]][route[index + 1]]

    return total_distance