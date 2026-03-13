"""
clarke.py - Clarke-Wright Savings Algorithm for the Capacitated VRP

Algorithm overview (Clarke & Wright, 1964)
------------------------------------------
1. Start with a "star" solution: one route per customer (depot -> i -> depot).
2. For every pair (i, j) compute the *savings* of merging their routes.
3. Sort savings in descending order.
4. Greedily merge route pairs ensuring capacity constraints.

Time complexity  : O(n^2 log n)  - dominated by the savings sort
Space complexity : O(n^2)        - distance matrix storage
"""

import logging

from .clarke_operators import (
    compute_savings,
    initialize_routes,
    merge_routes,
    calculate_total_distance,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_naive_solution(distance_matrix: list, demands: list, vehicle_capacity: int) -> dict:
    """
    Solve the CVRP using the Clarke-Wright Savings algorithm.
    """
    if not distance_matrix or vehicle_capacity <= 0:
        logger.error("Invalid input: Distance matrix empty or capacity <= 0.")
        return {"routes": [], "total_distance": 0.0}

    number_of_nodes = len(distance_matrix)
    if number_of_nodes <= 1:
        return {"routes": [[0, 0]], "total_distance": 0.0}

    # Step 1: Compute savings for all customer pairs (i, j)
    savings = compute_savings(distance_matrix)

    # Step 2: Initialise one route per customer
    routes, route_of, route_demand = initialize_routes(number_of_nodes, demands)

    # Step 3: Greedily merge routes
    for saving_value, customer_i, customer_j in savings:
        merge_routes(customer_i, customer_j, routes, route_of, route_demand, vehicle_capacity)

    # Step 4: Wrap routes with depot and compute total distance
    final_routes = [[0] + route + [0] for route in routes]
    total_distance = calculate_total_distance(final_routes, distance_matrix)

    logger.debug(f"Clarke-Wright Completed. Best distance: {total_distance:.4f}")

    return {"routes": final_routes, "total_distance": total_distance}