"""
branch.py - Nearest Neighbour Heuristic with 2-opt Local Search for the CVRP

This module was generated with the assistance of a generative AI tool (Claude,
Anthropic) using the following prompt:

    "Implement a nearest-neighbour constructive heuristic for the Capacitated
     Vehicle Routing Problem in Python.  After construction, apply a 2-opt
     intra-route improvement step to reduce the total travel distance.
     The function signature must be:
         run_ai_solution(distance_matrix, demands, vehicle_capacity) -> dict
     returning {'routes': list[list[int]], 'total_distance': float}."

The output was reviewed, corrected for edge-cases, and integrated into the
project by the student.

Algorithm overview
------------------
Phase 1 – Nearest Neighbour Construction
  Starting from the depot, repeatedly visit the closest unvisited customer
  that still fits in the current vehicle's remaining capacity.  When no
  feasible customer exists, return to the depot and start a new route.

  Time complexity: O(n²) per route; O(n² · k) overall where k = routes built.

Phase 2 – 2-opt Intra-route Improvement
  For each route, try all pairs (i, j) of non-adjacent edges.  Reversing the
  sub-sequence between i and j is accepted if it reduces the route distance.
  Repeat until no improving swap exists.

  Time complexity: O(n² · iterations) per route.

Assumptions & limitations
--------------------------
* Homogeneous fleet; unlimited number of vehicles (routes) allowed.
* The nearest-neighbour phase is sensitive to the starting order; a single
  pass from the depot is used here.
* 2-opt only improves individual routes (intra-route); inter-route swaps
  (e.g. Or-opt) are not implemented.
* No time-window constraints.
"""


def run_ai_solution(distance_matrix: list, demands: list,
                    vehicle_capacity: int) -> dict:
    """
    Solve the CVRP with a Nearest Neighbour heuristic followed by 2-opt.

    Parameters
    ----------
    distance_matrix  : list[list[float]]  n×n cost matrix (index 0 = depot).
    demands          : list[int]          demand[i] for location i.
    vehicle_capacity : int                Maximum load per vehicle.

    Returns
    -------
    dict  {"routes": list[list[int]], "total_distance": float}
    """
    # ------------------------------------------------------------------ #
    # Phase 1: Nearest Neighbour Construction
    # ------------------------------------------------------------------ #
    number_of_locations = len(distance_matrix)
    unvisited_customers = set(range(1, number_of_locations))
    delivery_routes = []

    while unvisited_customers:
        current_route = [0]
        current_capacity = vehicle_capacity
        current_location = 0

        while unvisited_customers:
            # Find the nearest feasible unvisited customer
            best_next_customer = None
            minimum_distance = float("inf")

            for customer in unvisited_customers:
                if (demands[customer] <= current_capacity
                        and distance_matrix[current_location][customer] < minimum_distance):
                    best_next_customer = customer
                    minimum_distance = distance_matrix[current_location][customer]

            if best_next_customer is None:
                break   # No feasible customer; close this route

            current_route.append(best_next_customer)
            unvisited_customers.remove(best_next_customer)
            current_capacity -= demands[best_next_customer]
            current_location = best_next_customer

        current_route.append(0)  # Return to depot
        delivery_routes.append(current_route)

    # ------------------------------------------------------------------ #
    # Phase 2: 2-opt Intra-route Improvement
    # ------------------------------------------------------------------ #
    improved_routes = [_two_opt(route, distance_matrix) for route in delivery_routes]

    # ------------------------------------------------------------------ #
    # Compute total distance
    # ------------------------------------------------------------------ #
    total_travel_distance = 0.0
    for route in improved_routes:
        for index in range(len(route) - 1):
            total_travel_distance += distance_matrix[route[index]][route[index + 1]]

    return {"routes": improved_routes, "total_distance": total_travel_distance}


# ---------------------------------------------------------------------------
# Helper: 2-opt local search
# ---------------------------------------------------------------------------

def _two_opt(route: list, distance_matrix: list) -> list:
    """
    Apply 2-opt improvement to a single route.

    Iteratively reverses sub-sequences of the route (excluding the depot
    endpoints) until no swap reduces the total route distance.

    Parameters
    ----------
    route           : list[int]  Route including depot at both ends.
    distance_matrix : list[list[float]]

    Returns
    -------
    list[int]  Improved route (same structure, depot at both ends).
    """
    # Routes with 3 or fewer nodes (depot → 1 customer → depot) cannot be improved
    if len(route) <= 3:
        return route

    best = route[:]
    improved = True

    while improved:
        improved = False
        # Indices 1 … len-2 are non-depot nodes
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                # Cost of existing edges: (i-1 → i) and (j → j+1)
                current_cost = (distance_matrix[best[i - 1]][best[i]]
                                + distance_matrix[best[j]][best[j + 1]])
                # Cost after reversal: (i-1 → j) and (i → j+1)
                new_cost = (distance_matrix[best[i - 1]][best[j]]
                            + distance_matrix[best[i]][best[j + 1]])
                if new_cost < current_cost - 1e-9:   # numerical tolerance
                    best[i: j + 1] = best[i: j + 1][::-1]
                    improved = True

    return best
