"""
clarke.py - Clarke-Wright Savings Algorithm for the Capacitated VRP

Algorithm overview (Clarke & Wright, 1964)
------------------------------------------
1. Start with a "star" solution: one route per customer (depot → i → depot).
2. For every pair (i, j) compute the *savings* of merging their routes:
       s(i, j) = d(0, i) + d(0, j) − d(i, j)
   Positive savings mean combining the two trips into one is cheaper.
3. Sort savings in descending order.
4. Greedily merge route pairs when:
   (a) i and j belong to different routes,
   (b) i is the last customer on its route and j is the first on its route
       (or vice-versa, to keep routes linear), and
   (c) the combined demand does not exceed vehicle capacity.

Time complexity  : O(n² log n)  – dominated by the savings sort.
Space complexity : O(n²)        – distance matrix storage.

Assumptions & limitations
--------------------------
* Homogeneous fleet (all vehicles have the same capacity).
* Symmetric distance matrix is NOT assumed; savings are computed directionally.
* The algorithm is a greedy constructive heuristic; it does not guarantee the
  global optimum.
* No time-window constraints are modelled.
"""

from classes.models import Route, Customer


def run_naive_solution(distance_matrix: list, demands: list,
                       vehicle_capacity: int) -> dict:
    """
    Solve the CVRP using the Clarke-Wright Savings algorithm.

    Parameters
    ----------
    distance_matrix  : list[list[float]]  n×n cost matrix (index 0 = depot).
    demands          : list[int]          demand[i] for location i.
    vehicle_capacity : int                Maximum load per vehicle.

    Returns
    -------
    dict  {"routes": list[list[int]], "total_distance": float}
    """
    number_of_nodes = len(distance_matrix)
    number_of_customers = number_of_nodes - 1  # exclude depot (index 0)

    # ------------------------------------------------------------------ #
    # Step 1: Compute savings for all customer pairs (i, j), i < j
    # ------------------------------------------------------------------ #
    savings = []
    for i in range(1, number_of_nodes):
        for j in range(i + 1, number_of_nodes):
            saving_value = (distance_matrix[0][i]
                            + distance_matrix[0][j]
                            - distance_matrix[i][j])
            savings.append((saving_value, i, j))

    # Sort descending so the most beneficial merges are attempted first
    savings.sort(key=lambda entry: entry[0], reverse=True)

    # ------------------------------------------------------------------ #
    # Step 2: Initialise one route per customer
    # ------------------------------------------------------------------ #
    # Each route is stored as a plain list (no depot wrapper yet).
    # route_of[i] maps customer index → its current route list object.
    routes = [[i] for i in range(1, number_of_nodes)]
    route_of = {i: routes[i - 1] for i in range(1, number_of_nodes)}

    # Track per-route demand using a dictionary keyed by route id
    route_demand = {id(r): demands[r[0]] for r in routes}

    # ------------------------------------------------------------------ #
    # Step 3: Greedily merge routes
    # ------------------------------------------------------------------ #
    for saving_value, customer_i, customer_j in savings:
        route_i = route_of.get(customer_i)
        route_j = route_of.get(customer_j)

        # Skip if either customer has already been merged into a larger route
        # in a way that prevents the required end/start position.
        if route_i is None or route_j is None or route_i is route_j:
            continue

        # Only merge if i is at the END of its route and j is at the START
        # of its route (preserving a single contiguous path).
        i_at_end = (route_i[-1] == customer_i)
        j_at_start = (route_j[0] == customer_j)

        if not (i_at_end and j_at_start):
            # Try the reverse orientation: j at end, i at start
            j_at_end = (route_j[-1] == customer_j)
            i_at_start = (route_i[0] == customer_i)
            if j_at_end and i_at_start:
                # Swap so that route_i always appends onto route_j
                route_i, route_j = route_j, route_i
                customer_i, customer_j = customer_j, customer_i
            else:
                continue   # Neither orientation allows a clean merge

        # Capacity check
        combined_demand = route_demand[id(route_i)] + route_demand[id(route_j)]
        if combined_demand > vehicle_capacity:
            continue

        # Merge: extend route_i with route_j
        merged_demand = combined_demand
        route_i.extend(route_j)

        # Update route_of references for all customers now in route_j
        for customer in route_j:
            route_of[customer] = route_i

        # Remove the now-absorbed route_j from the active set
        routes.remove(route_j)
        route_demand[id(route_i)] = merged_demand

    # ------------------------------------------------------------------ #
    # Step 4: Wrap routes with depot and compute total distance
    # ------------------------------------------------------------------ #
    final_routes = [[0] + route + [0] for route in routes]
    total_distance = 0.0
    for route in final_routes:
        for index in range(len(route) - 1):
            total_distance += distance_matrix[route[index]][route[index + 1]]

    return {"routes": final_routes, "total_distance": total_distance}
