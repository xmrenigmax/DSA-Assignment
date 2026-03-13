"""
utils.py - Utility functions for the Vehicle Routing Problem (VRP)

Provides:
  - load_test_case        : deserialise a JSON problem file into a VRPInstance
  - calculate_route_distance : sum edge weights along a node sequence
  - validate_solution     : verify feasibility of a returned solution dict
  - print_solution        : human-readable console output
  - plot_solution         : matplotlib visualisation (saved to file)
  - run_benchmark         : time multiple solvers on one instance and tabulate results
"""

import json
import time
import math

# matplotlib is imported lazily so the module still loads in headless environments
_plt_available = True
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; works without a display
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    _plt_available = False


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_test_case(filepath: str):
    """
    Load a VRP test case from a JSON file and return a VRPInstance.

    Expected JSON schema
    --------------------
    {
        "vehicle_capacity": int,
        "num_vehicles": int,
        "distance_matrix": [[float, ...]],  // n×n, index 0 = depot
        "demands": [int, ...]               // length n, index 0 = 0
    }

    Returns
    -------
    VRPInstance or None on error.
    """
    # Import here to avoid circular dependency (models imports nothing from utils)
    from classes.models import Customer, VRPInstance

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
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[load_test_case] Schema error in '{filepath}': {exc}")
        return None
    except OSError as exc:
        print(f"[load_test_case] Cannot open '{filepath}': {exc}")
        return None


# ---------------------------------------------------------------------------
# Distance / solution evaluation
# ---------------------------------------------------------------------------

def calculate_route_distance(route_nodes: list, distance_matrix: list) -> float:
    """
    Return the total travel distance for an ordered list of node indices.

    Parameters
    ----------
    route_nodes     : list[int]          Sequence including depot at both ends.
    distance_matrix : list[list[float]]  n×n cost matrix.

    Time complexity: O(|route|).
    """
    if not route_nodes:
        return 0.0
    total = 0.0
    for i in range(len(route_nodes) - 1):
        total += distance_matrix[route_nodes[i]][route_nodes[i + 1]]
    return total


def validate_solution(solution: dict, demands: list,
                      vehicle_capacity: int, num_customers: int) -> dict:
    """
    Check a solution dictionary for feasibility.

    Parameters
    ----------
    solution        : {"routes": [[int,...], ...], "total_distance": float}
    demands         : list of demand values indexed by customer id
    vehicle_capacity: int
    num_customers   : int  (excluding depot)

    Returns
    -------
    {"valid": bool, "errors": [str]}
    """
    errors = []
    routes = solution.get("routes", [])
    visited = set()

    for idx, route in enumerate(routes):
        # Must start and end at depot
        if route[0] != 0 or route[-1] != 0:
            errors.append(f"Route {idx}: must start and end at depot (0).")

        route_demand = 0
        for node in route[1:-1]:            # exclude depot occurrences
            if node == 0:
                errors.append(f"Route {idx}: depot appears mid-route.")
                continue
            if node in visited:
                errors.append(f"Route {idx}: customer {node} visited more than once.")
            visited.add(node)
            route_demand += demands[node]

        if route_demand > vehicle_capacity:
            errors.append(
                f"Route {idx}: demand {route_demand} exceeds capacity {vehicle_capacity}."
            )

    all_customers = set(range(1, num_customers + 1))
    missing = all_customers - visited
    if missing:
        errors.append(f"Unvisited customers: {sorted(missing)}.")

    return {"valid": len(errors) == 0, "errors": errors}


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_solution(solution: dict, label: str = "Solution") -> None:
    """Print a formatted summary of a VRP solution to stdout."""
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


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_solution(solution: dict, instance, title: str = "VRP Solution",
                  output_path: str = "solution.png") -> None:
    """
    Render a VRP solution as a route diagram and save to *output_path*.

    Each route is drawn in a distinct colour.  The depot is marked with a
    star; customers with filled circles labelled by their index.

    Parameters
    ----------
    solution    : solution dict returned by any solver
    instance    : VRPInstance
    title       : figure title
    output_path : file path for the saved PNG
    """
    if not _plt_available:
        print("[plot_solution] matplotlib not available – skipping plot.")
        return

    customers = instance.customers
    routes = solution["routes"]

    # Colour palette (cycles if more routes than colours)
    colours = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Draw routes
    for r_idx, route in enumerate(routes):
        colour = colours[r_idx % len(colours)]
        xs = [customers[n].x_coordinate for n in route]
        ys = [customers[n].y_coordinate for n in route]
        ax.plot(xs, ys, "-o", color=colour, linewidth=1.5,
                markersize=5, zorder=2)

    # Draw depot
    depot = customers[0]
    ax.plot(depot.x_coordinate, depot.y_coordinate, "*",
            color="black", markersize=14, zorder=4, label="Depot")

    # Draw customer nodes
    for c in customers[1:]:
        ax.plot(c.x_coordinate, c.y_coordinate, "o",
                color="steelblue", markersize=8, zorder=3)
        ax.annotate(str(c.customer_id),
                    (c.x_coordinate, c.y_coordinate),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8, zorder=5)

    # Legend patches for routes
    patches = [
        mpatches.Patch(color=colours[i % len(colours)],
                       label=f"Route {i + 1}")
        for i in range(len(routes))
    ]
    patches.insert(0, mpatches.Patch(color="black", label="Depot"))
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    ax.set_title(f"{title}\nTotal distance: {solution['total_distance']:.4f}",
                 fontsize=12)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot_solution] Saved → {output_path}")


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def run_benchmark(solvers: dict, instance, runs: int = 1) -> list:
    """
    Time each solver on *instance* and return a list of result records.

    Parameters
    ----------
    solvers  : {"label": callable}  Each callable must accept
               (distance_matrix, demands, vehicle_capacity) and return a
               solution dict.
    instance : VRPInstance
    runs     : int  Number of repeated runs (result = best distance seen).

    Returns
    -------
    list of dicts: [{"solver": str, "distance": float, "time_ms": float,
                     "routes": int, "valid": bool}]
    """
    results = []
    dm = instance.distance_matrix
    demands = instance.demands
    cap = instance.vehicle_capacity

    for label, solver_fn in solvers.items():
        best_distance = math.inf
        best_solution = None
        total_time = 0.0

        for _ in range(runs):
            t0 = time.perf_counter()
            sol = solver_fn(dm, demands, cap)
            t1 = time.perf_counter()
            total_time += (t1 - t0) * 1000   # ms
            if sol["total_distance"] < best_distance:
                best_distance = sol["total_distance"]
                best_solution = sol

        validity = validate_solution(
            best_solution, demands, cap, instance.num_customers
        )
        results.append({
            "solver": label,
            "distance": round(best_distance, 4),
            "time_ms": round(total_time / runs, 3),
            "routes": len(best_solution["routes"]),
            "valid": validity["valid"],
        })

    return results


def print_benchmark_table(results: list) -> None:
    """Print benchmark results as a formatted table."""
    header = f"{'Solver':<22} {'Distance':>12} {'Time (ms)':>12} {'Routes':>8} {'Valid':>7}"
    print("\n" + "=" * 65)
    print("  Benchmark Results")
    print("=" * 65)
    print(f"  {header}")
    print(f"  {'─' * 61}")
    for r in results:
        valid_str = "✓" if r["valid"] else "✗"
        print(f"  {r['solver']:<22} {r['distance']:>12.4f} "
              f"{r['time_ms']:>12.3f} {r['routes']:>8} {valid_str:>7}")
    print("=" * 65 + "\n")
