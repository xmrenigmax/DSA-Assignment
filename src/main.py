"""
main.py - Vehicle Routing Problem Solver — Entry Point

Loads a test case JSON file (or uses a built-in example), runs all three
solvers, validates results, prints a comparison table, and saves route plots.

Usage
-----
    python main.py                          # runs the built-in bakery example
    python main.py test_cases/tc_10.json    # loads a specific test case
    python main.py --all                    # runs all test cases in test_cases/
"""

import sys
import os
import glob

from classes.models import Customer, VRPInstance
from utils.utils import (load_test_case, print_solution, plot_solution, run_benchmark, print_benchmark_table)
from algorthms.clarke import run_naive_solution
from algorthms.branch import run_ai_solution
from algorthms.genetic import run_optimised_solution


# ---------------------------------------------------------------------------
# Built-in example: bakery scenario from the coursework brief
# ---------------------------------------------------------------------------

def _build_bakery_instance() -> VRPInstance:
    """Construct the bakery example from the coursework appendix."""
    demands = [0, 2, 3, 1, 4, 2, 3]   # index 0 = depot
    customers = [Customer(i, demands[i]) for i in range(7)]

    # Assign simple coordinates for plotting
    coords = [
        (5, 5),   # depot (bakery)
        (2, 7),   # café 1
        (1, 4),   # café 2
        (3, 2),   # café 3
        (6, 1),   # café 4
        (8, 3),   # café 5
        (9, 6),   # café 6
    ]
    for i, (x, y) in enumerate(coords):
        customers[i].x_coordinate = x
        customers[i].y_coordinate = y

    distance_matrix = [
        [0, 3, 5, 4, 6, 7, 8],
        [3, 0, 2, 6, 4, 5, 7],
        [5, 2, 0, 3, 5, 6, 4],
        [4, 6, 3, 0, 2, 5, 6],
        [6, 4, 5, 2, 0, 3, 4],
        [7, 5, 6, 5, 3, 0, 2],
        [8, 7, 4, 6, 4, 2, 0],
    ]

    return VRPInstance(
        customers=customers,
        distance_matrix=distance_matrix,
        vehicle_capacity=5,
        num_vehicles=3,
    )


# ---------------------------------------------------------------------------
# Single instance runner
# ---------------------------------------------------------------------------

def _solve_instance(instance: VRPInstance, label: str = "",
                    output_dir: str = "outputs") -> None:
    """Run all solvers on *instance*, print results, and save plots."""
    os.makedirs(output_dir, exist_ok=True)

    dm = instance.distance_matrix
    demands = instance.demands
    cap = instance.vehicle_capacity

    solvers = {
        "Clarke-Wright (Naive)": run_naive_solution,
        "Nearest Neighbour + 2-opt (AI)": run_ai_solution,
        "Genetic Algorithm (Optimised)": run_optimised_solution,
    }

    print(f"\n{'#' * 65}")
    print(f"  Instance: {label or 'Unknown'}")
    print(f"  Customers: {instance.num_customers}  |  "
          f"Capacity: {instance.vehicle_capacity}  |  "
          f"Vehicles: {instance.num_vehicles}")
    print(f"{'#' * 65}")

    solutions = {}
    for solver_label, solver_fn in solvers.items():
        sol = solver_fn(dm, demands, cap)
        solutions[solver_label] = sol
        print_solution(sol, label=solver_label)

    # Benchmarking (5 runs each for stable timing)
    results = run_benchmark(solvers, instance, runs=5)
    print_benchmark_table(results)

    # Save plots
    for solver_label, sol in solutions.items():
        safe_name = (
            (label or "instance").replace(" ", "_")
            + "__"
            + solver_label.replace(" ", "_").replace("/", "-")
            + ".png"
        )
        plot_solution(
            sol, instance,
            title=f"{label} — {solver_label}",
            output_path=os.path.join(output_dir, safe_name),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if "--all" in args:
        # Run every JSON file in test_cases/
        test_files = sorted(glob.glob("test_cases/*.json"))
        if not test_files:
            print("No JSON files found in test_cases/")
            return
        for filepath in test_files:
            instance = load_test_case(filepath)
            if instance is not None:
                _solve_instance(instance, label=os.path.basename(filepath))
        return

    if args and not args[0].startswith("--"):
        # Single file provided on command line
        filepath = args[0]
        instance = load_test_case(filepath)
        if instance is None:
            print(f"Failed to load '{filepath}'. Exiting.")
            sys.exit(1)
        _solve_instance(instance, label=os.path.basename(filepath))
        return

    # Default: built-in bakery example
    print("\nNo file specified — running built-in bakery example.\n")
    instance = _build_bakery_instance()
    _solve_instance(instance, label="Bakery Example")


if __name__ == "__main__":
    main()
