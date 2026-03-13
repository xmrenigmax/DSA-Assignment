import sys;
import os;
import glob;

from classes.models import Customer;
from classes.models import VRPInstance;

from utils.core_utils import load_test_case;
from utils.core_utils import print_solution;
from utils.benchmarking import run_benchmark;
from utils.benchmarking import print_benchmark_table;
from utils.visualisation import plot_solution;

from algorthms.clarke.clarke import run_naive_solution;
from algorthms.branch.branch import run_ai_solution;
from algorthms.genetic.genetic import run_optimised_solution;

def _build_bakery_instance() -> VRPInstance:
  demands = [0, 2, 3, 1, 4, 2, 3];
  customers = [Customer(i, demands[i]) for i in range(7)];
  coords = [(5, 5), (2, 7), (1, 4), (3, 2), (6, 1), (8, 3), (9, 6)];
  
  for i, (x, y) in enumerate(coords):
    customers[i].x_coordinate = x;
    customers[i].y_coordinate = y;

  distance_matrix = [
    [0, 3, 5, 4, 6, 7, 8],
    [3, 0, 2, 6, 4, 5, 7],
    [5, 2, 0, 3, 5, 6, 4],
    [4, 6, 3, 0, 2, 5, 6],
    [6, 4, 5, 2, 0, 3, 4],
    [7, 5, 6, 5, 3, 0, 2],
    [8, 7, 4, 6, 4, 2, 0]
  ];

  return VRPInstance(customers=customers, distance_matrix=distance_matrix, vehicle_capacity=5, num_vehicles=3);

def _solve_instance(instance: VRPInstance, label: str = "", output_dir: str = "outputs") -> None:
  os.makedirs(output_dir, exist_ok=True);

  solvers = {
    "Clarke-Wright (Naive)": run_naive_solution,
    "Nearest Neighbour + 2-opt (AI)": run_ai_solution,
    "Genetic Algorithm (Optimised)": run_optimised_solution
  };

  print(f"\n{'#' * 65}");
  print(f"  Instance: { label or 'Unknown' }");
  print(f"  Customers: { instance.num_customers }  |  Capacity: { instance.vehicle_capacity }");
  print(f"{'#' * 65}");

  solutions = {};
  for solver_label, solver_fn in solvers.items():
    sol = solver_fn(instance.distance_matrix, instance.demands, instance.vehicle_capacity);
    solutions[solver_label] = sol;
    print_solution(sol, label=solver_label);

  results = run_benchmark(solvers, instance, runs=5);
  print_benchmark_table(results);

  for solver_label, sol in solutions.items():
    safe_name = f"{ (label or 'instance').replace(' ', '_') }__{ solver_label.replace(' ', '_').replace('/', '-') }.png";
    plot_solution(sol, instance, title=f"{ label } — { solver_label }", output_path=os.path.join(output_dir, safe_name));

def main():
  args = sys.argv[1:];

  if "--all" in args:
    test_files = sorted(glob.glob("tests/*.json"));
    for filepath in test_files:
      instance = load_test_case(filepath);
      if instance is not None:
        _solve_instance(instance, label=os.path.basename(filepath));
    return;

  if args and not args[0].startswith("--"):
    instance = load_test_case(args[0]);
    if instance:
      _solve_instance(instance, label=os.path.basename(args[0]));
    return;

  instance = _build_bakery_instance();
  _solve_instance(instance, label="Bakery Example");

if __name__ == "__main__":
  main();