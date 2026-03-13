import time;
import math;

from utils.core_utils import validate_solution;

def run_benchmark(solvers: dict, instance, runs: int = 1) -> list:
  results = [];
  dm = instance.distance_matrix;
  demands = instance.demands;
  cap = instance.vehicle_capacity;

  for label, solver_fn in solvers.items():
    best_distance = math.inf;
    best_solution = None;
    total_time = 0.0;

    for _ in range(runs):
      t0 = time.perf_counter();
      sol = solver_fn(dm, demands, cap);
      t1 = time.perf_counter();
      total_time += (t1 - t0) * 1000;
      
      if sol["total_distance"] < best_distance:
        best_distance = sol["total_distance"];
        best_solution = sol;

    validity = validate_solution(best_solution, demands, cap, instance.num_customers);
    
    results.append({
      "solver": label,
      "distance": round(best_distance, 4),
      "time_ms": round(total_time / runs, 3),
      "routes": len(best_solution["routes"]),
      "valid": validity["valid"]
    });

  best_overall = min(r["distance"] for r in results);
  for r in results:
    r["gap"] = ((r["distance"] - best_overall) / best_overall * 100) if best_overall > 0 else 0.0;

  return results;

def print_benchmark_table(results: list) -> None:
  header = f"{'Solver':<30} {'Distance':>12} {'Gap (%)':>10} {'Time (ms)':>12} {'Routes':>8} {'Valid':>7}";
  print("\n" + "=" * 85);
  print("  Benchmark Results");
  print("=" * 85);
  print(f"  { header }");
  print(f"  {'─' * 81}");
  
  for r in results:
    valid_str = "✓" if r["valid"] else "✗";
    print(f"  {r['solver']:<30} {r['distance']:>12.4f} {r['gap']:>10.2f} {r['time_ms']:>12.3f} {r['routes']:>8} {valid_str:>7}");
  print("=" * 85 + "\n");