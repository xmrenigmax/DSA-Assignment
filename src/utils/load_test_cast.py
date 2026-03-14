import json
import random
import math

def generate_complex_test(filename, num_customers, vehicle_capacity, distribution_type="clustered"):
  # Customer 0 is the depot
  customers = [{"id": 0, "x": 50, "y": 50, "demand": 0}]

  if distribution_type == "clustered":
    # Create 4 distinct clusters (neighborhoods)
    centers = [(20, 20), (20, 80), (80, 20), (80, 80)]
    for i in range(1, num_customers + 1):
      center = centers[i % 4]
      # Customers are grouped tightly around center points
      x = center[0] + random.uniform(-10, 10)
      y = center[1] + random.uniform(-10, 10)
      demand = random.randint(2, 8)
      customers.append({"id": i, "x": x, "y": y, "demand": demand})
  else:
    # Fully random distribution
    for i in range(1, num_customers + 1):
      customers.append({
        "id": i,
        "x": random.uniform(0, 100),
        "y": random.uniform(0, 100),
        "demand": random.randint(1, 9)
      })

  # Build Distance Matrix (Euclidean)
  size = len(customers)
  matrix = [[0.0] * size for _ in range(size)]
  for i in range(size):
    for j in range(size):
      if i != j:
        d = math.hypot(customers[i]['x'] - customers[j]['x'],
        customers[i]['y'] - customers[j]['y'])
        matrix[i][j] = round(d, 4)

  test_data = {
    "vehicle_capacity": vehicle_capacity,
    "num_vehicles": num_customers // 2, # Provide plenty of vehicles
    "demands": [c['demand'] for c in customers],
    "distance_matrix": matrix
  }

  with open(f"tests/{filename}", "w") as f:
    json.dump(test_data, f, indent=2)
  print(f"Generated {filename}")

# Create two super complex tests
# 1. Stress Test: Large scale, random distribution
generate_complex_test("tc_06_stress_100cust.json", 100, 40, "random")

# 2. Hard Test: Clustered distribution with tight capacity
generate_complex_test("tc_07_clustered_120cust.json", 120, 30, "clustered")