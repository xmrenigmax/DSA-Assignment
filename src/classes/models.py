"""
models.py - Core data structures for the Vehicle Routing Problem (VRP)
"""

import math;

class Customer:
  """Time: O(1) init, Space: O(1) per instance using __slots__."""
  __slots__ = ("customer_id", "demand", "x_coordinate", "y_coordinate");

  def __init__(self, customer_id: int, demand: int, x_coordinate: float = 0.0, y_coordinate: float = 0.0):
    if demand < 0:
      raise ValueError(f"Customer {customer_id}: demand cannot be negative.");
    self.customer_id = customer_id;
    self.demand = demand;
    self.x_coordinate = x_coordinate;
    self.y_coordinate = y_coordinate;

  def __repr__(self) -> str:
    return f"Customer(id={self.customer_id}, demand={self.demand}, x={self.x_coordinate}, y={self.y_coordinate})";

class Vehicle:
  """Time: O(1) operations, Space: O(1)."""
  def __init__(self, vehicle_id: int, capacity: int):
    if capacity <= 0:
      raise ValueError(f"Vehicle {vehicle_id}: capacity must be positive.");
    self.vehicle_id = vehicle_id;
    self.capacity = capacity;
    self.current_load = 0;

  @property
  def remaining_capacity(self) -> int:
    return self.capacity - self.current_load;

  def can_serve(self, demand: int) -> bool:
    return self.current_load + demand <= self.capacity;

  def load(self, demand: int) -> None:
    if not self.can_serve(demand):
      raise ValueError(f"Vehicle {self.vehicle_id}: loading {demand} exceeds capacity.");
    self.current_load += demand;

  def reset(self) -> None:
    self.current_load = 0;

  def __repr__(self) -> str:
    return f"Vehicle(id={self.vehicle_id}, capacity={self.capacity}, load={self.current_load})";

class Route:
  """Time: O(1) appends, Space: O(n) where n is route length."""
  def __init__(self, depot_id: int = 0):
    self.nodes: list = [depot_id];
    self.total_demand: int = 0;
    self._depot_id = depot_id;

  def add_customer(self, customer: "Customer") -> None:
    self.nodes.append(customer.customer_id);
    self.total_demand += customer.demand;

  def close(self) -> None:
    if self.nodes[-1] != self._depot_id:
      self.nodes.append(self._depot_id);

  def calculate_distance(self, distance_matrix: list) -> float:
    """Time: O(n) to traverse route and sum distances."""
    total = 0.0;
    closed = self.nodes if self.nodes[-1] == self._depot_id else self.nodes + [self._depot_id];
    for i in range(len(closed) - 1):
      total += distance_matrix[closed[i]][closed[i + 1]];
    return total;

  def __len__(self) -> int:
    return len(self.nodes);

  def __repr__(self) -> str:
    return f"Route(nodes={self.nodes}, demand={self.total_demand})";

class VRPInstance:
  """Time: O(n^2) for from_coordinates, Space: O(n^2) for distance matrix."""
  def __init__(self, customers: list, distance_matrix: list, vehicle_capacity: int, num_vehicles: int):
    self._validate(customers, distance_matrix, vehicle_capacity, num_vehicles);
    self.customers = customers;
    self.distance_matrix = distance_matrix;
    self.vehicle_capacity = vehicle_capacity;
    self.num_vehicles = num_vehicles;

  @property
  def num_customers(self) -> int:
    return len(self.customers) - 1;

  @property
  def demands(self) -> list:
    return [c.demand for c in self.customers];

  @classmethod
  def from_coordinates(cls, customers: list, vehicle_capacity: int, num_vehicles: int) -> "VRPInstance":
    n = len(customers);
    matrix = [[0.0] * n for _ in range(n)];
    for i in range(n):
      for j in range(n):
        if i != j:
          dx = customers[i].x_coordinate - customers[j].x_coordinate;
          dy = customers[i].y_coordinate - customers[j].y_coordinate;
          matrix[i][j] = math.hypot(dx, dy);
    return cls(customers, matrix, vehicle_capacity, num_vehicles);

  @staticmethod
  def _validate(customers, distance_matrix, vehicle_capacity, num_vehicles):
    n = len(customers);
    if n == 0:
      raise ValueError("customers list must not be empty.");
    if customers[0].demand != 0:
      raise ValueError("Index 0 must be the depot with demand=0.");
    if len(distance_matrix) != n or any(len(row) != n for row in distance_matrix):
      raise ValueError("distance_matrix must be an n*n square matrix.");
    if vehicle_capacity <= 0:
      raise ValueError("vehicle_capacity must be positive.");
    if num_vehicles <= 0:
      raise ValueError("num_vehicles must be positive.");
    total_demand = sum(c.demand for c in customers);
    max_capacity = vehicle_capacity * num_vehicles;
    if total_demand > max_capacity:
      raise ValueError(f"Total demand exceeds fleet capacity.");

  def __repr__(self) -> str:
    return f"VRPInstance(customers={self.num_customers}, capacity={self.vehicle_capacity}, vehicles={self.num_vehicles})";