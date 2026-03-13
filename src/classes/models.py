"""
models.py - Core data structures for the Vehicle Routing Problem (VRP)

Data structures are chosen to balance memory efficiency and access speed:
  - Customer: lightweight value object using __slots__ for memory efficiency
  - Vehicle:  tracks load state during route construction
  - Route:    list-backed structure for O(1) append and O(n) traversal
  - VRPInstance: top-level container that validates problem inputs on creation
"""

import math


# ---------------------------------------------------------------------------
# Customer
# ---------------------------------------------------------------------------
class Customer:
    """
    Represents a delivery customer (or the depot when customer_id == 0).

    Attributes
    ----------
    customer_id  : int   Unique index; 0 is reserved for the depot.
    demand       : int   Number of units required (0 for depot).
    x_coordinate : float Cartesian x-position (used to auto-build distance matrix).
    y_coordinate : float Cartesian y-position.
    """

    __slots__ = ("customer_id", "demand", "x_coordinate", "y_coordinate")

    def __init__(self, customer_id: int, demand: int,
                 x_coordinate: float = 0.0, y_coordinate: float = 0.0):
        if demand < 0:
            raise ValueError(f"Customer {customer_id}: demand cannot be negative.")
        self.customer_id = customer_id
        self.demand = demand
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate

    def __repr__(self) -> str:
        return (f"Customer(id={self.customer_id}, demand={self.demand}, "
                f"x={self.x_coordinate}, y={self.y_coordinate})")


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------
class Vehicle:
    """
    Represents a delivery vehicle with a fixed capacity.

    Attributes
    ----------
    vehicle_id    : int  Unique identifier.
    capacity      : int  Maximum load the vehicle can carry.
    current_load  : int  Load currently assigned (updated during route building).
    """

    def __init__(self, vehicle_id: int, capacity: int):
        if capacity <= 0:
            raise ValueError(f"Vehicle {vehicle_id}: capacity must be positive.")
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.current_load = 0

    @property
    def remaining_capacity(self) -> int:
        """How much more demand this vehicle can absorb."""
        return self.capacity - self.current_load

    def can_serve(self, demand: int) -> bool:
        """Return True if adding *demand* units does not exceed capacity."""
        return self.current_load + demand <= self.capacity

    def load(self, demand: int) -> None:
        """Add *demand* to current load (raises ValueError if over capacity)."""
        if not self.can_serve(demand):
            raise ValueError(
                f"Vehicle {self.vehicle_id}: loading {demand} would exceed "
                f"capacity {self.capacity} (current load {self.current_load})."
            )
        self.current_load += demand

    def reset(self) -> None:
        """Reset load to zero (call between route constructions)."""
        self.current_load = 0

    def __repr__(self) -> str:
        return (f"Vehicle(id={self.vehicle_id}, capacity={self.capacity}, "
                f"load={self.current_load})")


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
class Route:
    """
    An ordered sequence of node indices representing one vehicle's journey.

    The route always begins and ends at the depot (index 0).  Internally the
    node list is stored as a plain Python list for O(1) appends and O(n)
    iteration, which is optimal given that VRP route lengths are small.

    Attributes
    ----------
    nodes        : list[int]  Sequence of location indices (depot at start).
    total_demand : int        Accumulated demand of all customers in this route.
    """

    def __init__(self, depot_id: int = 0):
        self.nodes: list = [depot_id]
        self.total_demand: int = 0
        self._depot_id = depot_id

    def add_customer(self, customer: "Customer") -> None:
        """Append a customer to the route and update accumulated demand."""
        self.nodes.append(customer.customer_id)
        self.total_demand += customer.demand

    def close(self) -> None:
        """Append the depot to mark the end of the route (idempotent)."""
        if self.nodes[-1] != self._depot_id:
            self.nodes.append(self._depot_id)

    def calculate_distance(self, distance_matrix: list) -> float:
        """Return total travel distance for this route using *distance_matrix*."""
        total = 0.0
        closed = self.nodes if self.nodes[-1] == self._depot_id else self.nodes + [self._depot_id]
        for i in range(len(closed) - 1):
            total += distance_matrix[closed[i]][closed[i + 1]]
        return total

    def __len__(self) -> int:
        """Number of nodes in the route (including depot occurrences)."""
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"Route(nodes={self.nodes}, demand={self.total_demand})"


# ---------------------------------------------------------------------------
# VRPInstance
# ---------------------------------------------------------------------------
class VRPInstance:
    """
    Top-level container that holds all data for one VRP problem instance.

    Parameters
    ----------
    customers        : list[Customer]   Index 0 must be the depot (demand=0).
    distance_matrix  : list[list[float]]  Square matrix; d[i][j] = travel cost.
    vehicle_capacity : int               Homogeneous fleet capacity.
    num_vehicles     : int               Maximum number of vehicles available.
    """

    def __init__(self, customers: list, distance_matrix: list,
                 vehicle_capacity: int, num_vehicles: int):
        self._validate(customers, distance_matrix, vehicle_capacity, num_vehicles)
        self.customers = customers
        self.distance_matrix = distance_matrix
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def num_customers(self) -> int:
        """Number of customers excluding the depot."""
        return len(self.customers) - 1

    @property
    def demands(self) -> list:
        """List of demands indexed by customer_id (index 0 = depot = 0)."""
        return [c.demand for c in self.customers]

    # ------------------------------------------------------------------
    # Factory: build from coordinates
    # ------------------------------------------------------------------
    @classmethod
    def from_coordinates(cls, customers: list, vehicle_capacity: int,
                         num_vehicles: int) -> "VRPInstance":
        """
        Construct a VRPInstance by computing Euclidean distances from
        customer (x, y) coordinates.

        Time complexity: O(n²) where n = len(customers).
        """
        n = len(customers)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = customers[i].x_coordinate - customers[j].x_coordinate
                    dy = customers[i].y_coordinate - customers[j].y_coordinate
                    matrix[i][j] = math.hypot(dx, dy)
        return cls(customers, matrix, vehicle_capacity, num_vehicles)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(customers, distance_matrix, vehicle_capacity, num_vehicles):
        n = len(customers)
        if n == 0:
            raise ValueError("customers list must not be empty.")
        if customers[0].demand != 0:
            raise ValueError("Index 0 must be the depot with demand=0.")
        if len(distance_matrix) != n or any(len(row) != n for row in distance_matrix):
            raise ValueError("distance_matrix must be an n×n square matrix.")
        if vehicle_capacity <= 0:
            raise ValueError("vehicle_capacity must be positive.")
        if num_vehicles <= 0:
            raise ValueError("num_vehicles must be positive.")
        total_demand = sum(c.demand for c in customers)
        max_capacity = vehicle_capacity * num_vehicles
        if total_demand > max_capacity:
            raise ValueError(
                f"Total demand ({total_demand}) exceeds fleet capacity "
                f"({num_vehicles} × {vehicle_capacity} = {max_capacity})."
            )

    def __repr__(self) -> str:
        return (f"VRPInstance(customers={self.num_customers}, "
                f"capacity={self.vehicle_capacity}, vehicles={self.num_vehicles})")
