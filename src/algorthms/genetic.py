"""
genetic.py - Genetic Algorithm for the Capacitated VRP (Optimised Solution)

Algorithm overview
------------------
A Genetic Algorithm (GA) is an evolutionary metaheuristic inspired by natural
selection (Holland, 1975).  A *population* of candidate solutions (chromosomes)
evolves over a number of *generations* via selection, crossover, and mutation.

Chromosome representation
  A chromosome is a permutation of customer indices (1 … n), encoding visit
  order.  Routes are decoded by a capacity-aware splitter: scan left-to-right
  and start a new route whenever the next customer would exceed capacity.

Operators
  Selection  : Binary tournament – randomly pick two chromosomes and keep the
               fitter one.  Preserves diversity better than roulette wheel.
  Crossover  : Order Crossover (OX1) – copies a random sub-sequence from
               parent A, then fills the remainder in the order they appear in
               parent B.  Guarantees valid permutations without repair.
  Mutation   : Swap mutation – randomly swap two alleles (customers) in the
               chromosome with probability p_mutation.

Elitism
  The single best chromosome from each generation is carried over unchanged
  to prevent regression.

Post-processing
  After the GA terminates, 2-opt local search is applied to each route of the
  best solution for further refinement.

Time complexity  : O(generations × population × n)  for fitness evaluations.
Space complexity : O(population × n).

Assumptions & limitations
--------------------------
* Homogeneous fleet; unlimited number of vehicles.
* GA parameters (population size, generations, mutation rate) are tuneable
  via function arguments.
* Stochastic; results vary between runs.  Use the `random_seed` parameter
  for reproducibility.
* No time-window or multi-depot constraints.

References
----------
Holland, J. H. (1975). *Adaptation in natural and artificial systems*.
    University of Michigan Press.
Davis, L. (1991). *Handbook of Genetic Algorithms*. Van Nostrand Reinhold.
"""

import random
import copy


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_optimised_solution(distance_matrix: list, demands: list,
                           vehicle_capacity: int,
                           population_size: int = 80,
                           generations: int = 300,
                           p_crossover: float = 0.85,
                           p_mutation: float = 0.15,
                           random_seed: int = None) -> dict:
    """
    Solve the CVRP using a Genetic Algorithm with 2-opt post-processing.

    Parameters
    ----------
    distance_matrix  : list[list[float]]  n×n cost matrix (index 0 = depot).
    demands          : list[int]          demand[i] for location i.
    vehicle_capacity : int                Maximum load per vehicle.
    population_size  : int                Number of chromosomes per generation.
    generations      : int                Number of evolutionary iterations.
    p_crossover      : float              Crossover probability ∈ [0, 1].
    p_mutation       : float              Mutation probability per chromosome.
    random_seed      : int | None         Seed for reproducibility.

    Returns
    -------
    dict  {"routes": list[list[int]], "total_distance": float}
    """
    if random_seed is not None:
        random.seed(random_seed)

    num_customers = len(distance_matrix) - 1  # exclude depot
    if num_customers == 0:
        return {"routes": [[0, 0]], "total_distance": 0.0}

    customer_ids = list(range(1, num_customers + 1))

    # ------------------------------------------------------------------ #
    # Initialise population
    # ------------------------------------------------------------------ #
    population = [_random_chromosome(customer_ids) for _ in range(population_size)]

    best_chromosome = min(population,
                          key=lambda c: _fitness(c, distance_matrix, demands, vehicle_capacity))
    best_fitness = _fitness(best_chromosome, distance_matrix, demands, vehicle_capacity)

    # ------------------------------------------------------------------ #
    # Evolve
    # ------------------------------------------------------------------ #
    for _generation in range(generations):
        new_population = [best_chromosome[:]]   # elitism: carry over best

        while len(new_population) < population_size:
            parent_a = _tournament_select(population, distance_matrix, demands, vehicle_capacity)
            parent_b = _tournament_select(population, distance_matrix, demands, vehicle_capacity)

            if random.random() < p_crossover:
                child_a, child_b = _order_crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a[:], parent_b[:]

            if random.random() < p_mutation:
                child_a = _swap_mutation(child_a)
            if random.random() < p_mutation:
                child_b = _swap_mutation(child_b)

            new_population.append(child_a)
            if len(new_population) < population_size:
                new_population.append(child_b)

        population = new_population

        # Track global best
        generation_best = min(population,
                              key=lambda c: _fitness(c, distance_matrix, demands, vehicle_capacity))
        generation_fitness = _fitness(generation_best, distance_matrix, demands, vehicle_capacity)
        if generation_fitness < best_fitness:
            best_fitness = generation_fitness
            best_chromosome = generation_best[:]

    # ------------------------------------------------------------------ #
    # Decode best chromosome → routes
    # ------------------------------------------------------------------ #
    routes = _decode_chromosome(best_chromosome, demands, vehicle_capacity)

    # ------------------------------------------------------------------ #
    # 2-opt post-processing on each route
    # ------------------------------------------------------------------ #
    improved_routes = [_two_opt(route, distance_matrix) for route in routes]

    total_distance = _total_distance(improved_routes, distance_matrix)
    return {"routes": improved_routes, "total_distance": total_distance}


# ---------------------------------------------------------------------------
# Chromosome helpers
# ---------------------------------------------------------------------------

def _random_chromosome(customer_ids: list) -> list:
    """Return a randomly shuffled permutation of customer_ids."""
    chromosome = customer_ids[:]
    random.shuffle(chromosome)
    return chromosome


def _fitness(chromosome: list, distance_matrix: list,
             demands: list, vehicle_capacity: int) -> float:
    """
    Decode a chromosome and return its total travel distance (lower = fitter).
    """
    routes = _decode_chromosome(chromosome, demands, vehicle_capacity)
    return _total_distance(routes, distance_matrix)


def _decode_chromosome(chromosome: list, demands: list,
                       vehicle_capacity: int) -> list:
    """
    Split a chromosome (customer permutation) into feasible routes.

    Customers are appended to the current route until the next customer
    would exceed the vehicle capacity, at which point a new route begins.

    Returns
    -------
    list[list[int]]  Each inner list: [0, c1, c2, ..., 0]
    """
    routes = []
    current_route = [0]
    current_load = 0

    for customer in chromosome:
        if current_load + demands[customer] <= vehicle_capacity:
            current_route.append(customer)
            current_load += demands[customer]
        else:
            current_route.append(0)
            routes.append(current_route)
            current_route = [0, customer]
            current_load = demands[customer]

    current_route.append(0)
    routes.append(current_route)
    return routes


def _total_distance(routes: list, distance_matrix: list) -> float:
    """Sum edge weights across all routes."""
    total = 0.0
    for route in routes:
        for i in range(len(route) - 1):
            total += distance_matrix[route[i]][route[i + 1]]
    return total


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _tournament_select(population: list, distance_matrix: list,
                       demands: list, vehicle_capacity: int,
                       tournament_size: int = 3) -> list:
    """
    Binary (or k-way) tournament selection.

    Pick `tournament_size` chromosomes at random and return the fittest.
    """
    contestants = random.sample(population, min(tournament_size, len(population)))
    return min(contestants,
               key=lambda c: _fitness(c, distance_matrix, demands, vehicle_capacity))


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def _order_crossover(parent_a: list, parent_b: list):
    """
    Order Crossover (OX1).

    1. Copy a random contiguous slice from parent_a into child_a at the same
       position.
    2. Fill the remaining positions of child_a with the values from parent_b
       in the order they appear (skipping values already in the slice).
    3. Repeat symmetrically for child_b.

    Guarantees that offspring are valid permutations.
    """
    size = len(parent_a)
    if size < 2:
        return parent_a[:], parent_b[:]

    start, end = sorted(random.sample(range(size), 2))

    def _ox(p1, p2):
        child = [None] * size
        child[start: end + 1] = p1[start: end + 1]
        segment_set = set(p1[start: end + 1])
        fill_pos = (end + 1) % size
        for gene in p2:
            if gene not in segment_set:
                child[fill_pos] = gene
                fill_pos = (fill_pos + 1) % size
        return child

    return _ox(parent_a, parent_b), _ox(parent_b, parent_a)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def _swap_mutation(chromosome: list) -> list:
    """
    Swap mutation: randomly swap two alleles.

    Returns a new chromosome list; the original is not modified.
    """
    mutant = chromosome[:]
    if len(mutant) < 2:
        return mutant
    i, j = random.sample(range(len(mutant)), 2)
    mutant[i], mutant[j] = mutant[j], mutant[i]
    return mutant


# ---------------------------------------------------------------------------
# 2-opt local search (intra-route)
# ---------------------------------------------------------------------------

def _two_opt(route: list, distance_matrix: list) -> list:
    """
    Apply 2-opt improvement to a single route (depot endpoints fixed).

    Iteratively reverses sub-sequences between positions i and j if doing
    so reduces the route cost.  Terminates when no improvement is found.
    """
    if len(route) <= 3:
        return route

    best = route[:]
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                current_cost = (distance_matrix[best[i - 1]][best[i]]
                                + distance_matrix[best[j]][best[j + 1]])
                new_cost = (distance_matrix[best[i - 1]][best[j]]
                            + distance_matrix[best[i]][best[j + 1]])
                if new_cost < current_cost - 1e-9:
                    best[i: j + 1] = best[i: j + 1][::-1]
                    improved = True

    return best
