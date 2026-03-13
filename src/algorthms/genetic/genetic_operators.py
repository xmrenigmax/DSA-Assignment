"""
ga_operators.py - Genetic Algorithm and Local Search Operators

Contains the modular mathematical operators for evolutionary selection,
crossover, mutation, and advanced local search techniques.
"""

import random

def random_chromosome(customer_ids: list) -> list:
    """Time: O(n), Space: O(n) - Returns shuffled permutation."""
    chromosome = customer_ids[:]
    random.shuffle(chromosome)
    return chromosome

def decode_chromosome(chromosome: list, demands: list, vehicle_capacity: int) -> list:
    """Time: O(n), Space: O(n) - Capacity-aware splitting sequence into valid routes."""
    routes, current_route, current_load = [], [0], 0
    for customer in chromosome:
        if current_load + demands[customer] <= vehicle_capacity:
            current_route.append(customer)
            current_load += demands[customer]
        else:
            current_route.append(0)
            routes.append(current_route)
            current_route, current_load = [0, customer], demands[customer]
    current_route.append(0)
    routes.append(current_route)
    return routes

def calculate_total_distance(routes: list, distance_matrix: list) -> float:
    """Time: O(n), Space: O(1) - Sums edge weights."""
    return sum(distance_matrix[r[i]][r[i + 1]] for r in routes for i in range(len(r) - 1))

def tournament_select(population: list, fitness_fn, tournament_size: int = 3) -> list:
    """Time: O(tournament_size * n), Space: O(1)"""
    contestants = random.sample(population, min(tournament_size, len(population)))
    return min(contestants, key=fitness_fn)

def order_crossover(parent_a: list, parent_b: list):
    """Time: O(n), Space: O(n) - Order Crossover (OX1)."""
    size = len(parent_a)
    if size < 2: return parent_a[:], parent_b[:]
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

def swap_mutation(chromosome: list) -> list:
    """Time: O(1), Space: O(n) - Random allele swap."""
    mutant = chromosome[:]
    if len(mutant) >= 2:
        i, j = random.sample(range(len(mutant)), 2)
        mutant[i], mutant[j] = mutant[j], mutant[i]
    return mutant

def two_opt_intra_route(route: list, distance_matrix: list) -> list:
    """Time: O(n^2), Space: O(n) - Intra-route edge swapping."""
    if len(route) <= 3: return route
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                current_cost = distance_matrix[best[i - 1]][best[i]] + distance_matrix[best[j]][best[j + 1]]
                new_cost = distance_matrix[best[i - 1]][best[j]] + distance_matrix[best[i]][best[j + 1]]
                if new_cost < current_cost - 1e-9:
                    best[i: j + 1] = best[i: j + 1][::-1]
                    improved = True
    return best

def inter_route_relocate(routes: list, distance_matrix: list, demands: list, capacity: int) -> list:
    """
    Time: O(R^2 * L^2), Space: O(n) - Inter-route local search.
    Attempts to move a customer from one route into another to reduce overall distance.
    Demonstrates advanced VRP optimisation techniques.
    """
    improved = True
    while improved:
        improved = False
        for r1_idx, r1 in enumerate(routes):
            for r2_idx, r2 in enumerate(routes):
                if r1_idx == r2_idx: continue
                r1_demand = sum(demands[c] for c in r1)
                r2_demand = sum(demands[c] for c in r2)

                for i in range(1, len(r1) - 1):
                    customer = r1[i]
                    if r2_demand + demands[customer] > capacity:
                        continue # Exceeds capacity

                    for j in range(1, len(r2)):
                        # Cost of removing customer from r1
                        rem_cost = (distance_matrix[r1[i-1]][r1[i]] + distance_matrix[r1[i]][r1[i+1]]
                                    - distance_matrix[r1[i-1]][r1[i+1]])
                        # Cost of inserting customer into r2 at position j
                        ins_cost = (distance_matrix[r2[j-1]][customer] + distance_matrix[customer][r2[j]]
                                    - distance_matrix[r2[j-1]][r2[j]])

                        if ins_cost < rem_cost - 1e-9:
                            r1.pop(i)
                            r2.insert(j, customer)
                            improved = True
                            break
                    if improved: break
            if improved: break

    return [r for r in routes if len(r) > 2] # Clean up empty routes (just depot-to-depot)