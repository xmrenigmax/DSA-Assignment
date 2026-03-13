"""
genetic.py - Genetic Algorithm for the Capacitated VRP (Optimised Solution)

Algorithm overview
------------------
A Genetic Algorithm (GA) is an evolutionary metaheuristic. This implementation
uses Binary Tournament Selection, Order Crossover (OX1), and Swap Mutation.
Routes are decoded via a capacity-aware greedy split.

Time complexity  : O(generations * population_size * n)
Space complexity : O(population_size * n)
"""

import random
import logging
from .genetic_operators import (
    random_chromosome, tournament_select, order_crossover,
    swap_mutation, two_opt_intra_route, inter_route_relocate,
    decode_chromosome, calculate_total_distance
)

# Professional logging setup replacing standard print statements
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_optimised_solution(distance_matrix: list, demands: list,
    vehicle_capacity: int,
    population_size: int = 80,
    generations: int = 300,
    p_crossover: float = 0.85,
    p_mutation: float = 0.15,
    random_seed: int = None) -> dict:
    """
    Solve the CVRP using a GA with 2-opt and inter-route post-processing.
    """
    # 1. Edge Case & Input Validation
    if not distance_matrix or vehicle_capacity <= 0:
        logger.error("Invalid input: Distance matrix empty or capacity <= 0.")
        return {"routes": [], "total_distance": 0.0}

    num_customers = len(distance_matrix) - 1
    if num_customers <= 0:
        return {"routes": [[0, 0]], "total_distance": 0.0}

    if random_seed is not None:
        random.seed(random_seed)

    customer_ids = list(range(1, num_customers + 1))

    # Helper function to evaluate fitness dynamically
    def get_fitness(chromosome):
        routes = decode_chromosome(chromosome, demands, vehicle_capacity)
        return calculate_total_distance(routes, distance_matrix)

    # 2. Initialise Population
    population = [random_chromosome(customer_ids) for _ in range(population_size)]
    best_chromosome = min(population, key=get_fitness)
    best_fitness = get_fitness(best_chromosome)

    # 3. Evolution Loop
    for _ in range(generations):
        new_population = [best_chromosome[:]]  # Elitism

        while len(new_population) < population_size:
            parent_a = tournament_select(population, get_fitness)
            parent_b = tournament_select(population, get_fitness)

            # Crossover
            if random.random() < p_crossover:
                child_a, child_b = order_crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a[:], parent_b[:]

            # Mutation
            if random.random() < p_crossover:
                child_a = swap_mutation(child_a)
            if random.random() < p_mutation:
                child_b = swap_mutation(child_b)

            new_population.extend([child_a, child_b])

        population = new_population[:population_size]

        # Track global best
        generation_best = min(population, key=get_fitness)
        generation_fitness = get_fitness(generation_best)
        if generation_fitness < best_fitness:
            best_fitness = generation_fitness
            best_chromosome = generation_best[:]

    # 4. Decode & Post-Processing (Local Search)
    routes = decode_chromosome(best_chromosome, demands, vehicle_capacity)

    # Intra-route optimization (2-opt)
    optimised_routes = [two_opt_intra_route(r, distance_matrix) for r in routes]

    # Inter-route optimization (Relocate)
    optimised_routes = inter_route_relocate(
        optimised_routes, distance_matrix, demands, vehicle_capacity
    )

    final_distance = calculate_total_distance(optimised_routes, distance_matrix)

    logger.debug(f"GA Completed. Best distance: {final_distance:.4f}")
    return {"routes": optimised_routes, "total_distance": final_distance}