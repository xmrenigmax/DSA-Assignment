# Algorithm Development Portfolio: Capacitated Vehicle Routing Problem (CVRP)

## 1. Problem Definition & Literature Review
The Capacitated Vehicle Routing Problem (CVRP) is an NP-hard combinatorial optimisation problem. The objective is to design optimal delivery routes for a fleet of homogeneous vehicles originating from a single depot to serve a set of customers with known demands. 

Let $G = (V, E)$ be a complete graph where $V = \{0, 1, ..., n\}$ is the set of vertices (with $0$ representing the depot) and $E$ is the set of edges. 
The objective function is to minimise the total travel distance:
$$\min \sum_{i=0}^{n} \sum_{j=0}^{n} d_{ij} x_{ij}$$
Where $d_{ij}$ is the distance between node $i$ and node $j$, and $x_{ij} \in \{0,1\}$ indicates if the edge is traversed. 

**Constraints:**
1. Each customer is visited exactly once by one vehicle.
2. The total demand of a route cannot exceed the vehicle capacity ($C$): $\sum_{i \in Route} demand_i \le C$.
3. Every route must start and terminate at the depot (node $0$).

*Literature:* Dantzig and Ramser (1959) first introduced the VRP. Given its NP-hard nature, exact solvers are computationally expensive for large datasets, making heuristic and metaheuristic approaches (like Genetic Algorithms and Local Search) the industry standard for scalable solutions.

---

## 2. Data Structures Justification
Careful selection of data structures was paramount to ensure efficient computation and memory management:
* **`Customer` Object with `__slots__`:** Python dictionaries have significant memory overhead. Using `__slots__` prevents the dynamic creation of `__dict__` attributes, heavily reducing the memory footprint, which is critical when scaling to thousands of customers.
* **Distance Matrix (2D Array):** A dense $n \times n$ array provides $O(1)$ lookups for edge costs, which is heavily utilized in millions of fitness evaluations during the Genetic Algorithm.
* **Routes as Python Lists:** Individual routes are stored as contiguous arrays (lists). This allows for $O(1)$ amortized appending during route construction and rapid slicing operations required for the 2-opt algorithms.

---

## 3. Algorithmic Solutions

### 3.1 Initial / Naive Solution: Clarke-Wright Savings
The Clarke-Wright algorithm (1964) is a greedy constructive heuristic.
* **Methodology:** It begins with a "star" configuration (every customer gets their own vehicle). It calculates the "savings" of merging two routes: $S(i,j) = d(0,i) + d(0,j) - d(i,j)$. Merges are executed greedily in descending order of savings, provided capacity constraints are respected.
* **Complexity:** Time complexity is $O(n^2 \log n)$ due to the sorting of the savings list. Space complexity is $O(n^2)$ to store the distance matrix.

### 3.2 AI-Generated Solution: Nearest Neighbour + 2-opt
* **AI Tool Used:** Claude (Anthropic).
* **Exact Prompt Used:** *"Implement a nearest-neighbour constructive heuristic for the Capacitated Vehicle Routing Problem in Python. After construction, apply a 2-opt intra-route improvement step to reduce the total travel distance. The function signature must be: run_ai_solution(distance_matrix, demands, vehicle_capacity) -> dict returning {'routes': list[list[int]], 'total_distance': float}."*
* **Methodology:** A greedy $O(n^2)$ traversal selecting the closest unvisited, feasible node. It is followed by a 2-opt local search which iteratively uncrosses overlapping edges within a single route until a local minimum is reached.

### 3.3 Optimised Solution: Genetic Algorithm with Inter-Route Search
To achieve superior results, a Genetic Algorithm (metaheuristic) was engineered.
* **Methodology:** A population of customer permutations evolves over generations. It utilizes *Binary Tournament Selection* to maintain genetic diversity and *Order Crossover (OX1)* to breed valid offspring without requiring repair mechanisms. 
* **Novelty:** After decoding chromosomes into capacity-aware routes, the algorithm applies an advanced **Inter-Route Relocate** local search. This $O(R^2 \cdot L^2)$ operation attempts to extract a customer from one route and insert them into another, often escaping the local optima that traps standard 2-opt algorithms.

---

## 4. Benchmarking and Comparative Analysis
Solutions were benchmarked across instances of varying sizes (6 to 50 customers).

| Solver | Distance | Gap (%) | Time (ms) | Routes | Valid |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Clarke-Wright (Naive) | 36.0000 | 16.13 | 0.045 | 3 | ✓ |
| Nearest Neighbour + 2-opt (AI) | 33.0000 | 6.45 | 0.061 | 3 | ✓ |
| Genetic Algorithm (Optimised) | 31.0000 | 0.00 | 85.102 | 3 | ✓ |
*Table 1: Benchmark results on `tc_02_small_10cust.json`*

**Analysis:**
The Clarke-Wright solution provides a rapid $O(n^2 \log n)$ baseline but lacks refinement. The AI-generated Nearest Neighbour approach slightly improved the gap due to the 2-opt uncrossing, but heavily depends on the starting node. The Genetic Algorithm consistently achieved the global minimum (0.00% Gap). While computationally heavier, the trade-off of milliseconds for vastly superior logistical efficiency is optimal for real-world scenarios.

---

## 5. Practical and Real-World Applications
The CVRP algorithms implemented here directly translate to modern industry challenges:
* **Last-Mile Delivery Logistics:** E-commerce giants use CVRP variants to distribute parcels, minimizing fleet fuel consumption and carbon emissions.
* **Waste Collection:** Municipalities map bins as "customers" with demand volume, optimizing garbage truck routes to prevent overflowing while respecting vehicle tonnage limits.
* **Public Transit Routing:** Ride-sharing systems dynamically group passengers (demand) into vans (capacity constraints) to optimize community travel times.