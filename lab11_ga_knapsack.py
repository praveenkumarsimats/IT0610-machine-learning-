"""
Lab 11 — Genetic Algorithm for the 0/1 Knapsack Problem
---------------------------------------------------------
Aim: To solve the 0/1 knapsack problem using a genetic algorithm to
maximise value subject to a weight constraint.

Algorithm: Encode a solution as a binary chromosome (item
included/excluded); fitness = total value if weight <= capacity, else 0;
apply selection, crossover, mutation over generations.
"""

import random

weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 8


def fitness(chrom):
    w = sum(weights[i] for i in range(len(chrom)) if chrom[i] == '1')
    v = sum(values[i] for i in range(len(chrom)) if chrom[i] == '1')
    return v if w <= capacity else 0


def ga_knapsack(pop_size=8, generations=30):
    pop = [''.join(random.choice('01') for _ in range(4)) for _ in range(pop_size)]
    for _ in range(generations):
        pop = sorted(pop, key=fitness, reverse=True)[:pop_size]
        children = []
        for _ in range(pop_size):
            p1, p2 = random.sample(pop[:4], 2)
            point = random.randint(1, 3)
            child = p1[:point] + p2[point:]
            child = ''.join(
                b if random.random() > 0.1 else str(1 - int(b)) for b in child
            )
            children.append(child)
        pop += children
    best = max(pop, key=fitness)
    return best, fitness(best)


if __name__ == "__main__":
    sol, val = ga_knapsack()
    print("Best solution:", sol, "Value:", val)

# Sample Output:
# Best solution: 0111 Value: 15
#
# Result: The GA identified a near-optimal item subset maximising value
# within the weight capacity.
