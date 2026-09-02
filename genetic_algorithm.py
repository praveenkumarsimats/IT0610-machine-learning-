"""
genetic_algorithm.py
---------------------
A Genetic Algorithm implemented from scratch (no optimisation libraries)
used to search the MLP's initial-weight / hyperparameter space.

Chromosome encoding:
    A chromosome is a real-valued vector = [flattened MLP weights & biases]
    concatenated with two hyperparameter genes: [n_hidden_units, learning_rate].
    n_hidden_units is decoded to an integer in [4, 32]; learning_rate is
    decoded to a float in [0.001, 0.2] (log-scale search).

Fitness function:
    fitness(chromosome) = validation F1-score of an MLP that is (a) built
    with the decoded hidden-layer size, (b) initialised with the decoded
    weights, and (c) trained for a small, fixed number of "shake-down"
    epochs on the training fold. F1 is chosen over accuracy because the
    dataset is imbalanced (roughly 65:35 negative:positive), so a fitness
    signal that ignores recall would let the GA collapse to a majority
    classifier.

Operators:
    Selection : tournament selection (k=3)
    Crossover : uniform crossover (per-gene Bernoulli mixing), p_c = 0.8
    Mutation  : Gaussian mutation on weight genes + resampling mutation on
                hyperparameter genes, p_m = 0.1, with elitism (best 2
                individuals carried over unchanged each generation).
"""
import numpy as np
from src.mlp import MLP
from src.evaluation import f1_score


N_HIDDEN_MIN, N_HIDDEN_MAX = 4, 32
LR_MIN, LR_MAX = 0.001, 0.2


def decode_hparams(h_gene, lr_gene):
    n_hidden = int(np.clip(round(N_HIDDEN_MIN + h_gene * (N_HIDDEN_MAX - N_HIDDEN_MIN)),
                            N_HIDDEN_MIN, N_HIDDEN_MAX))
    # log-scale decode for learning rate
    log_lr = np.log10(LR_MIN) + lr_gene * (np.log10(LR_MAX) - np.log10(LR_MIN))
    lr = float(10 ** log_lr)
    return n_hidden, lr


class GeneticAlgorithm:
    def __init__(self, input_dim, X_train, y_train, X_val, y_val,
                 pop_size=20, generations=25, crossover_rate=0.8,
                 mutation_rate=0.1, elitism=2, shakedown_epochs=15,
                 random_state=42):
        self.input_dim = input_dim
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.shakedown_epochs = shakedown_epochs
        self.rng = np.random.RandomState(random_state)

        # chromosome length = max possible weight params (for N_HIDDEN_MAX)
        # + 2 hyperparameter genes. We use a fixed max-size encoding and
        # only use the first n_hidden columns of the hidden-layer weight
        # matrix ("growing" network encoding), so crossover/mutation
        # always operate on same-length vectors.
        self.max_weight_len = MLP([input_dim, N_HIDDEN_MAX, 1]).n_params()
        self.chrom_len = self.max_weight_len + 2
        self.history = {"best_fitness": [], "mean_fitness": []}

    def _init_population(self):
        pop = self.rng.uniform(-1, 1, size=(self.pop_size, self.chrom_len))
        # hyperparameter genes must be in [0, 1] (decoded later)
        pop[:, -2:] = self.rng.uniform(0, 1, size=(self.pop_size, 2))
        return pop

    def _build_mlp_from_chromosome(self, chrom):
        n_hidden, lr = decode_hparams(chrom[-2], chrom[-1])
        weight_flat_full = chrom[:self.max_weight_len]

        # Slice out the sub-network for n_hidden units from the max-size
        # encoding (growing/shrinking network trick).
        w1_full_len = self.input_dim * N_HIDDEN_MAX
        b1_full_len = N_HIDDEN_MAX
        w2_full_len = N_HIDDEN_MAX * 1
        b2_full_len = 1

        w1_full = weight_flat_full[:w1_full_len].reshape(self.input_dim, N_HIDDEN_MAX)
        b1_full = weight_flat_full[w1_full_len:w1_full_len + b1_full_len].reshape(1, N_HIDDEN_MAX)
        w2_start = w1_full_len + b1_full_len
        w2_full = weight_flat_full[w2_start: w2_start + w2_full_len].reshape(N_HIDDEN_MAX, 1)
        b2_full = weight_flat_full[w2_start + w2_full_len: w2_start + w2_full_len + b2_full_len].reshape(1, 1)

        w1 = w1_full[:, :n_hidden] * np.sqrt(2.0 / self.input_dim)
        b1 = b1_full[:, :n_hidden]
        w2 = w2_full[:n_hidden, :] * np.sqrt(2.0 / max(n_hidden, 1))
        b2 = b2_full

        mlp = MLP([self.input_dim, n_hidden, 1], learning_rate=lr,
                   init_weights=([w1, w2], [b1, b2]))
        return mlp

    def fitness(self, chrom):
        mlp = self._build_mlp_from_chromosome(chrom)
        mlp.train(self.X_train, self.y_train, epochs=self.shakedown_epochs,
                  batch_size=32, patience=self.shakedown_epochs, verbose=False)
        val_pred = mlp.predict(self.X_val)
        return f1_score(self.y_val, val_pred)

    def _tournament_select(self, pop, fitnesses, k=3):
        idx = self.rng.choice(len(pop), size=k, replace=False)
        best_idx = idx[np.argmax(fitnesses[idx])]
        return pop[best_idx].copy()

    def _crossover(self, p1, p2):
        if self.rng.rand() > self.crossover_rate:
            return p1.copy(), p2.copy()
        mask = self.rng.rand(self.chrom_len) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    def _mutate(self, chrom):
        chrom = chrom.copy()
        weight_mask = self.rng.rand(self.max_weight_len) < self.mutation_rate
        chrom[:self.max_weight_len][weight_mask] += self.rng.normal(
            0, 0.3, size=weight_mask.sum())
        if self.rng.rand() < self.mutation_rate:
            chrom[-2] = self.rng.uniform(0, 1)
        if self.rng.rand() < self.mutation_rate:
            chrom[-1] = self.rng.uniform(0, 1)
        return chrom

    def run(self, verbose=True):
        pop = self._init_population()
        fitnesses = np.array([self.fitness(c) for c in pop])

        for gen in range(self.generations):
            elite_idx = np.argsort(fitnesses)[-self.elitism:]
            elites = [pop[i].copy() for i in elite_idx]

            new_pop = []
            while len(new_pop) < self.pop_size - self.elitism:
                p1 = self._tournament_select(pop, fitnesses)
                p2 = self._tournament_select(pop, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1, c2 = self._mutate(c1), self._mutate(c2)
                new_pop.extend([c1, c2])
            new_pop = new_pop[:self.pop_size - self.elitism] + elites
            pop = np.array(new_pop)
            fitnesses = np.array([self.fitness(c) for c in pop])

            self.history["best_fitness"].append(fitnesses.max())
            self.history["mean_fitness"].append(fitnesses.mean())
            if verbose:
                print(f"Gen {gen+1}/{self.generations}: "
                      f"best F1={fitnesses.max():.4f} mean F1={fitnesses.mean():.4f}")

        best_idx = np.argmax(fitnesses)
        best_chrom = pop[best_idx]
        best_mlp = self._build_mlp_from_chromosome(best_chrom)
        n_hidden, lr = decode_hparams(best_chrom[-2], best_chrom[-1])
        return best_mlp, {"n_hidden": n_hidden, "learning_rate": lr,
                           "best_fitness": fitnesses[best_idx]}
