import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Population size: at least 4*dim, capped to budget//2, min 3
        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

        # Initialization
        pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # JADE parameters
        mu_F = 0.5
        mu_CR = 0.5
        p = 0.1  # fraction of pbest
        num_pbest = max(2, int(p * pop_size))

        # Stagnation detection
        gen_max_restart = max(1, budget // (2 * pop_size))
        gen_no_improve = 0
        prev_best_val = best_val

        while evals < budget:
            # Sort population by fitness for pbest selection
            sort_idx = np.argsort(fitness)
            pbest_pool = sort_idx[:num_pbest]

            # Store successful parameters for adaptation
            successful_F = []
            successful_CR = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                # Generate F_i using Cauchy distribution
                F_i = mu_F + 0.1 * rng.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + 0.1 * rng.standard_cauchy()
                F_i = min(F_i, 1.0)

                # Generate CR_i using normal distribution
                CR_i = mu_CR + 0.1 * rng.randn()
                CR_i = np.clip(CR_i, 0, 1)

                # Select pbest (distinct from i)
                cand = [idx for idx in pbest_pool if idx != i]
                if not cand:
                    cand = pbest_pool
                pbest_idx = rng.choice(cand)

                # Select two random distinct indices from population excluding i and pbest? Actually standard JADE: r1 from population, r2 from union of population and archive. Here we skip archive, so r1 and r2 from population, all distinct.
                indices = [j for j in range(pop_size) if j != i and j != pbest_idx]
                if len(indices) < 2:
                    # Should not happen if pop_size >= 3, but handle edge
                    indices = list(range(pop_size))
                    indices.remove(i)
                r1, r2 = rng.choice(indices, size=2, replace=False)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)

                # Crossover binomial
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluation
                trial_fit = func(trial)
                evals += 1

                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    successful_F.append(F_i)
                    successful_CR.append(CR_i)
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Update parameter means using successful values
            if len(successful_F) > 0:
                sum_F = np.sum(successful_F)
                sum_F2 = np.sum(np.square(successful_F))
                if sum_F > 0:
                    mu_F = sum_F2 / sum_F
                mu_CR = np.mean(successful_CR)

            # Check stagnation
            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            if gen_no_improve >= gen_max_restart and evals < budget:
                # Restart: reinitialize population but keep best
                new_pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
                new_pop[0] = best_x
                # Add noise to some individuals for diversity
                for i in range(1, pop_size):
                    noise = rng.normal(0, 0.1 * (ub - lb), size=dim)
                    new_pop[i] = np.clip(new_pop[i] + noise, lb, ub)
                pop = new_pop
                fitness = np.full(pop_size, np.inf)
                fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    val = func(pop[i])
                    evals += 1
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                # Reset parameter means
                mu_F = 0.5
                mu_CR = 0.5
                prev_best_val = best_val
                gen_no_improve = 0

        return best_val, best_x