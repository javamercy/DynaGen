import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        rng = self.rng

        # Population size
        NP = max(4, min(50, budget // (dim + 1)))
        if NP < 4:
            NP = 4

        # Initialize population
        pop = rng.uniform(lb, ub, size=(NP, dim))
        pop_fitness = np.full(NP, np.inf)
        calls = 0
        best_x = None
        best_val = np.inf

        # Evaluate initial population
        for i in range(NP):
            if calls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            calls += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        if calls >= budget:
            return best_val, best_x

        F = 0.8
        CR = 0.9
        generation = 0
        stagnation_counter = 0
        prev_best_val = best_val

        # Local search parameters
        sigma = max(1e-3, (ub - lb).mean() / 20)
        local_successes = 0
        local_failures = 0

        while calls < budget:
            generation += 1
            improved_this_gen = False
            for i in range(NP):
                if calls >= budget:
                    break
                # Mutation
                idxs = [j for j in range(NP) if j != i]
                a, b, c = rng.choice(idxs, size=3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.uniform() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                calls += 1
                # Selection
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        improved_this_gen = True

            # Check stagnation
            if improved_this_gen:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Local search after generation if no improvement
            if not improved_this_gen and calls < budget:
                remaining = budget - calls
                # Adaptive sigma
                if local_successes + local_failures > 0:
                    success_rate = local_successes / (local_successes + local_failures)
                    if success_rate > 0.2:
                        sigma *= 1.2  # increase exploration
                    else:
                        sigma *= 0.9  # reduce exploration
                sigma = np.clip(sigma, 1e-6, (ub - lb).mean())
                # Generate a perturbed point from best
                x = best_x + rng.normal(0, sigma, size=dim)
                x = np.clip(x, lb, ub)
                val = func(x)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                    local_successes += 1
                else:
                    local_failures += 1

            # Population restart if stagnation >= 10 generations
            if stagnation_counter >= 10 and calls < budget:
                # Reinitialize population, keep best individual
                new_pop = rng.uniform(lb, ub, size=(NP, dim))
                new_pop[0] = best_x.copy()
                for i in range(NP):
                    if calls >= budget:
                        break
                    x = np.clip(new_pop[i], lb, ub)
                    val = func(x)
                    calls += 1
                    new_pop_fitness = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                # Replace population (except first) with new ones
                # But we need to evaluate all new individuals? To save budget, we can just replace pop without evaluating all again, but then we lose fitness values. Instead, we evaluate sequentially.
                # For simplicity, evaluate all new points (except index 0 already evaluated as best)
                # However, this may use many calls. To be safe, we can just reinitialize the population and evaluate only those we can afford.
                # We'll limit to at most NP evaluations.
                # Actually, we already evaluated index 0, so we evaluate the rest.
                pop = [None] * NP
                pop_fitness = [None] * NP
                pop[0] = new_pop[0].copy()
                pop_fitness[0] = best_val  # known
                for i in range(1, NP):
                    if calls >= budget:
                        break
                    x = np.clip(new_pop[i], lb, ub)
                    val = func(x)
                    calls += 1
                    pop[i] = x
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                # Fill remaining with existing best if budget exhausted
                for i in range(NP):
                    if pop[i] is None:
                        pop[i] = best_x.copy()
                        pop_fitness[i] = best_val
                pop = np.array(pop)
                pop_fitness = np.array(pop_fitness)
                stagnation_counter = 0
                # Reset local search history
                local_successes = 0
                local_failures = 0

            # Occasional global random restart (5% chance per generation)
            if rng.uniform() < 0.05 and calls < budget:
                x = rng.uniform(lb, ub)
                val = func(x)
                calls += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)

        return best_val, best_x