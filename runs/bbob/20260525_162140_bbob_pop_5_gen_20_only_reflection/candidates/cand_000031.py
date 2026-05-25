import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        budget = self.budget

        # Population size
        NP = min(10 * dim, max(4, budget // 2 - 1))
        if NP < 2:
            NP = 2
        if budget < NP:
            # Evaluate all points anyway
            best_x = lb + (ub - lb) * rng.rand(budget, dim)
            best_val = float('inf')
            for i in range(budget):
                val = func(best_x[i])
                if val < best_val:
                    best_val = val
                    best_x_pt = best_x[i].copy()
                    report_best(best_val, best_x_pt)
            return best_val, best_x_pt

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.full(NP, np.inf)
        for i in range(NP):
            fitness[i] = func(pop[i])
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_val = fitness[best_idx]
        report_best(best_val, best_x)
        func_evals = NP

        # SHADE memory
        H = 5
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        archive_max = NP  # same size as pop

        # Restart parameters
        no_improve_evals = 0
        max_no_improve_evals = 5 * dim
        restart_diversity_threshold = 0.01 * np.mean(ub - lb)

        # Main loop
        while func_evals < budget:
            # Check restart condition: no improvement for enough evaluations and low diversity
            if func_evals > NP:
                div = np.mean(np.std(pop, axis=0))
                if no_improve_evals >= max_no_improve_evals and div < restart_diversity_threshold:
                    # Restart: keep best, reinitialize rest
                    remaining = budget - func_evals
                    new_NP = min(10 * dim, max(4, remaining // 2 - 1))
                    if new_NP < 2:
                        new_NP = 2
                    # Keep best
                    new_pop = np.zeros((new_NP, dim))
                    new_fit = np.full(new_NP, np.inf)
                    new_pop[0] = best_x
                    new_fit[0] = best_val
                    # Generate new points
                    for i in range(1, new_NP):
                        x = lb + (ub - lb) * rng.rand(dim)
                        new_pop[i] = x
                        val = func(x)
                        func_evals += 1
                        new_fit[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        if func_evals >= budget:
                            break
                    pop = new_pop
                    fitness = new_fit
                    NP = new_NP
                    no_improve_evals = 0
                    archive = []
                    # Reset memory? Keep or not? Keep memory.
                    continue

            # Generate one generation
            new_pop = pop.copy()
            new_fit = fitness.copy()
            improved = False
            for i in range(NP):
                if func_evals >= budget:
                    break

                # Mutation: current-to-pbest/1 with archive
                # pbest selection (top 20%)
                sorted_idx = np.argsort(fitness)
                p = 0.2
                pbest_size = max(1, int(p * NP))
                pbest_idx = rng.randint(pbest_size)
                pbest = pop[sorted_idx[pbest_idx]]

                # Select two distinct indices from pop ∪ archive, not equal to i
                union_pop = list(range(NP))
                union_pop.remove(i)
                union_indices = union_pop + list(range(len(archive)))
                if len(union_indices) < 2:
                    continue  # cannot mutate
                chosen = rng.choice(union_indices, 2, replace=False)
                idx_a = chosen[0]
                idx_b = chosen[1]
                if idx_a < NP:
                    a = pop[idx_a]
                else:
                    a = archive[idx_a - NP]
                if idx_b < NP:
                    b = pop[idx_b]
                else:
                    b = archive[idx_b - NP]

                # Sample F and CR from memory
                r = rng.randint(H)
                F = np.clip(rng.normal(M_F[r], 0.1), 0, 1)
                CR = np.clip(rng.normal(M_CR[r], 0.1), 0, 1)

                # Mutation
                mutant = pop[i] + F * (pbest - pop[i]) + F * (a - b)

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)

                # Evaluate
                trial_fit = func(trial)
                func_evals += 1
                no_improve_evals += 1

                if trial_fit < fitness[i]:
                    # Add parent to archive
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(rng.randint(len(archive)))
                    # Update population
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
                    improved = True
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                    # Update memory
                    M_F = np.roll(M_F, -1)
                    M_F[-1] = F
                    M_CR = np.roll(M_CR, -1)
                    M_CR[-1] = CR

            if not improved:
                # If no improved in generation, keep old population? Actually we already copy; replace anyway
                pass
            pop = new_pop
            fitness = new_fit
            # Reset no_improve_evals if any improvement happened?
            if improved:
                no_improve_evals = 0

        return best_val, best_x