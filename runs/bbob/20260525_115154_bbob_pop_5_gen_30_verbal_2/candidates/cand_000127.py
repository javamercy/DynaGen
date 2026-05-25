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

        pop_size = max(4 * dim, 5)
        archive = []
        archive_size = pop_size

        # Initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

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

        # Main loop
        gen_no_improve = 0
        prev_best = best_val
        while evals < budget:
            # Sort indices for pbest selection
            sort_idx = np.argsort(fitness)
            pbest_pool = sort_idx[:max(2, int(0.2 * pop_size))]

            # Generation loop
            for i in range(pop_size):
                if evals >= budget:
                    break

                # Generate F and CR
                F = 0.5 + 0.1 * rng.standard_cauchy()
                F = max(0.0, min(F, 1.0))
                CR = 0.9 + 0.1 * rng.randn()
                CR = np.clip(CR, 0.0, 1.0)

                # Select pbest
                candidates = [idx for idx in pbest_pool if idx != i]
                if not candidates:
                    candidates = pbest_pool
                pbest_idx = rng.choice(candidates)

                # Select r1
                candidates_r1 = [j for j in range(pop_size) if j not in (i, pbest_idx)]
                if len(candidates_r1) == 0:
                    continue
                r1 = rng.choice(candidates_r1)

                # Select r2 from union of pop and archive
                candidates_r2 = [j for j in range(pop_size) if j not in (i, pbest_idx, r1)]
                if archive:
                    candidates_r2.extend(archive)
                if len(candidates_r2) == 0:
                    continue
                pick = rng.randint(len(candidates_r2))
                r2 = candidates_r2[pick]
                if isinstance(r2, int):
                    r2 = pop[r2]

                # Mutation and crossover
                mutant = pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (pop[r1] - r2)
                mutant = np.clip(mutant, lb, ub)

                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluation
                trial_fit = func(trial)
                evals += 1

                if trial_fit < fitness[i]:
                    archive.append(pop[i].copy())
                    if len(archive) > archive_size:
                        archive.pop(rng.randint(len(archive)))
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if best_val < prev_best:
                gen_no_improve = 0
                prev_best = best_val
            else:
                gen_no_improve += 1

            # Restart condition
            pop_std = np.std(pop, axis=0)
            norm_std = pop_std / (ub - lb)
            diversity_ok = np.mean(norm_std) > 1e-4
            if (gen_no_improve >= 50 or not diversity_ok) and evals < budget:
                # Reinitialize population around best
                new_pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
                new_pop[0] = best_x
                for i in range(1, pop_size):
                    noise = rng.normal(0, pop_std, size=dim)
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
                archive.clear()
                gen_no_improve = 0
                prev_best = best_val

        return best_val, best_x