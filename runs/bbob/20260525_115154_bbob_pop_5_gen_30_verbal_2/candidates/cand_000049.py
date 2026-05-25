import numpy as np
from numpy.random import RandomState

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Population size: at least 4, scaled with dim, capped by budget
        NP = min(max(4, 4 * dim), budget // 2)
        if NP < 4:
            # Fallback to random search
            best_val = np.inf
            best_x = None
            for _ in range(budget):
                x = rng.uniform(lb, ub, size=dim)
                val = func(x)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # Initialize population
        pop = rng.uniform(lb, ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        for i in range(NP):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        if evals >= budget:
            idx = np.argmin(fitness)
            return fitness[idx], pop[idx].copy()

        # JADE parameters
        muF = 0.5
        muCR = 0.5
        c = 0.1  # adaptation rate
        p = 0.1  # proportion for pbest
        stagnation_limit = 20
        stagnation_counter = 0

        # Archive of inferior solutions (same size as pop)
        archive = []
        archive_max = NP

        # For adaptation: success memories
        FS = []
        CRS = []

        # Main loop
        generation = 0
        while evals < budget:
            generation += 1
            # Sort indices by fitness for pbest selection
            sorted_idx = np.argsort(fitness)
            # Number of pbest individuals
            NP_pbest = max(1, int(p * NP))

            FS.clear()
            CRS.clear()

            for i in range(NP):
                if evals >= budget:
                    break

                # Generate F_i
                F = rng.standard_cauchy() * 0.1 + muF
                F = np.clip(F, 1e-10, 2.0)

                # Generate CR_i
                CR = rng.normal(muCR, 0.1)
                CR = np.clip(CR, 0.0, 1.0)

                # Select pbest
                pbest_idx = rng.choice(sorted_idx[:NP_pbest])
                x_pbest = pop[pbest_idx]

                # Select r1 != i
                candidates = list(range(NP))
                candidates.remove(i)
                r1 = rng.choice(candidates)

                # Select r2 from pop union archive, distinct from i and r1
                # Build pool: pop (excluding i and r1) + archive
                pool_indices = [j for j in range(NP) if j != i and j != r1]
                pool = [pop[j] for j in pool_indices] + archive
                if len(pool) == 0:
                    continue
                # Randomly select one from pool
                r2_idx = rng.randint(len(pool))
                r2 = pool[r2_idx]

                # Mutation
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (pop[r1] - r2)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = func(trial)
                evals += 1

                if trial_fit < fitness[i]:
                    # Successful update
                    FS.append(F)
                    CRS.append(CR)
                    # Store replaced individual to archive
                    if len(archive) < archive_max:
                        archive.append(pop[i].copy())
                    else:
                        # Replace random entry
                        idx = rng.randint(archive_max)
                        archive[idx] = pop[i].copy()

                    fitness[i] = trial_fit
                    pop[i] = trial

                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

            # Update muF and muCR if any success
            if len(FS) > 0:
                # Lehmer mean for F
                sum_F = np.sum(FS)
                sum_F_sq = np.sum(np.square(FS))
                muF = (1 - c) * muF + c * (sum_F_sq / sum_F) if sum_F != 0 else muF
                # Arithmetic mean for CR
                muCR = (1 - c) * muCR + c * np.mean(CRS)

            # Check stagnation (accumulated over individuals, approximate)
            if stagnation_counter >= stagnation_limit * NP:
                # Restart: keep best, reinitialize others randomly
                pop[0] = best_x.copy()
                fitness[0] = best_val
                for k in range(1, NP):
                    if evals >= budget:
                        break
                    pop[k] = rng.uniform(lb, ub, size=dim)
                    fitness[k] = func(pop[k])
                    evals += 1
                    if fitness[k] < best_val:
                        best_val = fitness[k]
                        best_x = pop[k].copy()
                        report_best(best_val, best_x)
                archive.clear()
                muF = 0.5
                muCR = 0.5
                stagnation_counter = 0

        idx_best = np.argmin(fitness)
        return fitness[idx_best], pop[idx_best].copy()