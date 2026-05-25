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

        # Population size schedule
        pop_size_start = max(4 * dim, 5)
        pop_size_end = max(3, dim)

        # Initialize
        best_val = np.inf
        best_x = None
        evals = 0

        # Initial population size
        pop_size = max(pop_size_start, pop_size_end)  # start with max
        pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
        fitness = np.full(pop_size, np.inf)

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
        archive = []
        prev_best_val = best_val
        gen_no_improve = 0

        # Main loop
        while evals < budget:
            # Update scheduled parameters based on progress
            progress = evals / budget
            # Population size: linear decrease
            pop_size = int(pop_size_start + (pop_size_end - pop_size_start) * progress)
            pop_size = max(3, pop_size)
            # pbest ratio: linear decrease from 0.2 to 0.05
            pbest_ratio = 0.2 - 0.15 * progress
            num_pbest = max(2, int(pbest_ratio * pop_size))
            # Adaptive scaling factor for F and CR generation
            scale_F = 0.2 - 0.15 * progress  # from 0.2 to 0.05
            scale_CR = 0.2 - 0.15 * progress
            archive_size = pop_size  # match current pop size

            # Sort for pbest
            sort_idx = np.argsort(fitness)
            pbest_pool = sort_idx[:num_pbest]

            successful_F = []
            successful_CR = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                # Generate F_i using Cauchy with adaptive scale
                F_i = mu_F + scale_F * rng.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + scale_F * rng.standard_cauchy()
                F_i = min(F_i, 1.0)

                # Generate CR_i using normal with adaptive scale
                CR_i = mu_CR + scale_CR * rng.randn()
                CR_i = np.clip(CR_i, 0, 1)

                # Select pbest (distinct from i)
                cand = [idx for idx in pbest_pool if idx != i]
                if not cand:
                    cand = pbest_pool
                pbest_idx = rng.choice(cand)

                # Select r1 from population (exclude i and pbest)
                candidates_r1 = [j for j in range(pop_size) if j not in (i, pbest_idx)]
                if len(candidates_r1) == 0:
                    continue
                r1 = rng.choice(candidates_r1)

                # Select r2 from population (exclude i, pbest, r1) and archive
                candidates_r2 = [j for j in range(pop_size) if j not in (i, pbest_idx, r1)]
                if archive:
                    candidates_r2.extend(archive)
                if len(candidates_r2) == 0:
                    continue
                pick = rng.randint(len(candidates_r2))
                if isinstance(candidates_r2[pick], int):
                    r2 = pop[candidates_r2[pick]]
                else:
                    r2 = candidates_r2[pick]

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - r2)
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
                    # Add parent to archive
                    archive.append(pop[i].copy())
                    if len(archive) > archive_size:
                        archive.pop(rng.randint(len(archive)))

                    fitness[i] = trial_fit
                    pop[i] = trial
                    successful_F.append(F_i)
                    successful_CR.append(CR_i)
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Update parameter means
            if len(successful_F) > 0:
                sum_F = np.sum(successful_F)
                sum_F2 = np.sum(np.square(successful_F))
                if sum_F > 0:
                    mu_F = sum_F2 / sum_F
                mu_CR = np.mean(successful_CR)

            # Stagnation detection based on remaining budget
            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            remaining_evals = budget - evals
            if remaining_evals > 0:
                threshold_gen = max(1, int(0.1 * remaining_evals / pop_size))
                if gen_no_improve >= threshold_gen and evals < budget:
                    # Compute population spread
                    pop_std = np.std(pop, axis=0)
                    pop_std = np.maximum(pop_std, 1e-12)

                    # Restart: keep best, reinitialize others with scaled perturbation
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

                    # Reset parameters
                    mu_F = 0.5
                    mu_CR = 0.5
                    archive = []
                    prev_best_val = best_val
                    gen_no_improve = 0

        return best_val, best_x