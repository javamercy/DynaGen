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

        # Initial population size
        pop_size_start = max(4 * dim, 5)
        pop_size_end = max(3, dim)
        pop_size = pop_size_start

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

        # JADE parameters
        mu_F = 0.5
        mu_CR = 0.5
        archive = []
        prev_best_val = best_val
        gen_no_improve = 0
        diversity_threshold = 0.01 * np.mean(ub - lb)

        while evals < budget:
            progress = evals / budget
            pop_size = int(pop_size_start + (pop_size_end - pop_size_start) * progress)
            pop_size = max(3, pop_size)
            pbest_ratio = 0.2 - 0.15 * progress
            num_pbest = max(2, int(pbest_ratio * pop_size))
            scale_F = 0.2 - 0.15 * progress
            scale_CR = 0.2 - 0.15 * progress
            archive_size = pop_size

            sort_idx = np.argsort(fitness)
            pbest_pool = sort_idx[:num_pbest]

            successful_F = []
            successful_CR = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                # Generate F_i
                F_i = mu_F + scale_F * rng.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + scale_F * rng.standard_cauchy()
                F_i = min(F_i, 1.0)

                # Generate CR_i
                CR_i = mu_CR + scale_CR * rng.randn()
                CR_i = np.clip(CR_i, 0, 1)

                # Select pbest (different from i)
                cand = [idx for idx in pbest_pool if idx != i]
                if not cand:
                    cand = pbest_pool
                pbest_idx = rng.choice(cand)

                # Select r1 from population (exclude i and pbest)
                candidates_r1 = [j for j in range(pop_size) if j not in (i, pbest_idx)]
                if len(candidates_r1) == 0:
                    continue
                r1 = rng.choice(candidates_r1)

                # Select r2 from population and archive
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

                # Mutation: current-to-pbest/1 (or rand/1 when diversity low)
                pop_center = np.mean(pop, axis=0)
                distances = np.mean(np.sqrt(np.sum((pop - pop_center)**2, axis=1)))
                use_rand1 = (distances < diversity_threshold) and (rng.rand() < 0.5)
                if use_rand1:
                    idxs = rng.choice(pop_size, 3, replace=False)
                    mutant = pop[idxs[0]] + F_i * (pop[idxs[1]] - pop[idxs[2]])
                else:
                    mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (pop[r1] - r2)
                mutant = np.clip(mutant, lb, ub)

                # Binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR_i or j == j_rand:
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

            # Stagnation and diversity detection
            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            remaining_evals = budget - evals
            if pop_size > 1:
                pop_center = np.mean(pop, axis=0)
                distances = np.mean(np.sqrt(np.sum((pop - pop_center)**2, axis=1)))
            else:
                distances = 0.0

            restart = False
            if remaining_evals > 0:
                stagnation_threshold = max(1, int(0.1 * remaining_evals / pop_size))
                if gen_no_improve >= stagnation_threshold:
                    restart = True
                if distances < diversity_threshold:
                    restart = True

            if restart and evals < budget:
                # Local refinement: pattern search cycling through coordinates
                local_budget = int(0.05 * remaining_evals)
                if local_budget > 0:
                    step_size = 0.01 * (ub - lb)
                    coord_cycle = 0
                    # Ensure we cover all coordinates systematically
                    while local_budget > 0:
                        # Try positive step
                        step = np.zeros(dim)
                        step[coord_cycle] = step_size[coord_cycle]
                        trial = np.clip(best_x + step, lb, ub)
                        val = func(trial)
                        evals += 1
                        local_budget -= 1
                        if val < best_val:
                            best_val = val
                            best_x = trial.copy()
                            report_best(best_val, best_x)
                            step_size[coord_cycle] *= 1.2
                            step_size = np.clip(step_size, 1e-10, 0.1 * (ub - lb))
                            coord_cycle = (coord_cycle + 1) % dim
                            continue
                        # If positive fails, try negative step
                        if local_budget <= 0:
                            step_size[coord_cycle] *= 0.5
                            step_size = np.clip(step_size, 1e-10, 0.1 * (ub - lb))
                            coord_cycle = (coord_cycle + 1) % dim
                            break
                        step = np.zeros(dim)
                        step[coord_cycle] = -step_size[coord_cycle]
                        trial = np.clip(best_x + step, lb, ub)
                        val = func(trial)
                        evals += 1
                        local_budget -= 1
                        if val < best_val:
                            best_val = val
                            best_x = trial.copy()
                            report_best(best_val, best_x)
                            step_size[coord_cycle] *= 1.2
                        else:
                            step_size[coord_cycle] *= 0.5
                        step_size = np.clip(step_size, 1e-10, 0.1 * (ub - lb))
                        coord_cycle = (coord_cycle + 1) % dim

                # Restart population
                pop_std = np.std(pop, axis=0)
                pop_std = np.maximum(pop_std, 1e-12)
                new_pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
                new_pop[0] = best_x
                for i in range(1, pop_size):
                    if rng.rand() < 0.5:
                        noise = rng.normal(0, pop_std, size=dim)
                        new_pop[i] = np.clip(new_pop[i] + noise, lb, ub)
                    else:
                        radius = 0.1 * (ub - lb)
                        new_pop[i] = np.clip(best_x + rng.uniform(-radius, radius), lb, ub)
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