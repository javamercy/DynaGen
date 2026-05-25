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
        # initial population size (larger for exploration)
        pop_size_start = max(6 * dim, 10)
        pop_size_end = max(4, dim)  # but we will keep it larger than in parent
        pop_size = pop_size_start
        # initialize population uniformly
        pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
        # initial evaluations
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
        # memorize for restart (not used directly but for improvement tracking)
        initial_best_val = best_val
        initial_best_x = best_x.copy() if best_x is not None else None
        # JADE parameters with exploration bias
        mu_F = 0.7  # slightly higher mean
        mu_CR = 0.9  # high crossover probability
        archive = []
        gen_no_improve = 0
        # scaling factors for coordinate-wise mutation (standard deviation of top half)
        scale = np.ones(dim)
        while evals < budget:
            progress = evals / budget
            # keep population size larger: linear reduction from start to end but not too small
            pop_size = int(pop_size_start + (pop_size_end - pop_size_start) * progress)
            pop_size = max(5, pop_size)  # ensure at least 5
            pbest_ratio = 0.2 - 0.1 * progress  # slightly lower pbest ratio to use more diverse parents
            num_pbest = max(2, int(pbest_ratio * pop_size))
            archive_size = pop_size * 2  # larger archive
            # sort fitness
            sort_idx = np.argsort(fitness)[:pop_size]
            pbest_pool = sort_idx[:num_pbest]
            # update coordinate-wise scaling from top half
            half = pop_size // 2
            top_idx = sort_idx[:half]
            if len(top_idx) >= 2:
                scale = np.std(pop[top_idx], axis=0) + 1e-10
            else:
                scale = np.ones(dim)
            successful_F = []
            successful_CR = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                # sample F from Cauchy with larger scale
                F_i = mu_F + 0.2 * rng.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + 0.2 * rng.standard_cauchy()
                F_i = min(F_i, 1.0)
                # sample CR from Beta distribution favoring high values, or use normal with high mean
                # Using a normal with mean 0.9 and std 0.2 truncated to [0,1]
                CR_i = mu_CR + 0.2 * rng.randn()
                CR_i = np.clip(CR_i, 0, 1)
                # select pbest
                cand = [idx for idx in pbest_pool if idx != i]
                if not cand:
                    cand = pbest_pool
                pbest_idx = rng.choice(cand)
                # select r1
                candidates_r1 = [j for j in range(pop_size) if j not in (i, pbest_idx)]
                if len(candidates_r1) == 0:
                    continue
                r1 = rng.choice(candidates_r1)
                # select r2 from population or archive
                candidates_r2 = [j for j in range(pop_size) if j not in (i, pbest_idx, r1)]
                if archive:
                    candidates_r2.extend(archive)
                if len(candidates_r2) == 0:
                    continue
                pick = rng.randint(len(candidates_r2))
                r2 = candidates_r2[pick]
                if isinstance(r2, int):
                    diff = (pop[r1] - pop[r2]) * scale
                else:
                    diff = (pop[r1] - r2) * scale
                # mutation
                mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * diff
                mutant = np.clip(mutant, lb, ub)
                # crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]
                # with a small probability, replace trial with a random point to increase exploration
                if rng.rand() < 0.1:  # 10% chance
                    trial = rng.uniform(lb, ub, size=dim)
                # evaluation
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    archive.append(pop[i].copy())
                    if len(archive) > archive_size:
                        # remove random element from archive
                        archive.pop(rng.randint(len(archive)))
                    fitness[i] = trial_fit
                    pop[i] = trial
                    successful_F.append(F_i)
                    successful_CR.append(CR_i)
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            # update success-history (Lehmer mean for F)
            if len(successful_F) > 0:
                sum_F = np.sum(successful_F)
                sum_F2 = np.sum(np.square(successful_F))
                if sum_F > 0:
                    mu_F = sum_F2 / sum_F
                mu_CR = np.mean(successful_CR)
            # check improvement
            if best_val < initial_best_val:
                gen_no_improve = 0
                initial_best_val = best_val
                initial_best_x = best_x.copy()
            else:
                gen_no_improve += 1
            # restart condition: more generous to explore
            remaining_evals = budget - evals
            if remaining_evals > 0:
                threshold_gen = max(1, int(0.05 * remaining_evals / pop_size))  # smaller threshold
                if gen_no_improve >= threshold_gen and evals < budget:
                    # restart: combine best perturbation and random points
                    new_pop = np.empty((pop_size, dim))
                    # keep best point
                    new_pop[0] = best_x
                    # generate half from best perturbation, half random
                    for i in range(1, pop_size // 2):
                        candidate = best_x + rng.uniform(-0.1, 0.1, size=dim) * (ub - lb)
                        candidate = np.clip(candidate, lb, ub)
                        new_pop[i] = candidate
                    for i in range(pop_size // 2, pop_size):
                        candidate = rng.uniform(lb, ub, size=dim)
                        new_pop[i] = candidate
                    pop = new_pop
                    fitness = np.full(pop_size, np.inf)
                    # evaluate new population (skip first already evaluated? but we re-evaluate for simplicity)
                    # evaluate all
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
                    # reset adaptation
                    mu_F = 0.7
                    mu_CR = 0.9
                    archive = []
                    initial_best_val = best_val
                    initial_best_x = best_x.copy()
                    gen_no_improve = 0
        return best_val, best_x