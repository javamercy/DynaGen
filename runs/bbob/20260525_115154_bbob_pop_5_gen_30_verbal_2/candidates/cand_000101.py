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
        # initial population size
        pop_size_start = max(4 * dim, 5)
        pop_size_end = max(3, dim)
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
        # memorize for restart
        initial_best_val = best_val
        initial_best_x = best_x.copy() if best_x is not None else None
        # JADE parameters
        mu_F = 0.5
        mu_CR = 0.5
        archive = []
        gen_no_improve = 0
        # scaling factors for coordinate-wise mutation (standard deviation of top half)
        scale = np.ones(dim)
        while evals < budget:
            progress = evals / budget
            pop_size = int(pop_size_start + (pop_size_end - pop_size_start) * progress)
            pop_size = max(3, pop_size)
            pbest_ratio = 0.2 - 0.15 * progress
            num_pbest = max(2, int(pbest_ratio * pop_size))
            archive_size = pop_size
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
                # sample F from Cauchy
                F_i = mu_F + 0.1 * rng.standard_cauchy()
                while F_i <= 0:
                    F_i = mu_F + 0.1 * rng.standard_cauchy()
                F_i = min(F_i, 1.0)
                # sample CR from normal
                CR_i = mu_CR + 0.1 * rng.randn()
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
                # select r2 (from population or archive)
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
                # evaluation
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
            # update success-history
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
            # restart condition
            remaining_evals = budget - evals
            if remaining_evals > 0:
                threshold_gen = max(1, int(0.1 * remaining_evals / pop_size))
                if gen_no_improve >= threshold_gen and evals < budget:
                    # restart around best with small uniform perturbations
                    new_pop = np.empty((pop_size, dim))
                    new_pop[0] = best_x
                    for i in range(1, pop_size):
                        candidate = best_x + rng.uniform(-0.05, 0.05, size=dim) * (ub - lb)
                        candidate = np.clip(candidate, lb, ub)
                        new_pop[i] = candidate
                    pop = new_pop
                    fitness = np.full(pop_size, np.inf)
                    # evaluate new population (skip first if already evaluated? to be safe, re-evaluate)
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
                    # reset adaptation
                    mu_F = 0.5
                    mu_CR = 0.5
                    archive = []
                    initial_best_val = best_val
                    initial_best_x = best_x.copy()
                    gen_no_improve = 0
        return best_val, best_x