import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Population size
        NP = min(10 * dim, budget // 2)
        NP = max(4, NP)
        if NP > budget:
            NP = budget

        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.array([func(pop[i]) for i in range(NP)])
        evals = NP

        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        best_x = pop[best_idx].copy()
        report_best(best_val, best_x)

        if NP == 1 or evals >= budget:
            return best_val, best_x

        # SHADE memory
        H = 5
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.9)
        p = 0

        # Archive
        archive = []
        archive_max = NP

        # Restart control
        gen_since_restart = 0
        max_gen_without_restart = 8
        diversity_threshold = 1e-6

        gen = 0
        while evals < budget:
            # Check restart condition
            if gen > 0:
                std_dim = np.std(pop, axis=0)
                diversity = np.mean(std_dim) / (np.mean(ub - lb) + 1e-30)
                if diversity < diversity_threshold or gen_since_restart >= max_gen_without_restart:
                    # Restart
                    remaining = budget - evals
                    needed = NP - 1
                    if remaining >= needed:
                        pop[0] = best_x.copy()
                        new_pop = lb + (ub - lb) * rng.rand(NP - 1, dim)
                        for i in range(1, NP):
                            pop[i] = new_pop[i-1]
                        for i in range(1, NP):
                            f = func(pop[i])
                            evals += 1
                            if f < best_val:
                                best_val = f
                                best_x = pop[i].copy()
                                report_best(best_val, best_x)
                        M_F[:] = 0.5
                        M_CR[:] = 0.9
                        p = 0
                        archive = []
                        gen_since_restart = 0
                    # else: cannot restart, continue without

            # Evolution
            S_F = []
            S_CR = []
            delta_f = []
            pop_archive = pop.tolist() + archive
            pop_archive = np.array(pop_archive) if len(pop_archive) > 0 else pop

            for i in range(NP):
                if evals >= budget:
                    break
                r = rng.randint(H)
                F = M_F[r] + 0.1 * rng.standard_cauchy()
                if F <= 0:
                    F = 0.1
                elif F > 1:
                    F = 1.0
                CR = M_CR[r] + 0.1 * rng.randn()
                CR = np.clip(CR, 0, 1)

                # current-to-pbest/1
                p_best = max(2, int(0.2 * NP))
                sorted_idx = np.argsort(fitness)
                pbest_idx = rng.choice(sorted_idx[:p_best])
                x_pbest = pop[pbest_idx]
                # choose two distinct random individuals from pop_archive, excluding i
                candidates = list(range(len(pop_archive)))
                # Exclude current index if it's in pop_archive? But pop[i] is in pop, which is part of pop_archive.
                # We'll simply not exclude any, but ensure distinct from each other and from x_pbest? Not strictly needed.
                if len(candidates) < 2:
                    continue
                idx = rng.choice(candidates, 2, replace=False)
                a = pop_archive[idx[0]]
                b = pop_archive[idx[1]]
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (a - b)

                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    delta_f.append(fitness[i] - f_trial)
                    S_F.append(F)
                    S_CR.append(CR)
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(rng.randint(len(archive)))
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_val:
                        best_val = f_trial
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if len(S_F) > 0:
                w = np.array(delta_f) / (np.sum(delta_f) + 1e-30)
                M_F[p] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
                M_CR[p] = np.sum(w * np.array(S_CR)**2) / (np.sum(w * np.array(S_CR)) + 1e-30)
                p = (p + 1) % H

            gen += 1
            gen_since_restart += 1

        return best_val, best_x