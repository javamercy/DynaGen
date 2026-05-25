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

        NP_init = min(10 * dim, max(4, budget // 2 - 1))
        NP_min = 4
        NP = NP_init
        archive_max = NP_init
        archive = []

        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.array([func(pop[i]) for i in range(NP)])
        evals = NP

        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        best_x = pop[best_idx].copy()
        report_best(best_val, best_x)

        if NP == 1:
            return best_val, best_x

        H = 7
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.9)
        p = 0

        max_gen = (budget - evals) // NP
        gen = 0
        stall_count = 0
        best_val_prev = best_val

        while evals < budget and gen < max_gen + 1:
            # Linear population size reduction
            if gen > 0 and evals > 0.05 * budget:
                target_NP = max(NP_min, int(round(NP_init - (NP_init - NP_min) * (evals / budget))))
                if target_NP < NP:
                    sorted_idx = np.argsort(fitness)
                    keep = sorted_idx[:target_NP]
                    pop = pop[keep]
                    fitness = fitness[keep]
                    NP = target_NP
                    if len(archive) > archive_max:
                        archive = archive[-archive_max:]

            # Diversity check and restart
            if gen > 0 and evals > 0.1 * budget:
                std_dim = np.std(pop, axis=0)
                diversity = np.mean(std_dim) / np.mean(ub - lb)
                if diversity < 1e-5 and stall_count >= 10:
                    # Local search
                    ls_budget = min(3 * dim, budget - evals)
                    for _ in range(ls_budget):
                        step = 0.01 * (ub - lb) * rng.randn(dim)
                        candidate = best_x + step
                        candidate = np.clip(candidate, lb, ub)
                        f = func(candidate)
                        evals += 1
                        if f < best_val:
                            best_val = f
                            best_x = candidate.copy()
                            report_best(best_val, best_x)
                    # Restart
                    if evals < budget:
                        new_pop = [best_x.copy()]
                        remaining = min(NP - 1, budget - evals)
                        if remaining > 0:
                            for _ in range(remaining):
                                new_pop.append(lb + (ub - lb) * rng.rand(dim))
                        else:
                            new_pop = [best_x.copy()] * NP
                        pop = np.array(new_pop[:NP])
                        for i in range(1, NP):
                            if evals >= budget:
                                break
                            fitness[i] = func(pop[i])
                            evals += 1
                            if fitness[i] < best_val:
                                best_val = fitness[i]
                                best_x = pop[i].copy()
                                report_best(best_val, best_x)
                        M_F[:] = 0.5
                        M_CR[:] = 0.9
                        p = 0
                    stall_count = 0

            # Mutation, crossover, selection with archive
            S_F = []
            S_CR = []
            delta_f = []
            pbest_size = max(2, int(0.2 * NP))
            sorted_indices = np.argsort(fitness)
            pbest_indices = set(sorted_indices[:pbest_size])

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

                # Select pbest
                pbest_list = [idx for idx in pbest_indices if idx != i]
                if len(pbest_list) == 0:
                    r1 = rng.randint(NP)
                    while r1 == i:
                        r1 = rng.randint(NP)
                else:
                    r1 = rng.choice(pbest_list)

                # Select r2 from pop (≠ i, ≠ r1)
                candidates = list(range(NP))
                candidates.remove(i)
                if r1 in candidates:
                    candidates.remove(r1)
                if len(candidates) < 2:
                    continue
                r2 = rng.choice(candidates)

                # Select r3 from pop∪archive (≠ i, ≠ r1, ≠ r2)
                union_list = list(range(NP)) + [('a', j) for j in range(len(archive))]
                # Filter out i, r1, r2
                valid_union = []
                for item in union_list:
                    if isinstance(item, int):
                        if item != i and item != r1 and item != r2:
                            valid_union.append(item)
                    else:
                        valid_union.append(item)
                if len(valid_union) == 0:
                    continue
                chosen = rng.choice(valid_union)
                if isinstance(chosen, int):
                    r3_point = pop[chosen]
                else:
                    r3_point = archive[chosen[1]]

                # Mutation: current-to-pbest/1 with archive
                a = pop[r1]
                b = pop[r2]
                c = r3_point
                mutant = a + F * (b - c)

                # Crossover
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
                    # Add replaced individual to archive
                    if len(archive) < archive_max:
                        archive.append(pop[i].copy())
                    else:
                        archive[rng.randint(archive_max)] = pop[i].copy()
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_val:
                        best_val = f_trial
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if len(S_F) > 0:
                w = np.array(delta_f) / (np.sum(delta_f) + 1e-30)
                M_F[p] = np.sum(w * np.array(S_F) ** 2) / (np.sum(w * np.array(S_F)) + 1e-30)
                M_CR[p] = np.sum(w * np.array(S_CR) ** 2) / (np.sum(w * np.array(S_CR)) + 1e-30)
                p = (p + 1) % H

            # Update stall count
            if best_val < best_val_prev:
                best_val_prev = best_val
                stall_count = 0
            else:
                stall_count += 1

            gen += 1

        return best_val, best_x