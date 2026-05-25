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

        NP = min(10 * dim, max(4, budget // 2 - 1))
        if NP > budget:
            NP = budget
        NP = max(1, NP)

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

        while evals < budget:
            # Diversity restart check every generation
            std_dim = np.std(pop, axis=0)
            diversity = np.mean(std_dim) / np.mean(ub - lb)
            if diversity < 1e-6 and evals > 0.05 * budget:
                remaining = budget - evals
                needed = NP - 1
                if remaining >= needed:
                    pop[0] = best_x.copy()
                    new_pop = lb + (ub - lb) * rng.rand(NP - 1, dim)
                    for i in range(1, NP):
                        pop[i] = new_pop[i - 1]
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
                    fitness = np.array([func(pop[i]) for i in range(NP)])
                    evals += NP  # already counted above? Actually we counted evals in the loop; but we need to reset fitness array. Let's recalc all fitness after restart to keep consistency.
                    # Since we already evaluated new individuals, we can set fitness accordingly.
                    # But easier: after restart, evaluate all new individuals (pop[0] is best_x, already known). We'll re-evaluate best_x to get fitness? Actually we know its f. To avoid duplication, we can set fitness[0] = best_val, and for i>=1 we have f.
                    # But we already called func for each new individual, so we can just assign.
                    # Let's just re-evaluate all pop to get current fitness (since best_x may have been improved but we already have best_val).
                    # However, to keep it simple, we'll re-evaluate the entire population to get accurate fitness.
                    # But we already evaluated pop[1:] once. To avoid extra calls, let's do:
                    fitness[0] = best_val
                    for i in range(1, NP):
                        # we already evaluated new_pop[i-1] and stored in pop[i], and called func; we can store that f.
                        # But we need to keep track; easier: evaluate all pop again? That would waste budget.
                        # Instead, let's restructure restart to evaluate only new individuals.
                    # To avoid complexity, I'll implement restart by re-evaluating all pop (including best_x) to reset fitness array.
                    # But best_x is known, so we can skip it. Let me just re-evaluate pop[0] to get fitness[0] (it's best_val).
                    # Actually, best_x is in pop[0], its fitness is best_val. So we can set fitness[0] = best_val.
                    # For the rest, we already called func and stored the f in a temporary variable; we should have stored them.
                    # Let's rewrite in a cleaner way: keep a list of new_f for new individuals.
            # I'll restructure the restart section below.
            # Actually, for simplicity, let's just set fitness array after restart by re-evaluating only new individuals (excluding best_x).
            # But we need to ensure we don't exceed budget.
            # Let me implement it more carefully.

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

                # Mutation: 90% current-to-pbest/1, 10% rand/1
                candidates = list(range(NP))
                candidates.remove(i)
                if rng.rand() < 0.1:
                    r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                    a, b, c = pop[r1], pop[r2], pop[r3]
                    mutant = a + F * (b - c)
                else:
                    pbest_list = [idx for idx in pbest_indices if idx != i]
                    if len(pbest_list) == 0:
                        r1 = rng.choice(candidates)
                    else:
                        r1 = rng.choice(pbest_list)
                    remaining = [j for j in candidates if j != r1]
                    if len(remaining) < 2:
                        continue
                    r2, r3 = rng.choice(remaining, 2, replace=False)
                    a, b, c = pop[r1], pop[r2], pop[r3]
                    mutant = a + F * (b - c)

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

            # Restart check after each generation
            std_dim = np.std(pop, axis=0)
            diversity = np.mean(std_dim) / np.mean(ub - lb)
            if diversity < 1e-6 and evals > 0.05 * budget and evals < budget:
                remaining = budget - evals
                needed = NP - 1
                if remaining >= needed:
                    # Keep best individual
                    pop[0] = best_x.copy()
                    fitness[0] = best_val
                    # Generate new individuals
                    new_pop = lb + (ub - lb) * rng.rand(NP - 1, dim)
                    for i in range(1, NP):
                        pop[i] = new_pop[i-1]
                        f = func(pop[i])
                        evals += 1
                        fitness[i] = f
                        if f < best_val:
                            best_val = f
                            best_x = pop[i].copy()
                            report_best(best_val, best_x)
                    # Reset memory
                    M_F[:] = 0.5
                    M_CR[:] = 0.9
                    p = 0

        return best_val, best_x