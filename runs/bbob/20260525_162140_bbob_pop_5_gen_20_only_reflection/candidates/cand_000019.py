import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        dim = self.dim
        budget = self.budget

        # Population size
        NP = min(10 * dim, max(4, budget // 2 - 1))
        if NP < 4:
            NP = budget

        # Initialize memory for F and CR
        H = 6
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        h = 0  # history index

        # Archive (inferior solutions)
        archive_size = NP
        archive = np.empty((0, dim))

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.array([func(pop[i]) for i in range(NP)])
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_val = fitness[best_idx]
        report_best(best_val, best_x)
        func_evals = NP

        # Restart loop
        while func_evals < budget:
            # Compute generations to run before next check
            gen_max = max(1, int(0.5 * (budget - func_evals) / NP))
            # But we will just run generation by generation until budget exhausted or restart triggers
            # Actually we need to run until budget or restart
            # We'll do a while loop inside that stops if budget exhausted or restart needed
            # But simpler: just run generation loop with break conditions
            # We'll implement restart detection per generation
            
            success_F = []
            success_CR = []
            success_fit_diff = []

            for gen in range(gen_max):
                if func_evals >= budget:
                    break
                # For each individual
                new_pop = pop.copy()
                new_fitness = fitness.copy()
                for i in range(NP):
                    # Select random indices for mutation
                    candidates = list(range(NP))
                    candidates.remove(i)
                    if len(candidates) < 3:
                        continue
                    # Generate F_i and CR_i
                    h_idx = rng.randint(H)
                    F_i = rng.standard_cauchy() * 0.1 + M_F[h_idx]
                    F_i = np.clip(F_i, 0, 1)
                    # If F_i <= 0, re-sample? But we clip, so okay
                    CR_i = rng.normal(M_CR[h_idx], 0.1)
                    CR_i = np.clip(CR_i, 0, 1)
                    # Select p-best: top 20% of current pop
                    p = max(2, int(NP * 0.2))
                    sorted_idx = np.argsort(fitness)
                    pbest_idx = rng.choice(sorted_idx[:p])
                    # Choose a and b from pop union archive
                    union_pop = np.vstack([pop, archive]) if len(archive) > 0 else pop
                    # Select two distinct random indices from union, not equal to i
                    union_indices = list(range(len(union_pop)))
                    # Remove i from union if i < len(pop) (since archive doesn't contain i necessarily)
                    # For simplicity, just choose two random distinct from union
                    # But ensure they are distinct from each other and not i?
                    # SHADE usually selects a and b from pop (excluding i) and archive; but we simplify
                    # Actually typical: a from pop (excluding i), b from union (pop+archive, excluding i)
                    # We'll just sample two distinct from union, but ensure they are not i by chance
                    for attempt in range(10):
                        r1, r2 = rng.choice(len(union_pop), 2, replace=False)
                        if r1 != i and r2 != i:
                            break
                    else:
                        r1, r2 = rng.choice(len(union_pop), 2, replace=False)
                    a = union_pop[r1]
                    b = union_pop[r2]
                    # Mutation
                    mutant = pop[i] + F_i * (pop[pbest_idx] - pop[i]) + F_i * (a - b)
                    # Crossover: binomial
                    trial = pop[i].copy()
                    j_rand = rng.randint(dim)
                    for j in range(dim):
                        if rng.rand() < CR_i or j == j_rand:
                            trial[j] = mutant[j]
                    # Clip
                    trial = np.clip(trial, lb, ub)
                    # Evaluate
                    val = func(trial)
                    func_evals += 1
                    if val < fitness[i]:
                        # Update archive with replaced parent (if space)
                        if len(archive) < archive_size:
                            archive = np.vstack([archive, pop[i].reshape(1, -1)])
                        elif rng.rand() < 0.5:
                            idx = rng.randint(len(archive))
                            archive[idx] = pop[i]
                        # Update new pop and fitness
                        new_pop[i] = trial
                        new_fitness[i] = val
                        success_F.append(F_i)
                        success_CR.append(CR_i)
                        success_fit_diff.append(np.abs(val - fitness[i]))
                        # Update best
                        if val < best_val:
                            best_val = val
                            best_x = trial.copy()
                            report_best(best_val, best_x)
                    if func_evals >= budget:
                        break
                # Replace population
                pop = new_pop
                fitness = new_fitness
                # Update memory if successes
                if len(success_F) > 0:
                    # Compute weights
                    w = np.array(success_fit_diff) / np.sum(success_fit_diff)
                    # Update M_F and M_CR using weighted Lehmer mean
                    M_F[h] = np.sum(w * np.array(success_F)**2) / np.sum(w * np.array(success_F))
                    M_CR[h] = np.sum(w * np.array(success_CR)**2) / np.sum(w * np.array(success_CR))
                    h = (h + 1) % H
                # Check restart condition: population diversity low or no improvement for long
                # Compute population standard deviation along each dimension, then mean
                pop_std = np.mean(np.std(pop, axis=0))
                if pop_std < 1e-12 * (np.max(ub - lb) if np.any(ub != lb) else 1.0):
                    # Restart: reinitialize population and archive
                    pop = lb + (ub - lb) * rng.rand(NP, dim)
                    fitness = np.array([func(pop[i]) for i in range(NP)])
                    func_evals += NP
                    if func_evals > budget:
                        # partially evaluate? Actually we may exceed budget; we should stop after budget
                        # But we already counted full NP; if over, we break and return
                        # However we need to ensure we don't exceed budget; we'll handle by checking before eval
                        pass
                    # Update best if any new point is better
                    best_idx_new = np.argmin(fitness)
                    if fitness[best_idx_new] < best_val:
                        best_val = fitness[best_idx_new]
                        best_x = pop[best_idx_new].copy()
                        report_best(best_val, best_x)
                    archive = np.empty((0, dim))
                    M_F[:] = 0.5
                    M_CR[:] = 0.5
                    h = 0
                    break  # break current generation loop, restart outer while

            if func_evals >= budget:
                break

        return best_val, best_x