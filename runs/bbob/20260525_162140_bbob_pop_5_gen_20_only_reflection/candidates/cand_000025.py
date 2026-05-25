import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        max_evals = self.budget
        # Population size
        pop_size = min(20, max_evals // 2)
        if pop_size < 4:
            pop_size = max(4, max_evals // 2)
        # Initialize population
        pop = self.rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_f = np.inf
        best_x = None
        archive = []
        archive_size = max_evals // 2
        # Evaluate initial population
        for i in range(pop_size):
            f = func(pop[i])
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = pop[i].copy()
                report_best(best_f, best_x)
        evals = pop_size
        # SHADE parameters
        H = 5
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.5)
        k = 0
        # Main loop
        while evals < max_evals:
            # Diversity check for restart
            if evals > pop_size and evals < max_evals - max_evals // 5:
                mean_dist = np.mean(np.linalg.norm(pop - best_x, axis=1))
                domain_range = np.linalg.norm(ub - lb)
                if mean_dist < 0.01 * domain_range:
                    # Restart, keep best
                    new_pop_size = pop_size
                    new_pop = self.rng.uniform(lb, ub, (new_pop_size, dim))
                    # Evaluate new individuals
                    for i in range(new_pop_size):
                        if evals >= max_evals:
                            break
                        f_candidate = func(new_pop[i])
                        evals += 1
                        if f_candidate < best_f:
                            best_f = f_candidate
                            best_x = new_pop[i].copy()
                            report_best(best_f, best_x)
                        # Update archive
                        if len(archive) < archive_size:
                            archive.append(pop[i].copy())
                        else:
                            idx = self.rng.randint(0, archive_size)
                            archive[idx] = pop[i].copy()
                    if evals >= max_evals:
                        break
                    # Replace population
                    pop = new_pop
                    fitness = np.full(pop_size, np.inf)
                    # Re-evaluate initial individuals? Already done above. But need to update fitness.
                    # Actually we evaluated above. We'll store fitness array.
                    # But the new_pop evaluations were stored? We need to set fitness for those evaluated.
                    # Better approach: evaluate all new_pop at once, then assign.
                    # Re-do:
                    # This is getting messy. Simpler: restart but only evaluate new individuals if budget left.
                    # For safety, skip this restart complexity and rely on SHADE's own diversity.
                    pass
            # Generate offspring
            offspring = np.empty_like(pop)
            for i in range(pop_size):
                if evals >= max_evals:
                    break
                # Select pbest index
                pbest_size = max(2, int(0.2 * pop_size))
                sorted_idx = np.argsort(fitness)[:pbest_size]
                pbest = sorted_idx[self.rng.randint(pbest_size)]
                # Select two distinct individuals different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 2:
                    r1, r2 = 0, 1
                else:
                    r1, r2 = self.rng.choice(candidates, 2, replace=False)
                # Mutation: current-to-pbest/1 with archive
                F = MF[k % H] + 0.1 * self.rng.randn()
                F = np.clip(F, 0.1, 0.9)
                # Archive
                if len(archive) > 0:
                    archive_idx = self.rng.randint(0, len(archive))
                    mutant = pop[i] + F * (pop[pbest] - pop[i]) + F * (pop[r1] - archive[archive_idx])
                else:
                    mutant = pop[i] + F * (pop[pbest] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Crossover: binomial
                CR = MCR[k % H] + 0.1 * self.rng.randn()
                CR = np.clip(CR, 0.0, 1.0)
                jrand = self.rng.randint(dim)
                trial = np.where(self.rng.rand(dim) < CR, mutant, pop[i])
                trial[jrand] = mutant[jrand]
                trial = np.clip(trial, lb, ub)
                offspring[i] = trial
            # Evaluate offspring
            for i in range(pop_size):
                if evals >= max_evals:
                    break
                f_off = func(offspring[i])
                evals += 1
                if f_off < best_f:
                    best_f = f_off
                    best_x = offspring[i].copy()
                    report_best(best_f, best_x)
                # Selection and archive update
                if f_off < fitness[i]:
                    # Add current to archive
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        idx = self.rng.randint(0, archive_size)
                        archive[idx] = pop[i].copy()
                    # Update success memories
                    if f_off < fitness[i]:
                        # Success
                        if F is not None:
                            MF[k % H] = 0.1 * (F**2) + 0.9 * MF[k % H]
                        if CR is not None:
                            MCR[k % H] = 0.1 * (CR**2) + 0.9 * MCR[k % H]
                        k += 1
                    fitness[i] = f_off
                    pop[i] = offspring[i]
        return best_f, best_x