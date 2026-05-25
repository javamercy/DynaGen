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
        budget = self.budget
        rng = self.rng

        # Population size: larger for exploration, capped by budget/2
        pop_size = max(5, min(5 * dim, budget // 2))
        # LHS initialization
        points = np.empty((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            u = rng.rand(pop_size)
            points[:, d] = lb[d] + (perm + u) / pop_size * (ub[d] - lb[d])

        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0

        # Initial evaluation
        for i in range(pop_size):
            if evals >= budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # DE parameters
        CR_base = 0.9
        F_base = 0.5
        CR = CR_base
        F = F_base
        CR_min, CR_max = 0.3, 1.0
        F_min, F_max = 0.1, 1.0
        success_memory = []
        window = 10

        # Diversity parameters
        diversity_threshold = 0.05 * np.mean(ub - lb)
        stagnation_counter = 0
        max_stagnation = max(30, budget // 15)
        gen_count = 0

        # Local search parameters
        sigma = 0.2 * (ub - lb)
        sigma_min = 1e-8 * (ub - lb)
        local_every = 2 * pop_size
        gen_evals = 0

        # Sinusoidal oscillation period (in generations)
        period = max(10, pop_size // 2)

        while evals < budget:
            gen_count += 1
            # Update F and CR with oscillation
            F = 0.5 + 0.5 * np.sin(2 * np.pi * gen_count / period)
            CR = 0.8 + 0.2 * np.sin(2 * np.pi * gen_count / period + 1)
            F = np.clip(F, F_min, F_max)
            CR = np.clip(CR, CR_min, CR_max)

            # Compute diversity (average distance to centroid)
            centroid = np.mean(points, axis=0)
            distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
            div = np.mean(distances)
            low_diversity = div < diversity_threshold

            gen_success = 0
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Mutation: DE/rand/2
                idxs = [j for j in range(pop_size) if j != i]
                if len(idxs) < 4:
                    # fallback: random sampling
                    trial = lb + rng.rand(dim) * (ub - lb)
                    f_trial = func(trial)
                    evals += 1
                    if f_trial < fitness[i]:
                        points[i] = trial
                        fitness[i] = f_trial
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = trial.copy()
                            report_best(best_f, best_x)
                            stagnation_counter = 0
                        else:
                            stagnation_counter += 1
                    continue
                chosen = rng.choice(idxs, 5, replace=False)
                a, b, c, d, e = chosen
                mutant = points[a] + F * (points[b] - points[c]) + F * (points[d] - points[e])
                # Crossover with binomial
                trial = points[i].copy()
                j_rand = rng.randint(dim)
                mask = rng.rand(dim) < CR
                mask[j_rand] = True
                trial[mask] = mutant[mask]
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1
                gen_evals += 1
                if f_trial < fitness[i] or (low_diversity and rng.rand() < 0.05):
                    points[i] = trial
                    fitness[i] = f_trial
                    gen_success += 1
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

                # Local search after certain evaluations
                if gen_evals >= local_every and evals < budget:
                    gen_evals = 0
                    delta = sigma * rng.randn(dim)
                    candidate = best_x + delta
                    candidate = np.clip(candidate, lb, ub)
                    f_candidate = func(candidate)
                    evals += 1
                    if f_candidate < best_f:
                        best_f = f_candidate
                        best_x = candidate.copy()
                        sigma = np.clip(sigma * 1.2, None, ub - lb)
                        report_best(best_f, best_x)
                        stagnation_counter = 0
                    else:
                        sigma = np.clip(sigma * 0.8, sigma_min, None)
                        stagnation_counter += 1

            # Update success memory and potentially adjust parameters
            success_memory.append(gen_success / pop_size if pop_size > 0 else 0)
            if len(success_memory) > window:
                success_memory.pop(0)
            avg_success = np.mean(success_memory) if success_memory else 0.5
            # (no further adaptation since F/CR are oscillated)

            # Inject diversity if low
            if low_diversity and evals < budget:
                # Replace worst performer with random point
                worst_idx = np.argmax(fitness)
                new_point = lb + rng.rand(dim) * (ub - lb)
                f_new = func(new_point)
                evals += 1
                if f_new < fitness[worst_idx]:
                    points[worst_idx] = new_point
                    fitness[worst_idx] = f_new
                    if f_new < best_f:
                        best_f = f_new
                        best_x = new_point.copy()
                        report_best(best_f, best_x)
                        stagnation_counter = 0
                else:
                    stagnation_counter += 1

            # Restart if stagnation
            if stagnation_counter >= max_stagnation and evals < budget:
                # Reduce population size by 1 and reinitialize half
                new_pop = max(4, pop_size - 1)
                new_points = np.empty((new_pop, dim))
                for d in range(dim):
                    perm = rng.permutation(new_pop)
                    u = rng.rand(new_pop)
                    new_points[:, d] = lb[d] + (perm + u) / new_pop * (ub[d] - lb[d])
                # Replace worst individuals
                worst_indices = np.argsort(fitness)[-new_pop:] if len(fitness) >= new_pop else list(range(len(fitness)))
                for idx in worst_indices[:new_pop]:
                    if evals >= budget:
                        break
                    x = new_points[idx]
                    f = func(x)
                    evals += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                    points[idx] = x
                    fitness[idx] = f
                # Reset stagnation and sigma
                stagnation_counter = 0
                sigma = 0.2 * (ub - lb)

        return best_f, best_x