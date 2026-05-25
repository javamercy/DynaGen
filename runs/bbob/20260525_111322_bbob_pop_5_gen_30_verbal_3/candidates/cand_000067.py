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

        # Population size
        pop_size = max(4, min(4 * dim, budget // 2))
        if pop_size < 2:
            pop_size = min(2, budget)
        if pop_size == 0:
            # budget 0 case
            best_x = lb + rng.rand(dim) * (ub - lb)
            best_f = func(best_x)
            report_best(best_f, best_x)
            return best_f, best_x

        # LHS initialization
        points = np.empty((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            u = rng.rand(pop_size)
            points[:, d] = lb[d] + (perm + u) / pop_size * (ub[d] - lb[d])

        evals = 0
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf

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

        if evals == 0:
            # Should not happen
            best_x = lb + rng.rand(dim) * (ub - lb)
            best_f = func(best_x)
            report_best(best_f, best_x)
            return best_f, best_x

        # jDE parameters per individual
        F = rng.uniform(0.1, 1.0, pop_size)
        CR = rng.uniform(0, 1.0, pop_size)

        # Local pattern search setup
        step = 0.1 * np.mean(ub - lb)
        step_initial = step
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)

        stagnation_counter = 0
        max_stagnation = max(10, budget // (pop_size * 10))

        # Main loop
        while evals < budget:
            # DE generation
            new_pop = []
            new_fitness = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Update F and CR with probability 0.1 (jDE)
                new_F = F[i]
                new_CR = CR[i]
                if rng.rand() < 0.1:
                    new_F = rng.uniform(0.1, 1.0)
                if rng.rand() < 0.1:
                    new_CR = rng.uniform(0, 1.0)

                # Mutation: DE/rand/1
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                mutant = points[a] + new_F * (points[b] - points[c])

                # Crossover: binomial
                trial = points[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < new_CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    new_pop.append(trial.copy())
                    new_fitness.append(f_trial)
                    F[i] = new_F
                    CR[i] = new_CR
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        stagnation_counter = max(0, stagnation_counter - 2)
                else:
                    new_pop.append(points[i].copy())
                    new_fitness.append(fitness[i])

            if len(new_pop) == pop_size:
                points = np.array(new_pop)
                fitness = np.array(new_fitness)
            else:
                # Only partial generation, keep remaining from previous
                pass

            if evals >= budget:
                break

            # Local pattern search on best solution
            local_evals = 0
            max_local_evals = min(2 * dim, budget - evals)
            current_x = best_x.copy()
            current_f = best_f
            local_step = step
            while local_evals < max_local_evals and evals < budget:
                improved = False
                for d in directions:
                    if local_evals >= max_local_evals or evals >= budget:
                        break
                    candidate = current_x + local_step * d
                    candidate = np.clip(candidate, lb, ub)
                    f_val = func(candidate)
                    evals += 1
                    local_evals += 1
                    if f_val < current_f:
                        current_f = f_val
                        current_x = candidate.copy()
                        if current_f < best_f:
                            best_f = current_f
                            best_x = current_x.copy()
                            report_best(best_f, best_x)
                            stagnation_counter = max(0, stagnation_counter - 2)
                        improved = True
                        break
                if improved:
                    local_step *= 1.2
                else:
                    local_step *= 0.5
                    if local_step < 1e-15:
                        break
            step = max(local_step, 1e-15)

            # Stagnation check and restart
            stagnation_counter += 1
            if stagnation_counter > max_stagnation and evals < budget:
                # Restart population except best
                new_pop = [(best_x.copy(), best_f)]
                for _ in range(pop_size - 1):
                    if evals >= budget:
                        break
                    x = lb + rng.rand(dim) * (ub - lb)
                    f = func(x)
                    evals += 1
                    new_pop.append((x.copy(), f))
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                points = np.array([p[0] for p in new_pop])
                fitness = np.array([p[1] for p in new_pop])
                # Reset F and CR
                F = rng.uniform(0.1, 1.0, pop_size)
                CR = rng.uniform(0, 1.0, pop_size)
                stagnation_counter = 0
                step = step_initial

        return best_f, best_x