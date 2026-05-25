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
        pop_size = max(4 * dim, 10)
        if pop_size > budget // 2:
            pop_size = budget // 2
        if pop_size < 2:
            pop_size = 2

        # Latin hypercube sampling for initialization
        points = np.empty((pop_size, dim))
        for i in range(dim):
            perm = rng.permutation(pop_size)
            points[:, i] = lb[i] + (perm + 0.5) / pop_size * (ub[i] - lb[i])

        best_f = np.inf
        best_x = np.zeros(dim)
        evals = 0
        population = []
        for i in range(pop_size):
            x = points[i]
            f = func(x)
            evals += 1
            population.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # DE parameters
        F = rng.uniform(0.1, 1.0, pop_size)  # per-individual scaling factor
        CR = rng.uniform(0, 1.0, pop_size)   # per-individual crossover rate

        # Pattern search step for local refinement
        step = 0.1 * np.mean(ub - lb)
        step_initial = step
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)

        stagnation_counter = 0
        max_stagnation = 10

        while evals < budget:
            # DE generation
            new_pop = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Generate new F and CR with probability 0.1 (jDE)
                new_F = F[i]
                new_CR = CR[i]
                if rng.rand() < 0.1:
                    new_F = rng.uniform(0.1, 1.0)
                if rng.rand() < 0.1:
                    new_CR = rng.uniform(0, 1.0)

                # Mutation: DE/rand/1
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                mutant = population[a][0] + new_F * (population[b][0] - population[c][0])
                # Crossover: binomial
                trial = population[i][0].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < new_CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1

                if f_trial <= population[i][1]:
                    new_pop.append((trial.copy(), f_trial))
                    F[i] = new_F
                    CR[i] = new_CR
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        stagnation_counter = max(0, stagnation_counter - 2)
                else:
                    new_pop.append(population[i])
            population = new_pop

            if evals >= budget:
                break

            # Local pattern search on best every generation
            local_evals = 0
            max_local_evals = min(2 * dim, budget - evals)
            current_x = best_x.copy()
            current_f = best_f
            local_step = step
            while local_evals < max_local_evals:
                improved = False
                for d in directions:
                    if local_evals >= max_local_evals:
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
            # Update step for next generation
            step = local_step
            if step < 1e-15:
                step = step_initial

            # Check stagnation and restart
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
                population = new_pop
                # Reset F and CR for new individuals
                F = rng.uniform(0.1, 1.0, pop_size)
                CR = rng.uniform(0, 1.0, pop_size)
                stagnation_counter = 0

        return best_f, best_x