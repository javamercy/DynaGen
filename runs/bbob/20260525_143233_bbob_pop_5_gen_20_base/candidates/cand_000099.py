import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(5 * dim, budget // 4))
        self.restart_threshold = max(10, 2 * dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng

        best_val = np.inf
        best_x = None
        evals = 0

        # Initialize population and parameters
        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        F = rng.uniform(0.5, 1.0, pop_size)
        CR = rng.uniform(0.0, 1.0, pop_size)
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        no_improve = 0
        generation = 0
        max_gen = (budget - evals) // pop_size if pop_size > 0 else 0

        while evals < budget and generation < max_gen:
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                F_i = F[i]
                mutant = pop[i] + F_i * (best_x - pop[i]) + F_i * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)

                CR_i = CR[i]
                cross_points = rng.rand(dim) < CR_i
                if not np.any(cross_points):
                    cross_points[rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                else:
                    # Update F and CR with small probability
                    tau1 = 0.1
                    tau2 = 0.1
                    if rng.rand() < tau1:
                        F[i] = rng.uniform(0.5, 1.0)
                    if rng.rand() < tau2:
                        CR[i] = rng.uniform(0.0, 1.0)

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.restart_threshold:
                new_pop = rng.uniform(lb, ub, (pop_size, dim))
                new_pop[0] = best_x.copy()
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                new_F = rng.uniform(0.5, 1.0, pop_size)
                new_CR = rng.uniform(0.0, 1.0, pop_size)
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    x = new_pop[i]
                    val = func(x)
                    evals += 1
                    new_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                pop = new_pop
                fitness = new_fitness
                F = new_F
                CR = new_CR
                no_improve = 0

            generation += 1

        return best_val, best_x