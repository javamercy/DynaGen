import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = min(budget, max(20, min(4 * dim, budget // 2)))
        self.stall_limit = max(10, budget // 20)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        pop = self.rng.uniform(lb, ub, size=(popsize, dim))
        vel = np.zeros((popsize, dim))
        fitness = np.full(popsize, np.inf)
        personal_best = pop.copy()
        personal_best_fitness = np.full(popsize, np.inf)
        global_best = None
        global_best_fitness = np.inf
        evaluations = 0
        for i in range(popsize):
            if evaluations >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            personal_best_fitness[i] = val
            if val < global_best_fitness:
                global_best_fitness = val
                global_best = x.copy()
                report_best(global_best_fitness, global_best)
        if global_best is None:
            global_best = self.rng.uniform(lb, ub)
            global_best_fitness = func(global_best)
            evaluations += 1
            report_best(global_best_fitness, global_best)
        stall_generations = 0
        generation = 0
        while evaluations < self.budget:
            w = 0.9 - 0.5 * (generation / (self.budget / popsize))
            w = max(0.4, min(0.9, w))
            c1 = 2.0
            c2 = 2.0
            improved = False
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                vel[i] = w * vel[i] + c1 * r1 * (personal_best[i] - pop[i]) + c2 * r2 * (global_best - pop[i])
                pop[i] = pop[i] + vel[i]
                pop[i] = np.clip(pop[i], lb, ub)
                val = func(pop[i])
                evaluations += 1
                fitness[i] = val
                if val < personal_best_fitness[i]:
                    personal_best_fitness[i] = val
                    personal_best[i] = pop[i].copy()
                    if val < global_best_fitness:
                        global_best_fitness = val
                        global_best = pop[i].copy()
                        report_best(global_best_fitness, global_best)
                        improved = True
            if improved:
                stall_generations = 0
            else:
                stall_generations += 1
            if stall_generations > self.stall_limit:
                sorted_indices = np.argsort(fitness)
                n_restart = popsize // 2
                worst_indices = sorted_indices[-n_restart:]
                for idx in worst_indices:
                    if evaluations >= self.budget:
                        break
                    pop[idx] = self.rng.uniform(lb, ub)
                    vel[idx] = np.zeros(dim)
                    val = func(pop[idx])
                    evaluations += 1
                    fitness[idx] = val
                    personal_best_fitness[idx] = val
                    personal_best[idx] = pop[idx].copy()
                    if val < global_best_fitness:
                        global_best_fitness = val
                        global_best = pop[idx].copy()
                        report_best(global_best_fitness, global_best)
                stall_generations = 0
            generation += 1
        return global_best_fitness, global_best