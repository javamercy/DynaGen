import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        initial_pop = max(4, min(4 * dim, budget // 2))
        initial_pop = min(initial_pop, budget)
        self.initial_pop = max(initial_pop, 1)
        self.restart_threshold = max(5, 2 * dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # handle degenerate case
        pop_size = self.initial_pop
        if pop_size == 1:
            best_x = rng.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < budget:
                x = rng.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # initialization
        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
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

        # parameters
        F = 0.5
        CR = 0.9
        no_improve = 0
        generation = 0
        max_evals = budget

        # main loop
        while evals < max_evals:
            # linear population size reduction
            remaining = max_evals - evals
            total_remaining = max_evals - evals
            # target pop size: linearly from initial_pop to 4
            if total_remaining > 0:
                frac = evals / max_evals
                target_pop = int(self.initial_pop + (4 - self.initial_pop) * frac)
                target_pop = max(4, min(target_pop, pop_size))
            else:
                target_pop = pop_size
            # if current pop > target, remove worst individuals
            if pop_size > target_pop:
                # combine pop and fitness, sort by fitness, keep best target_pop
                idx = np.argsort(fitness)
                pop = pop[idx[:target_pop]]
                fitness = fitness[idx[:target_pop]]
                pop_size = target_pop

            improved_this_gen = False
            # generate new individuals
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            for i in range(pop_size):
                if evals >= max_evals:
                    break
                # mutation: best/1/bin
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 2:
                    continue
                a, b = rng.choice(candidates, size=2, replace=False)
                # best vector
                best_idx = np.argmin(fitness)
                mutant = pop[best_idx] + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < new_fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = val
                    improved_this_gen = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            pop = new_pop
            fitness = new_fitness

            # adapt F
            if improved_this_gen:
                F *= 1.1
                F = min(F, 0.9)
                no_improve = 0
            else:
                F *= 0.9
                F = max(F, 0.1)
                no_improve += 1

            # restart if stagnation
            if no_improve >= self.restart_threshold:
                new_pop = rng.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    new_pop[0] = best_x.copy()
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= max_evals:
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
                F = 0.5
                no_improve = 0

            generation += 1

        return best_val, best_x