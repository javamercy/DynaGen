import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.pop_size = max(4, min(4 * dim, budget // 2))
        self.pop_size = max(self.pop_size, 1)
        self.max_generations = (budget - self.pop_size) // self.pop_size if self.pop_size > 0 else 0
        self.restart_threshold = max(5, int(budget / (4 * self.pop_size))) if self.pop_size > 0 else 5

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        if pop_size <= 0:
            best_x = np.random.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < self.budget:
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
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
        while evals < self.budget and generation < self.max_generations:
            gen_frac = generation / max(1, self.max_generations)
            F = 0.6 + 0.4 * np.sin(2 * np.pi * gen_frac)  # range 0.2-1.0
            CR = 0.5 + 0.4 * np.cos(2 * np.pi * gen_frac)  # range 0.1-0.9
            F = np.clip(F, 0.2, 1.0)
            CR = np.clip(CR, 0.1, 0.9)
            improved_this_gen = False
            # Diversity measure: standard deviation of population per dimension
            pop_std = np.std(pop, axis=0).mean()
            low_diversity = pop_std < 0.01 * (ub - lb).mean()
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                # Choose mutation strategy randomly
                strategy = np.random.choice(['rand1', 'rand2', 'best1', 'best2'], p=[0.5, 0.2, 0.2, 0.1])
                if strategy == 'rand1':
                    a, b, c = np.random.choice(candidates, size=3, replace=False)
                    mutant = pop[a] + F * (pop[b] - pop[c])
                elif strategy == 'rand2':
                    a, b, c, d, e = np.random.choice(candidates, size=5, replace=False)
                    mutant = pop[a] + F * (pop[b] - pop[c]) + F * (pop[d] - pop[e])
                elif strategy == 'best1':
                    b, c = np.random.choice(candidates, size=2, replace=False)
                    mutant = best_x + F * (pop[b] - pop[c])
                else:  # best2
                    b, c, d, e = np.random.choice(candidates, size=4, replace=False)
                    mutant = best_x + F * (pop[b] - pop[c]) + F * (pop[d] - pop[e])
                mutant = np.clip(mutant, lb, ub)
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved_this_gen = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            # If population is too similar, perturb some individuals
            if low_diversity and not improved_this_gen:
                for i in range(pop_size):
                    if np.random.rand() < 0.3:
                        perturbation = np.random.uniform(-0.1, 0.1, dim) * (ub - lb)
                        new_x = np.clip(pop[i] + perturbation, lb, ub)
                        val = func(new_x)
                        evals += 1
                        if val < fitness[i]:
                            pop[i] = new_x
                            fitness[i] = val
                            if val < best_val:
                                best_val = val
                                best_x = new_x.copy()
                                report_best(best_val, best_x)
                improved_this_gen = True  # to reset counter?
            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.restart_threshold:
                # Restart with best preserved and noise added
                new_pop = np.random.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    new_pop[0] = best_x.copy()
                # Add some perturbed copies of best
                for i in range(1, min(5, pop_size)):
                    noise = np.random.uniform(-0.05, 0.05, dim) * (ub - lb)
                    new_pop[i] = np.clip(best_x + noise, lb, ub)
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= self.budget:
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
                no_improve = 0
            generation += 1
        return best_val, best_x