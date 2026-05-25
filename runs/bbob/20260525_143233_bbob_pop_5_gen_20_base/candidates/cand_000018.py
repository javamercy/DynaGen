import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        # population size: at least 4, at most budget//2, scaling with dim
        self.pop_size = max(4, min(4 * dim, budget // 2))
        if self.pop_size < 1:
            self.pop_size = 1
        # restart thresholds
        self.restart_gen = max(5, 2 * dim)
        # adaptation window
        self.window = max(1, int(0.2 * self.pop_size))

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        
        # fallback for tiny budget
        if pop_size <= 1:
            best_x = np.random.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < budget:
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x
        
        # initialization
        pop = np.random.uniform(lb, ub, (pop_size, dim))
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
        
        # DE parameters
        F = 0.5
        CR = 0.9
        success_hist = []  # store success per individual for adaptation
        no_improve = 0
        generation = 0
        max_gen = (budget - evals) // pop_size if pop_size > 0 else 0
        
        while evals < budget and generation < max_gen:
            # generate offspring
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            improved_global = False
            gen_success = 0
            for i in range(pop_size):
                if evals >= budget:
                    break
                # current-to-best/1 mutation
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b = np.random.choice(candidates, size=2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = val
                    gen_success += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        improved_global = True
            # update population
            pop = new_pop
            fitness = new_fitness
            # record success rate
            success_hist.append(gen_success / pop_size)
            if len(success_hist) > self.window:
                success_hist.pop(0)
            # adapt F and CR
            if len(success_hist) >= 2:
                mean_success = np.mean(success_hist)
                if mean_success > 0.2:
                    F = min(0.9, F * 1.05)
                    CR = min(0.95, CR * 1.02)
                else:
                    F = max(0.1, F * 0.95)
                    CR = max(0.2, CR * 0.98)
            # restart check
            if improved_global:
                no_improve = 0
            else:
                no_improve += 1
            # also check diversity
            if pop_size > 1:
                diversity = np.mean(np.std(pop, axis=0)) / np.mean(ub - lb) if np.mean(ub - lb) > 0 else 0
            else:
                diversity = 1.0
            restart_flag = (no_improve >= self.restart_gen) or (diversity < 1e-3)
            if restart_flag:
                # restart except best
                new_pop = np.random.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    new_pop[0] = best_x.copy()
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
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
                F = 0.5
                CR = 0.9
                success_hist = []
                no_improve = 0
            generation += 1
        return best_val, best_x