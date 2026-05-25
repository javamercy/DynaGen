import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.pop_size = max(4, min(4 * dim, budget // 2))
        self.restart_threshold = max(5, dim)

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
            x = pop[i].copy()
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Adaptive parameters
        F = 0.7
        CR = 0.9
        local_sigma = 0.01
        # Success history for adaptation
        success_F = []
        success_CR = []
        generation = 0
        no_improve = 0
        max_gen = (self.budget - evals) // pop_size if pop_size > 0 else 0
        while evals < self.budget and generation < max_gen:
            # Mutation using DE/best/1 with adaptive F and CR
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                # Generate F and CR for this individual from adaptive distributions
                if len(success_F) > 0:
                    mu_F = np.mean(success_F)
                    mu_CR = np.mean(success_CR)
                else:
                    mu_F = F
                    mu_CR = CR
                # Cauchy distribution for F, truncated to [0,1]
                F_i = np.random.cauchy(mu_F, 0.1)
                F_i = np.clip(F_i, 0, 1)
                # Normal distribution for CR, truncated to [0,1]
                CR_i = np.random.normal(mu_CR, 0.1)
                CR_i = np.clip(CR_i, 0, 1)
                mutant = best_x + F_i * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                crossover = np.random.rand(dim) < CR_i
                if not np.any(crossover):
                    crossover[np.random.randint(0, dim)] = True
                trial = np.where(crossover, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    success_F.append(F_i)
                    success_CR.append(CR_i)
                    if len(success_F) > dim:  # keep memory limited
                        success_F.pop(0)
                        success_CR.pop(0)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            # Local search around best
            if evals < self.budget:
                local_evals = min(max(1, int(0.1 * pop_size)), self.budget - evals)
                for _ in range(local_evals):
                    x = best_x + local_sigma * np.random.randn(dim) * (ub - lb)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
            # Adapt local sigma (decrease over time)
            local_sigma *= 0.99
            if local_sigma < 1e-6:
                local_sigma = 1e-6
            # Check stagnation
            if evals > 0 and np.min(fitness) <= best_val:  # Improvement occurred
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.restart_threshold:
                # Focused restart: 30% around best, 70% random
                new_pop = np.random.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    num_focused = int(0.3 * pop_size)
                    for j in range(num_focused):
                        new_pop[j] = best_x + 0.1 * np.random.randn(dim) * (ub - lb)
                        new_pop[j] = np.clip(new_pop[j], lb, ub)
                    new_pop[0] = best_x.copy()
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= self.budget:
                        break
                    x = new_pop[i].copy()
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
                # Reset adaptive parameters after restart
                success_F = []
                success_CR = []
            generation += 1
        return best_val, best_x