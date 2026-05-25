import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng
        
        # Handle tiny budgets
        if budget < 3:
            for _ in range(budget):
                x = lb + (ub - lb) * rng.rand(dim)
                val = func(x)
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = x.copy()
                    report_best(self.best_val, self.best_x)
            return self.best_val, self.best_x
        
        # Parameters
        pop_size = min(budget, max(4, min(20, budget // 5)))
        F = 0.8
        CR = 0.9
        T0 = 1.0
        T_end = 1e-3
        # Cooling per evaluation
        if budget > 1:
            cooling_rate = np.exp((np.log(T_end) - np.log(T0)) / (budget - 1))
        else:
            cooling_rate = 1.0
        T = T0
        
        evals = 0
        # Initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fit = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            pop_fit[i] = func(pop[i])
            evals += 1
            T *= cooling_rate
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        
        # Main loop
        while evals < budget:
            new_pop = pop.copy()
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Mutation: current-to-best/1
                idxs = [j for j in range(pop_size) if j != i]
                a, b = rng.choice(idxs, 2, replace=False)
                mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                # Crossover
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fit = func(trial)
                evals += 1
                T *= cooling_rate
                # Acceptance
                if trial_fit < pop_fit[i]:
                    # Accept better
                    new_pop[i] = trial
                    pop_fit[i] = trial_fit
                    if trial_fit < self.best_val:
                        self.best_val = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_val, self.best_x)
                else:
                    # Possibly accept worse with SA probability
                    delta = trial_fit - pop_fit[i]
                    prob = np.exp(-delta / T)
                    if rng.rand() < prob:
                        new_pop[i] = trial
                        pop_fit[i] = trial_fit
            pop = new_pop
        
        return self.best_val, self.best_x