import numpy as np
class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = None
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0
        popsize = min(budget, max(10, 5*dim))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.zeros(popsize)
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if evals == 1 or pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x
        # SHADE memory for F
        H = 5
        F_memory = np.full(H, 0.8)
        # Strategy probabilities (0: current-to-best, 1: rand/1)
        prob = np.array([0.5, 0.5])
        CR = 0.9
        stagnation_counter = 0
        max_stagnation = max(5, int(budget / (3 * popsize)))
        while evals < budget:
            # For storing successes this generation
            success_F = []
            success_strategy = []  # 0 or 1
            for i in range(popsize):
                if evals >= budget:
                    break
                # Choose mutation strategy
                if rng.rand() < prob[0]:
                    strategy = 0
                else:
                    strategy = 1
                # Sample F from memory
                idx = rng.randint(H)
                F = rng.cauchy(F_memory[idx], 0.1)
                F = np.clip(F, 0.1, 0.9)
                if strategy == 0:
                    # current-to-best/1
                    candidates = list(range(popsize))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    r1, r2 = candidates[0], candidates[1]
                    mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                else:
                    # rand/1
                    candidates = list(range(popsize))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    r1, r2, r3 = candidates[0], candidates[1], candidates[2]
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness <= pop_fitness[i]:
                    if trial_fitness < pop_fitness[i]:  # strict improvement for success tracking (<= is success)
                        success_F.append(F)
                        success_strategy.append(strategy)
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
            if evals >= budget:
                break
            # Update memory with successes
            if len(success_F) > 0:
                # Replace random memory entries
                k = min(len(success_F), H)
                idxs = rng.choice(H, k, replace=False)
                for idx, j in enumerate(idxs):
                    F_memory[j] = success_F[idx]
            # Update strategy probabilities using success rates
            if len(success_strategy) > 0:
                count0 = success_strategy.count(0)
                count1 = success_strategy.count(1)
                total = count0 + count1
                if total > 0:
                    prob[0] = 0.9 * prob[0] + 0.1 * (count0 / total)
                    prob[1] = 1 - prob[0]
            # Check stagnation
            improved_this_gen = len(success_F) > 0
            if improved_this_gen:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            if stagnation_counter >= max_stagnation and evals + popsize <= budget:
                # Restart: perturb best and generate random points
                new_pop = np.zeros((popsize, dim))
                new_fitness = np.zeros(popsize)
                new_pop[0] = self.best_x + 0.1 * (ub - lb) * rng.randn(dim)
                new_pop[0] = np.clip(new_pop[0], lb, ub)
                new_fitness[0] = func(new_pop[0])
                evals += 1
                if new_fitness[0] < self.best_value:
                    self.best_value = new_fitness[0]
                    self.best_x = new_pop[0].copy()
                    report_best(self.best_value, self.best_x)
                for i in range(1, popsize):
                    if evals >= budget:
                        break
                    x = lb + (ub - lb) * rng.rand(dim)
                    x = np.clip(x, lb, ub)
                    f = func(x)
                    evals += 1
                    new_pop[i] = x
                    new_fitness[i] = f
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)
                pop = new_pop
                pop_fitness = new_fitness
                # Reset memories
                F_memory[:] = 0.8
                prob[:] = 0.5
                stagnation_counter = 0
                if evals >= budget:
                    break
        return self.best_value, self.best_x