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
        popsize = max(10, min(5*dim, budget//2))
        if popsize * 3 > budget:
            popsize = max(3, budget//3)
        # Initialize population
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.full(popsize, np.inf)
        best_value = np.inf
        best_x = None
        for i in range(popsize):
            f = func(pop[i])
            evals += 1
            pop_fitness[i] = f
            if f < best_value:
                best_value = f
                best_x = pop[i].copy()
                report_best(best_value, best_x)
        if evals >= budget:
            return best_value, best_x
        # JADE memory
        mu_F = 0.5
        mu_CR = 0.5
        memory_size = 10
        memory_F = np.full(memory_size, 0.5)
        memory_CR = np.full(memory_size, 0.5)
        memory_index = 0
        # Strategy adaptation
        strategy_probs = np.array([0.5, 0.5])
        learning_rate = 0.1
        stagnation_counter = 0
        max_stagnation = max(5, int(budget / (5 * popsize)))
        while evals < budget:
            # Sort for pbest
            sorted_idx = np.argsort(pop_fitness)
            pbest_size = max(1, int(0.2 * popsize))
            pbest_pool = sorted_idx[:pbest_size]
            successful_F = []
            successful_CR = []
            strategy_success = [0, 0]
            strategy_trials = [0, 0]
            improved_gen = False
            for i in range(popsize):
                if evals >= budget:
                    break
                # Choose strategy
                s = rng.choice([0,1], p=strategy_probs)
                # Generate F and CR
                F = np.clip(rng.normal(mu_F, 0.1), 0, 1)
                CR = np.clip(rng.normal(mu_CR, 0.1), 0, 1)
                # Mutation
                if s == 0:  # current-to-pbest/1
                    pbest = rng.choice(pbest_pool)
                    candidates = [j for j in range(popsize) if j not in (i, pbest)]
                    r1, r2 = rng.choice(candidates, 2, replace=False)
                    mutant = pop[i] + F * (pop[pbest] - pop[i]) + F * (pop[r1] - pop[r2])
                else:  # rand/1
                    candidates = [j for j in range(popsize) if j != i]
                    r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # Crossover
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                # Selection
                if f_trial <= pop_fitness[i]:
                    pop_fitness[i] = f_trial
                    pop[i] = trial
                    if f_trial < best_value:
                        best_value = f_trial
                        best_x = trial.copy()
                        report_best(best_value, best_x)
                        improved_gen = True
                    successful_F.append(F)
                    successful_CR.append(CR)
                    strategy_success[s] += 1
                strategy_trials[s] += 1
                if evals >= budget:
                    break
            # Update strategy probabilities
            total_success = sum(strategy_success)
            total_trials = sum(strategy_trials)
            if total_trials > 0 and total_success > 0:
                success_rates = [strategy_success[k]/max(1,strategy_trials[k]) for k in range(2)]
                baseline = total_success / total_trials
                for k in range(2):
                    if success_rates[k] > baseline + 0.05:
                        strategy_probs[k] = min(1.0, strategy_probs[k] + learning_rate)
                    elif success_rates[k] < baseline - 0.05:
                        strategy_probs[k] = max(0.1, strategy_probs[k] - learning_rate)
                strategy_probs /= strategy_probs.sum()
            # Update F/CR memory
            if len(successful_F) > 0:
                mu_F = (1 - 0.1) * mu_F + 0.1 * np.mean(successful_F)
                mu_CR = (1 - 0.1) * mu_CR + 0.1 * np.mean(successful_CR)
                memory_F[memory_index] = np.mean(successful_F)
                memory_CR[memory_index] = np.mean(successful_CR)
                memory_index = (memory_index + 1) % memory_size
                mu_F = np.mean(memory_F)
                mu_CR = np.mean(memory_CR)
            # Stagnation and restart
            if improved_gen:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            if stagnation_counter >= max_stagnation and evals + popsize - 1 <= budget:
                new_pop = np.zeros((popsize, dim))
                new_fitness = np.zeros(popsize)
                new_pop[0] = best_x + 0.1 * (ub - lb) * rng.randn(dim)
                new_pop[0] = np.clip(new_pop[0], lb, ub)
                new_fitness[0] = func(new_pop[0])
                evals += 1
                if new_fitness[0] < best_value:
                    best_value = new_fitness[0]
                    best_x = new_pop[0].copy()
                    report_best(best_value, best_x)
                for i in range(1, popsize):
                    x = lb + (ub - lb) * rng.rand(dim)
                    x = np.clip(x, lb, ub)
                    f = func(x)
                    evals += 1
                    new_pop[i] = x
                    new_fitness[i] = f
                    if f < best_value:
                        best_value = f
                        best_x = x.copy()
                        report_best(best_value, best_x)
                pop = new_pop
                pop_fitness = new_fitness
                stagnation_counter = 0
                # Reset F/CR memory? Keep as is
                if evals >= budget:
                    break
        return best_value, best_x