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
        # population size
        popsize = min(budget, max(4, min(5*dim, 30)))
        # initial population
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

        # self-adaptive control parameters: each individual has its own F and CR
        # initial F and CR means; adaptation using historical success
        F_mean = 0.5
        CR_mean = 0.9
        # memory for successful F and CR
        F_success = []
        CR_success = []
        # max size of success memory
        memory_size = min(20, popsize)
        
        # stagnation tracking
        stagnation_counter = 0
        max_stagnation = max(5, int(budget / (2 * popsize)))
        # local search parameters
        local_search_step = 0.05 * (ub - lb)  # initial step size for local perturbation
        min_step = 1e-8
        
        # initialize F and CR arrays for each individual
        F = np.full(popsize, 0.5)
        CR = np.full(popsize, 0.9)
        
        while evals < budget:
            improved_this_gen = False
            # generate new F and CR for each individual (self-adaptation)
            for i in range(popsize):
                if rng.rand() < 0.1:
                    F[i] = rng.rand()  # reset with small probability
                else:
                    F[i] = F_mean + 0.1 * rng.randn()
                F[i] = np.clip(F[i], 0.1, 0.9)
                
                if rng.rand() < 0.1:
                    CR[i] = rng.rand()
                else:
                    CR[i] = CR_mean + 0.1 * rng.randn()
                CR[i] = np.clip(CR[i], 0.0, 1.0)
            
            # generate trial vectors
            for i in range(popsize):
                if evals >= budget:
                    break
                # mutation: current-to-best
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2 = candidates[0], candidates[1]
                mutant = pop[i] + F[i] * (self.best_x - pop[i]) + F[i] * (pop[r1] - pop[r2])
                # crossover: binomial
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                # clip
                trial = np.clip(trial, lb, ub)
                # evaluate
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness <= pop_fitness[i]:
                    # update population
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    # update best
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        improved_this_gen = True
                    # record successful parameters
                    F_success.append(F[i])
                    CR_success.append(CR[i])
                    if len(F_success) > memory_size:
                        F_success.pop(0)
                        CR_success.pop(0)
            
            if evals >= budget:
                break
            
            # update means using successful parameters (if any)
            if len(F_success) > 0:
                F_mean = np.mean(F_success)
                CR_mean = np.mean(CR_success)
            
            # stagnation check
            if improved_this_gen:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # restart condition
            if stagnation_counter >= max_stagnation and evals + popsize - 1 <= budget:
                # local search on best before restart
                for _ in range(min(5, (budget - evals) // 1)):
                    if evals >= budget:
                        break
                    step = local_search_step * (1 - evals / budget)  # shrinking step
                    if step < min_step:
                        step = min_step
                    perturbation = rng.uniform(-step, step, dim)
                    candidate = np.clip(self.best_x + perturbation, lb, ub)
                    f = func(candidate)
                    evals += 1
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = candidate.copy()
                        report_best(self.best_value, self.best_x)
                # restart: keep best, reinitialize rest
                new_pop = np.zeros((popsize, dim))
                new_fitness = np.zeros(popsize)
                new_pop[0] = self.best_x
                new_fitness[0] = self.best_value
                for i in range(1, popsize):
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
                stagnation_counter = 0
                # reinitialize F and CR
                F = np.full(popsize, 0.5)
                CR = np.full(popsize, 0.9)
                F_success = []
                CR_success = []
                F_mean = 0.5
                CR_mean = 0.9
                local_search_step = 0.05 * (ub - lb)  # reset step size
                if evals >= budget:
                    break
        return self.best_value, self.best_x