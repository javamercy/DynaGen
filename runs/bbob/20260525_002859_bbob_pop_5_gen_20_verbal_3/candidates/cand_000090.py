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

        # Population size
        popsize = min(budget // 2, max(4, 5 * dim))
        popsize = max(4, popsize)
        # Latin hypercube initialization
        pop = np.zeros((popsize, dim))
        for j in range(dim):
            perm = rng.permutation(popsize)
            pop[:, j] = lb[j] + (ub[j] - lb[j]) * (perm + rng.uniform(size=popsize)) / popsize
        fit = np.empty(popsize)

        evals = 0
        for i in range(popsize):
            fit[i] = func(pop[i])
            evals += 1
            if evals == 1 or fit[i] < self.best_value:
                self.best_value = fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        # Adaptive parameters
        F = 0.8
        CR = 0.9
        memory_size = 10
        F_success = []
        CR_success = []
        stagnation_limit = max(10 * dim, int(0.15 * budget))
        stagnation_counter = 0

        # Diversity monitoring
        initial_diversity = np.mean([np.linalg.norm(pop[i] - np.mean(pop, axis=0)) for i in range(popsize)])
        diversity_threshold = 0.1 * initial_diversity
        generation = 0
        max_generations = budget // popsize
        p_explore = 0.3

        while evals < budget:
            generation += 1
            new_pop = pop.copy()
            new_fit = fit.copy()
            # Update p_explore (decay)
            p_explore = max(0.05, 0.3 - 0.25 * (generation / max_generations))
            for i in range(popsize):
                if evals >= budget:
                    break
                # Select mutation strategy
                if rng.rand() < p_explore:
                    # DE/rand/1
                    candidates = list(range(popsize))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    r1, r2, r3 = candidates[0], candidates[1], candidates[2]
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                else:
                    # DE/current-to-best/1
                    candidates = list(range(popsize))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    r1, r2 = candidates[0], candidates[1]
                    mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fit = func(trial)
                evals += 1
                if trial_fit <= fit[i]:
                    new_fit[i] = trial_fit
                    new_pop[i] = trial
                    if trial_fit < self.best_value:
                        self.best_value = trial_fit
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        stagnation_counter = 0
                        F_success.append(F)
                        CR_success.append(CR)
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1
                # Diversity check: if trial is far from any existing, maybe keep? Not implemented
            pop = new_pop
            fit = new_fit

            # Update adaptive parameters
            if len(F_success) >= memory_size:
                F = np.mean(F_success[-memory_size:])
                CR = np.mean(CR_success[-memory_size:])
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

            # Diversity preservation: if diversity too low, diversify
            if generation % 5 == 0 and evals < budget - popsize:
                current_diversity = np.mean([np.linalg.norm(pop[i] - np.mean(pop, axis=0)) for i in range(popsize)])
                if current_diversity < diversity_threshold:
                    # Replace worst half (excluding best) with random perturbations
                    sorted_idx = np.argsort(fit)
                    worst_indices = sorted_idx[1:]  # exclude best
                    rng.shuffle(worst_indices)
                    num_replace = min(len(worst_indices), popsize // 2)
                    for idx in worst_indices[:num_replace]:
                        if evals >= budget:
                            break
                        # Random perturbation of best + uniform
                        if rng.rand() < 0.5:
                            sigma = (ub - lb) * 0.2 * (1 + 0.5 * rng.randn())
                            new_x = self.best_x + rng.randn(dim) * sigma
                            new_x = np.clip(new_x, lb, ub)
                        else:
                            new_x = lb + (ub - lb) * rng.rand(dim)
                        f = func(new_x)
                        evals += 1
                        pop[idx] = new_x
                        fit[idx] = f
                        if f < self.best_value:
                            self.best_value = f
                            self.best_x = new_x.copy()
                            report_best(self.best_value, self.best_x)
                    # Reset success memories after diversification
                    F_success = []
                    CR_success = []
                    # Reset stagnation counter
                    stagnation_counter = 0
                    # Reset p_explore to initial
                    p_explore = 0.3

            # Restart if stagnation and enough budget left
            if stagnation_counter >= stagnation_limit and evals < budget - popsize:
                stagnation_counter = 0
                # Keep best individual
                new_pop = [self.best_x.copy()]
                new_fit = [self.best_value]
                evals_for_restart = 1  # best already evaluated
                # Reinitialize worst half
                sorted_idx = np.argsort(fit)
                worst_indices = sorted_idx[1:]  # exclude best
                rng.shuffle(worst_indices)
                num_new = min(len(worst_indices), popsize - 1)
                for idx in worst_indices[:num_new]:
                    if evals >= budget:
                        break
                    if rng.rand() < 0.5:
                        sigma = (ub - lb) * 0.2 * (1 + 0.5 * rng.randn())
                        new_x = self.best_x + rng.randn(dim) * sigma
                        new_x = np.clip(new_x, lb, ub)
                    else:
                        new_x = lb + (ub - lb) * rng.rand(dim)
                    f = func(new_x)
                    evals += 1
                    new_pop.append(new_x)
                    new_fit.append(f)
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = new_x.copy()
                        report_best(self.best_value, self.best_x)
                # Fill remaining (if any) with random uniform
                while len(new_pop) < popsize and evals < budget:
                    new_x = lb + (ub - lb) * rng.rand(dim)
                    f = func(new_x)
                    evals += 1
                    new_pop.append(new_x)
                    new_fit.append(f)
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = new_x.copy()
                        report_best(self.best_value, self.best_x)
                pop = np.array(new_pop[:popsize])
                fit = np.array(new_fit[:popsize])
                # Reset success memories
                F_success = []
                CR_success = []
                # Reset p_explore
                p_explore = 0.3
                # Local refinement on best
                if evals < budget - 1:
                    sigma = (ub - lb) * 0.05
                    for _ in range(min(5, budget - evals)):
                        candidate = self.best_x + rng.randn(dim) * sigma
                        candidate = np.clip(candidate, lb, ub)
                        f = func(candidate)
                        evals += 1
                        if f < self.best_value:
                            self.best_value = f
                            self.best_x = candidate.copy()
                            report_best(self.best_value, self.best_x)
                        if evals >= budget:
                            break
                if evals >= budget:
                    break
        return self.best_value, self.best_x