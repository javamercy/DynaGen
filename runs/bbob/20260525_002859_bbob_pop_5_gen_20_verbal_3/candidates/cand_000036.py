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
        popsize = min(budget, max(4, min(5 * dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.full(popsize, np.inf)

        # initial evaluation
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if self.best_value is None or pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
            if evals >= budget:
                return self.best_value, self.best_x

        # JADE parameters
        memory_size = 5
        mu_F = 0.5
        mu_CR = 0.5
        F_memory = []
        CR_memory = []

        gen_since_improve = 0
        stagnation_limit = max(5, dim)

        while evals < budget:
            # generate F and CR for each individual
            F_list = np.zeros(popsize)
            CR_list = np.zeros(popsize)
            for i in range(popsize):
                F = rng.standard_cauchy()
                F = mu_F + 0.1 * F
                F = np.clip(F, 0, 1)
                F_list[i] = F
                CR = mu_CR + 0.1 * rng.randn()
                CR = np.clip(CR, 0, 1)
                CR_list[i] = CR

            improvement = False
            successful_F = []
            successful_CR = []

            for i in range(popsize):
                # mutation: current-to-best/1
                best_idx = np.argmin(pop_fitness)
                # select two distinct random indices != i
                candidates = list(range(popsize))
                candidates.remove(i)
                if best_idx in candidates:
                    candidates.remove(best_idx)
                # need at least 2 candidates
                if len(candidates) < 2:
                    a = b = 0  # fallback, but should not happen
                else:
                    rng.shuffle(candidates)
                    a, b = candidates[:2]
                mutant = pop[i] + F_list[i] * (pop[best_idx] - pop[i]) + F_list[i] * (pop[a] - pop[b])

                # binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR_list[i] or j == j_rand:
                        trial[j] = mutant[j]

                # clip to bounds
                trial = np.clip(trial, lb, ub)

                # evaluate
                trial_fitness = func(trial)
                evals += 1

                # selection
                if trial_fitness <= pop_fitness[i]:
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        improvement = True
                    # store successful parameters if improvement (strictly better than parent? For JADE, usually if <= parent)
                    if trial_fitness <= pop_fitness[i]:  # equal counts as success to keep memory fresh
                        successful_F.append(F_list[i])
                        successful_CR.append(CR_list[i])

                if evals >= budget:
                    break

            # update memory and mu values
            if successful_F:
                # update F memory with replacement if full
                for f, cr in zip(successful_F, successful_CR):
                    if len(F_memory) < memory_size:
                        F_memory.append(f)
                        CR_memory.append(cr)
                    else:
                        idx = rng.randint(memory_size)
                        F_memory[idx] = f
                        CR_memory[idx] = cr
                # Lehmer mean for F
                if len(F_memory) > 0:
                    sum_F = sum(F_memory)
                    sum_F2 = sum(f*f for f in F_memory)
                    if sum_F > 0:
                        mu_F = sum_F2 / sum_F
                # arithmetic mean for CR
                if len(CR_memory) > 0:
                    mu_CR = np.mean(CR_memory)

            # stagnation check
            if improvement:
                gen_since_improve = 0
            else:
                gen_since_improve += 1

            if gen_since_improve >= stagnation_limit and evals < budget:
                # restart: reinitialize all except best
                best_idx = np.argmin(pop_fitness)
                for i in range(popsize):
                    if i != best_idx:
                        pop[i] = lb + (ub - lb) * rng.rand(dim)
                        pop_fitness[i] = func(pop[i])
                        evals += 1
                        if pop_fitness[i] < self.best_value:
                            self.best_value = pop_fitness[i]
                            self.best_x = pop[i].copy()
                            report_best(self.best_value, self.best_x)
                        if evals >= budget:
                            break
                # reset stagnation counter and memory?
                gen_since_improve = 0
                F_memory = []
                CR_memory = []
                mu_F = 0.5
                mu_CR = 0.5

        return self.best_value, self.best_x