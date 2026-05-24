import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        evals = 0

        best_value = float('inf')
        best_x = None

        # initial population size
        popsize = max(4, min(5 * dim, 20))
        popsize = min(popsize, budget)  # never exceed budget

        stagnation_limit = 5

        # outer loop for restarts
        while evals < budget:
            # initialize population
            pop = lb + (ub - lb) * rng.rand(popsize, dim)
            pop_fitness = np.zeros(popsize)
            for i in range(popsize):
                pop_fitness[i] = func(pop[i])
                evals += 1
                if pop_fitness[i] < best_value:
                    best_value = pop_fitness[i]
                    best_x = pop[i].copy()
                    report_best(best_value, best_x)
                if evals >= budget:
                    break
            if evals >= budget:
                break

            # memory for this run
            H = 5
            F_memory = np.ones(H) * 0.5
            CR_memory = np.ones(H) * 0.5
            memory_counter = 0
            generations_no_improve = 0

            # inner loop for generations
            while evals < budget:
                improved_this_gen = False
                F_success = []
                CR_success = []

                for i in range(popsize):
                    if evals >= budget:
                        break
                    # generate F and CR
                    r = rng.randint(H)
                    F = F_memory[r] + 0.1 * rng.randn()
                    CR = CR_memory[r] + 0.1 * rng.randn()
                    F = np.clip(F, 0, 1)
                    CR = np.clip(CR, 0, 1)

                    # mutation: DE/rand/1/bin
                    candidates = list(range(popsize))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    a, b, c = candidates[:3]
                    mutant = pop[a] + F * (pop[b] - pop[c])

                    # binomial crossover
                    trial = pop[i].copy()
                    j_rand = rng.randint(dim)
                    for j in range(dim):
                        if rng.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    trial = np.clip(trial, lb, ub)

                    trial_fitness = func(trial)
                    evals += 1
                    if trial_fitness <= pop_fitness[i]:
                        pop_fitness[i] = trial_fitness
                        pop[i] = trial
                        if trial_fitness < best_value:
                            best_value = trial_fitness
                            best_x = trial.copy()
                            report_best(best_value, best_x)
                            improved_this_gen = True
                        F_success.append(F)
                        CR_success.append(CR)

                if evals >= budget:
                    break

                # update memory if successes
                if len(F_success) > 0:
                    sum_F = np.sum(F_success)
                    sum_F2 = np.sum(np.square(F_success))
                    mean_F = sum_F2 / sum_F if sum_F > 0 else 0.5
                    mean_CR = np.mean(CR_success)
                    F_memory[memory_counter % H] = mean_F
                    CR_memory[memory_counter % H] = mean_CR
                    memory_counter += 1

                # stagnation detection
                if improved_this_gen:
                    generations_no_improve = 0
                else:
                    generations_no_improve += 1

                if generations_no_improve >= stagnation_limit:
                    # try restart with larger population
                    new_popsize = min(int(popsize * 1.5), budget - evals)
                    new_popsize = max(4, new_popsize)
                    if evals + new_popsize <= budget and new_popsize > popsize:
                        popsize = new_popsize
                        break  # go to outer loop to reinitialize
                    else:
                        # cannot restart, reset counter but continue
                        generations_no_improve = 0
            # end inner loop
        # end outer loop
        return best_value, best_x