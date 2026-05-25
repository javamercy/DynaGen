import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub

        pop_size = min(10 * dim, max(4, budget // 2))
        if pop_size < 4:
            # fallback to random search
            best_x = None
            best_f = np.inf
            for _ in range(budget):
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                    report_best(best_f, best_x)
            return best_f, best_x

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0

        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            fcalls += 1
            pop_f[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)

        stall_counter = 0
        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break

                # mutation with adaptive F
                indices = list(range(pop_size))
                indices.remove(i)
                r0, r1, r2 = np.random.choice(indices, 3, replace=False)
                F = np.random.uniform(0.5, 1.0)
                mutant = pop[r0] + F * (pop[r1] - pop[r2])

                # crossover with adaptive CR
                CR = np.random.uniform(0.0, 1.0)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                j_rand = np.random.randint(dim)
                trial[j_rand] = mutant[j_rand]
                trial = np.clip(trial, lb, ub)

                val = func(trial)
                fcalls += 1
                if val < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        stall_counter = 0
                    else:
                        stall_counter += 1
                else:
                    stall_counter += 1

                # occasional random injection every 10 evaluations
                if fcalls % 10 == 0 and fcalls < budget:
                    x_rand = np.random.uniform(lb, ub, dim)
                    val_rand = func(x_rand)
                    fcalls += 1
                    worst_idx = np.argmax(pop_f)
                    if val_rand < pop_f[worst_idx]:
                        pop[worst_idx] = x_rand
                        pop_f[worst_idx] = val_rand
                        if val_rand < best_f:
                            best_f = val_rand
                            best_x = x_rand.copy()
                            report_best(best_f, best_x)
                            stall_counter = 0
                        else:
                            stall_counter += 1
                    else:
                        stall_counter += 1

                # restart if stagnation
                if stall_counter >= 2 * pop_size:
                    # reinitialize half of the population (excluding best)
                    num_reinit = pop_size // 2
                    # find best index
                    best_idx = np.argmin(pop_f)
                    others = [j for j in range(pop_size) if j != best_idx]
                    if len(others) >= num_reinit:
                        reinit_indices = np.random.choice(others, num_reinit, replace=False)
                        for idx in reinit_indices:
                            if fcalls >= budget:
                                break
                            x_new = np.random.uniform(lb, ub, dim)
                            val_new = func(x_new)
                            fcalls += 1
                            pop[idx] = x_new
                            pop_f[idx] = val_new
                            if val_new < best_f:
                                best_f = val_new
                                best_x = x_new.copy()
                                report_best(best_f, best_x)
                                stall_counter = 0
                    stall_counter = 0

        return best_f, best_x