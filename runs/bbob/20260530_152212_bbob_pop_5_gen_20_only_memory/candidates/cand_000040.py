import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(budget // 10, 5 * dim))
        self.restart_threshold = max(10, int(0.05 * budget))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size

        # DE parameters
        F = 0.5
        CR = 0.5

        # Initialization
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
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

        # Main loop
        no_improve_count = 0
        while evals < budget:
            # DE generation
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        no_improve_count = 0
                else:
                    no_improve_count += 1

            # Pattern search around best
            if evals < budget:
                step = 0.2 * (ub - lb).mean() * (1 - evals / budget)
                improved = False
                order = np.random.permutation(dim)
                for i in order:
                    if evals >= budget:
                        break
                    for direction in [1, -1]:
                        x_try = best_x.copy()
                        x_try[i] += direction * step
                        x_try[i] = np.clip(x_try[i], lb[i], ub[i])
                        val = func(x_try)
                        evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = x_try.copy()
                            report_best(best_val, best_x)
                            # Line search along direction
                            direction_vec = np.zeros(dim)
                            direction_vec[i] = direction
                            line_length = 0.5
                            while evals < budget:
                                x_line = best_x + line_length * direction_vec
                                x_line = np.clip(x_line, lb, ub)
                                v = func(x_line)
                                evals += 1
                                if v < best_val:
                                    best_val = v
                                    best_x = x_line.copy()
                                    report_best(best_val, best_x)
                                    line_length *= 2
                                else:
                                    line_length *= 0.5
                                    if line_length < 1e-8:
                                        break
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    no_improve_count = 0
                else:
                    no_improve_count += dim

            # Restart if stagnation
            if no_improve_count >= self.restart_threshold and evals < budget:
                # Keep best, reinitialize population
                pop = np.random.uniform(lb, ub, (pop_size, dim))
                pop[0] = best_x.copy()
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    if i == 0:
                        continue
                    x = pop[i]
                    val = func(x)
                    evals += 1
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                no_improve_count = 0

        return best_val, best_x