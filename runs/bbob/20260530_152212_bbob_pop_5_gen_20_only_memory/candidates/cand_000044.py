import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(10, min(budget // 3, 10 * dim))  # larger population for diversity

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        dim = self.dim
        pop_size = self.pop_size
        F0 = 0.9  # larger F for exploration
        CR = 0.9
        sigma0 = 0.3 * (ub - lb).mean()  # larger initial step

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

        # Stagnation tracking
        no_improve = 0
        max_stagnation = max(10, pop_size)

        while evals < budget:
            # DE/rand/1/bin (explorative)
            F = F0 * (0.5 + 0.5 * np.random.rand())  # random F in [0.5,1.0]
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
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            # Restart if stagnated
            if no_improve > max_stagnation and evals < budget:
                # Reinitialize half of population randomly
                n_restart = max(1, pop_size // 2)
                idxs = np.random.choice(pop_size, n_restart, replace=False)
                for idx in idxs:
                    if evals >= budget:
                        break
                    x = np.random.uniform(lb, ub, dim)
                    val = func(x)
                    evals += 1
                    pop[idx] = x
                    fitness[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                no_improve = 0

            # Minimal local refinement (only if budget remains)
            if evals < budget:
                ratio = 1.0 - evals / budget
                sigma = sigma0 * ratio  # linear decay
                n_local = min(3, budget - evals)  # few local steps
                if pop_size > 1:
                    C = np.cov(pop.T) + 1e-8 * np.eye(dim)
                else:
                    C = np.eye(dim)
                for _ in range(n_local):
                    if evals >= budget:
                        break
                    delta = np.random.multivariate_normal(np.zeros(dim), sigma**2 * C)
                    trial = best_x + delta
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        no_improve = 0
                    else:
                        sigma *= 0.95

        return best_val, best_x