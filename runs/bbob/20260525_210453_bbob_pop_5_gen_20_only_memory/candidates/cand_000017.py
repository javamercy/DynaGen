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

        pop_size = min(3 * dim, max(4, budget // 4))
        if pop_size < 4:
            best_x = np.random.uniform(lb, ub, dim)
            best_f = func(best_x)
            report_best(best_f, best_x)
            fcalls = 1
            while fcalls < budget:
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                fcalls += 1
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

        F = 0.3
        CR = 0.9
        budget_local = max(1, int(0.1 * budget))

        while fcalls < budget - budget_local:
            for i in range(pop_size):
                if fcalls >= budget - budget_local:
                    break
                best_idx = np.argmin(pop_f)
                indices = [j for j in range(pop_size) if j != i and j != best_idx]
                if len(indices) < 2:
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(indices, 2, replace=False)
                    mutant = pop[r1] + F * (pop[r2] - pop[r1])
                else:
                    r1, r2 = np.random.choice(indices, 2, replace=False)
                    mutant = pop[best_idx] + F * (pop[r1] - pop[r2])
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

        step_size = (ub - lb) * 0.02
        improvement = True
        while improvement and fcalls < budget:
            improvement = False
            for d in range(dim):
                if fcalls >= budget:
                    break
                for delta in [step_size[d], -step_size[d]]:
                    if fcalls >= budget:
                        break
                    x_new = best_x.copy()
                    x_new[d] += delta
                    x_new = np.clip(x_new, lb, ub)
                    val = func(x_new)
                    fcalls += 1
                    if val < best_f:
                        best_f = val
                        best_x = x_new.copy()
                        report_best(best_f, best_x)
                        improvement = True
                        step_size[d] *= 2
                        break
                    else:
                        step_size[d] *= 0.5
        return best_f, best_x