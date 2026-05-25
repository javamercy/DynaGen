import numpy as np
import random

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = min(5 * dim, max(5, budget // 5))
        pop_size = max(3, min(pop_size, budget // 2))
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0
        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            fcalls += 1
            pop_f[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)
        F_start = 0.9
        F_end = 0.2
        CR_start = 0.9
        CR_end = 0.2
        local_budget = max(10, int(0.3 * budget))
        de_budget = budget - local_budget
        while fcalls < de_budget:
            for i in range(pop_size):
                if fcalls >= de_budget:
                    break
                progress = fcalls / budget
                F = F_start - (F_start - F_end) * progress
                CR = CR_start - (CR_start - CR_end) * progress
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    r0 = random.choice(candidates)
                    r1 = random.choice(candidates)
                    r2 = random.choice([c for c in candidates if c != r0 and c != r1])
                else:
                    r0, r1, r2 = random.sample(candidates, 3)
                mutant = pop[r0] + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                trial = pop[i].copy()
                j_rand = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                val = func(trial)
                fcalls += 1
                if val < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
        if best_x is None:
            best_x = np.random.uniform(lb, ub)
            best_f = func(best_x)
            fcalls += 1
            report_best(best_f, best_x)
        step = 0.1 * (ub - lb)
        while fcalls < budget:
            improved = False
            for j in range(dim):
                if fcalls >= budget:
                    break
                # try positive step
                x_pos = best_x.copy()
                x_pos[j] = np.clip(best_x[j] + step[j], lb[j], ub[j])
                val_pos = func(x_pos)
                fcalls += 1
                if val_pos < best_f:
                    best_f = val_pos
                    best_x = x_pos.copy()
                    report_best(best_f, best_x)
                    step[j] = min(step[j] * 1.2, 0.5 * (ub[j] - lb[j]))
                    improved = True
                    break
                else:
                    # try negative step
                    x_neg = best_x.copy()
                    x_neg[j] = np.clip(best_x[j] - step[j], lb[j], ub[j])
                    val_neg = func(x_neg)
                    fcalls += 1
                    if val_neg < best_f:
                        best_f = val_neg
                        best_x = x_neg.copy()
                        report_best(best_f, best_x)
                        step[j] = min(step[j] * 1.2, 0.5 * (ub[j] - lb[j]))
                        improved = True
                        break
                    else:
                        step[j] = max(step[j] * 0.5, 1e-10 * (ub[j] - lb[j]))
            if not improved and fcalls < budget:
                # random perturbation
                x_rand = best_x + np.random.normal(0, np.mean(step), dim)
                x_rand = np.clip(x_rand, lb, ub)
                val_rand = func(x_rand)
                fcalls += 1
                if val_rand < best_f:
                    best_f = val_rand
                    best_x = x_rand.copy()
                    report_best(best_f, best_x)
                    step = 0.1 * (ub - lb)  # reset step
        return best_f, best_x