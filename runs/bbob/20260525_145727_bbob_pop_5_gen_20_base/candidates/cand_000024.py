import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        pop_size = max(5, min(30, budget//2, 2*dim))
        if pop_size > budget:
            pop_size = budget

        pop_x = rng.uniform(lb, ub, (pop_size, dim))
        pop_y = np.full(pop_size, np.inf)
        best_x = None
        best_y = np.inf

        for i in range(pop_size):
            pop_y[i] = func(pop_x[i])
            if pop_y[i] < best_y:
                best_y = pop_y[i]
                best_x = pop_x[i].copy()
                report_best(best_y, best_x)

        evals = pop_size
        remaining = budget - evals
        if remaining <= 0:
            return best_y, best_x

        DE_frac = 0.7
        N_DE = int(DE_frac * remaining) if pop_size >= 4 else 0
        N_local = remaining - N_DE

        no_improve = 0
        restart_interval = max(1, N_DE // 5)

        # DE phase
        if N_DE > 0:
            for i in range(N_DE):
                progress = i / max(1, N_DE)
                F = 0.9 - 0.5 * progress
                CR = 0.9 - 0.7 * progress

                # Restart if no improvement for restart_interval evaluations
                if no_improve >= restart_interval:
                    # Reinitialize worst half of population
                    idx_sorted = np.argsort(-pop_y)  # descending order
                    num_restart = pop_size // 2
                    for idx in idx_sorted[:num_restart]:
                        pop_x[idx] = rng.uniform(lb, ub, size=dim)
                        pop_y[idx] = func(pop_x[idx])
                        evals += 1
                        if pop_y[idx] < best_y:
                            best_y = pop_y[idx]
                            best_x = pop_x[idx].copy()
                            report_best(best_y, best_x)
                    no_improve = 0

                # DE/rand/2 mutation
                target_idx = rng.randint(pop_size)
                indices = list(range(pop_size))
                indices.remove(target_idx)
                if len(indices) < 4:
                    break
                a_idx, b_idx, c_idx, d_idx, e_idx = rng.choice(indices, 5, replace=False)
                a = pop_x[a_idx]
                b = pop_x[b_idx]
                c = pop_x[c_idx]
                d = pop_x[d_idx]
                e = pop_x[e_idx]
                mutant = a + F * (b - c) + F * (d - e)
                trial = pop_x[target_idx].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_y = func(trial)
                evals += 1
                if trial_y < pop_y[target_idx]:
                    pop_x[target_idx] = trial
                    pop_y[target_idx] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()
                        report_best(best_y, best_x)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

        # Local search phase with Cauchy perturbations
        if N_local > 0:
            step_init = 0.2 * (ub - lb)
            for i in range(N_local):
                progress = i / max(1, N_local)
                step = step_init * (1 - progress)**2
                # Cauchy distribution for heavier tails
                cauchy = rng.standard_cauchy(size=dim)
                trial = best_x + step * cauchy
                trial = np.clip(trial, lb, ub)
                trial_y = func(trial)
                evals += 1
                if trial_y < best_y:
                    best_y = trial_y
                    best_x = trial.copy()
                    report_best(best_y, best_x)
                    worst_idx = np.argmax(pop_y)
                    if trial_y < pop_y[worst_idx]:
                        pop_x[worst_idx] = trial
                        pop_y[worst_idx] = trial_y

        return best_y, best_x