import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        dim = self.dim
        budget = self.budget
        rng = self.rng

        pop_size = max(2, min(5, dim // 2))
        walkers = []
        evals = 0
        best_f = None
        best_x = None

        for _ in range(pop_size):
            x = lb + rng.rand(dim) * (ub - lb)
            f = func(x)
            evals += 1
            walkers.append({
                'x': x.copy(),
                'f': f,
                'step': (ub - lb).mean() / 5.0,
                'stagnation': 0
            })
            if best_f is None or f < best_f:
                best_f = f
                best_x = x.copy()

        # report initial best
        from array import array
        # report_best(best_f, best_x)  # uncomment if available

        max_stagnation = max(10, 2 * dim)

        while evals < budget:
            for w in walkers:
                if evals >= budget:
                    break
                # generate candidate
                x_cand = w['x'] + w['step'] * rng.randn(dim)
                x_cand = np.clip(x_cand, lb, ub)
                f_cand = func(x_cand)
                evals += 1

                if f_cand < w['f']:
                    w['x'] = x_cand.copy()
                    w['f'] = f_cand
                    w['step'] *= 1.2
                    w['stagnation'] = 0
                    if f_cand < best_f:
                        best_f = f_cand
                        best_x = x_cand.copy()
                        # report_best(best_f, best_x)
                else:
                    w['step'] *= 0.8
                    w['stagnation'] += 1

                # ensure minimum step size
                w['step'] = max(w['step'], 1e-10 * (ub - lb).mean())

            # reinitialize stagnated walkers after each full pass
            for i, w in enumerate(walkers):
                if evals >= budget:
                    break
                if w['stagnation'] >= max_stagnation:
                    x_new = lb + rng.rand(dim) * (ub - lb)
                    f_new = func(x_new)
                    evals += 1
                    walkers[i] = {
                        'x': x_new.copy(),
                        'f': f_new,
                        'step': (ub - lb).mean() / 5.0,
                        'stagnation': 0
                    }
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        # report_best(best_f, best_x)

        return best_f, best_x