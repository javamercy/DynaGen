import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = min(self.budget, max(4, min(5 * self.dim, self.budget // 3)))
        def lhs_sample(n):
            points = np.zeros((n, self.dim))
            for i in range(self.dim):
                perm = self.rng.permutation(n)
                u = self.rng.rand(n)
                points[:, i] = lb[i] + (perm + u) / n * (ub[i] - lb[i])
            return points
        points = lhs_sample(pop_size)
        fits = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            fits[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        F = 0.5
        CR = 0.9
        stagnation = 0
        while evals < self.budget:
            old_best = best_f
            # one generation
            target_order = self.rng.permutation(pop_size)
            for target in target_order:
                if evals >= self.budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(target)
                if len(candidates) < 3:
                    continue
                a, b, c = self.rng.choice(candidates, 3, replace=False)
                mutant = points[a] + F * (points[b] - points[c])
                trial = points[target].copy()
                j_rand = self.rng.randint(self.dim)
                for j in range(self.dim):
                    if self.rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < fits[target]:
                    points[target] = trial
                    fits[target] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
            if best_f < old_best:
                stagnation = 0
            else:
                stagnation += 1
            # restart if stagnation
            if stagnation >= 1 and evals < self.budget:
                # keep best, reinitialize others
                remaining_evals = self.budget - evals
                new_pop_size = min(pop_size, remaining_evals)
                if new_pop_size <= 1:
                    continue
                new_points = lhs_sample(new_pop_size - 1)  # minus best
                # replace all but best, and adjust population size
                # we'll keep best at index 0, and replace the rest
                new_points = np.vstack([best_x.reshape(1, -1), new_points])
                # evaluate new points (except best already evaluated)
                for i in range(1, new_pop_size):
                    if evals >= self.budget:
                        break
                    x = new_points[i]
                    f = func(x)
                    evals += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                # update population
                points = new_points[:new_pop_size]
                fits = np.full(new_pop_size, np.inf)
                fits[0] = best_f
                for i in range(1, new_pop_size):
                    fits[i] = func(points[i])  # already evaluated above, but recalc? No, we stored f but need to get from loop. Let's restructure.
                # Actually we already computed f in the loop. Store in temporary array.
                # Better: evaluate new points and store immediately.
                # Let's redo this section.
                # We'll just re-evaluate the new points to avoid complexity, but that wastes evals. To be efficient, we should keep the evaluations.
                # I'll do a cleaner approach: after the loop, we have the fitnesses.
                # We'll just keep the best and ignore others? We'll reinitialize the whole population with best kept.
                # Simpler: keep best, reinitialize rest with LHS, evaluate them, and set population.
                # But we already evaluated some of those points in the main loop? No, restart happens after a generation, so we have a full generation done.
                # We'll just reinitialize the entire population (except best) with LHS and evaluate them.
                # This uses extra evals for the new points. That's fine.
                # So:
                new_population = [best_x.copy()]
                for _ in range(pop_size - 1):
                    if evals >= self.budget:
                        break
                    x = lhs_sample(1)[0]
                    f = func(x)
                    evals += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                    new_population.append(x)
                pop_size = len(new_population)
                points = np.array(new_population)
                fits = np.array([best_f] + [func(p) for p in points[1:]])
                stagnation = 0
        return best_f, best_x