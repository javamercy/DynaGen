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

        # Population size
        NP = min(10 * dim, max(4, budget // 2 - 1))
        if budget < NP:
            NP = budget

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.zeros(NP)
        for i in range(NP):
            fitness[i] = func(pop[i])

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_val = fitness[best_idx]
        report_best(best_val, best_x)
        func_evals = NP

        # SHADE memory
        H = 5
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        archive_size = NP

        # Restart counter
        no_improve = 0
        max_no_improve = 6 * dim

        while func_evals < budget:
            if no_improve >= max_no_improve:
                # Restart: keep best, reinitialize rest
                remaining_budget = budget - func_evals
                new_NP = min(10 * dim, max(4, remaining_budget // 2 - 1))
                if new_NP < 2:
                    break
                # Keep best
                new_pop = np.zeros((new_NP, dim))
                new_fit = np.zeros(new_NP)
                new_pop[0] = best_x
                new_fit[0] = best_val
                # Generate new random individuals
                for i in range(1, new_NP):
                    new_pop[i] = lb + (ub - lb) * rng.rand(dim)
                    new_fit[i] = func(new_pop[i])
                    func_evals += 1
                    if new_fit[i] < best_val:
                        best_val = new_fit[i]
                        best_x = new_pop[i].copy()
                        report_best(best_val, best_x)
                    if func_evals >= budget:
                        break
                pop = new_pop
                fitness = new_fit
                NP = new_NP
                no_improve = 0
                continue

            # Generation loop
            gen_evals = 0
            for i in range(NP):
                if func_evals >= budget:
                    break
                # Select random indices for mutation
                candidates = list(range(NP))
                candidates.remove(i)
                # pbest selection
                p = 0.2
                pbest_size = max(1, int(p * NP))
                sorted_idx = np.argsort(fitness)
                pbest_idx = sorted_idx[:pbest_size]
                pbest = pop[pbest_idx[rng.randint(pbest_size)]]
                # Select from population and archive
                if len(archive) > 0:
                    # one from population (excluding i), one from archive
                    idx_pop = rng.choice(candidates)
                    idx_arch = rng.randint(len(archive))
                    a = pop[idx_pop]
                    b = archive[idx_arch]
                else:
                    # two from population
                    idx = rng.choice(candidates, 2, replace=False)
                    a, b = pop[idx[0]], pop[idx[1]]
                # Sample F and CR
                r = rng.randint(H)
                F = np.clip(rng.normal(M_F[r], 0.1), 0, 1)
                CR = np.clip(rng.normal(M_CR[r], 0.1), 0, 1)
                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (pbest - pop[i]) + F * (a - b)
                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # Clip
                trial = np.clip(trial, lb, ub)
                # Evaluate
                trial_fit = func(trial)
                func_evals += 1
                gen_evals += 1
                if trial_fit < fitness[i]:
                    # Add parent to archive
                    archive.append(pop[i].copy())
                    if len(archive) > archive_size:
                        archive.pop(rng.randint(len(archive)))
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        no_improve = 0
                    # Update memory
                    M_F = np.roll(M_F, -1)
                    M_F[-1] = F
                    M_CR = np.roll(M_CR, -1)
                    M_CR[-1] = CR
            if gen_evals == 0:
                break
            no_improve += 1
        return best_val, best_x