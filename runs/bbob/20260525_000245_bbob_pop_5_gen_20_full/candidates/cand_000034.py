import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        rng = self.rng

        # Population size
        NP = max(4, min(20, budget // (dim + 1)))
        if NP < 4:
            NP = 4

        # Initialize population
        pop = rng.uniform(lb, ub, size=(NP, dim))
        pop_fitness = np.full(NP, np.inf)
        calls = 0
        best_x = None
        best_val = np.inf

        # Evaluate initial population
        for i in range(NP):
            if calls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            calls += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # DE parameters
        F_start = 0.9
        F_end = 0.5
        CR_min = 0.5
        CR_max = 0.9
        CR_period = 10  # for sinusoidal variation
        generation = 0

        # Main loop
        while calls < budget:
            # Adaptive F
            if budget > 0:
                F = F_start - (F_start - F_end) * (generation / (budget // NP if NP > 0 else 1))
                F = np.clip(F, F_end, F_start)
            else:
                F = F_start
            # Sinusoidal CR
            CR = CR_min + (CR_max - CR_min) * (0.5 + 0.5 * np.sin(2 * np.pi * generation / CR_period))

            # Diversity-triggered restart (check fitness variance)
            if generation > 5 and np.var(pop_fitness) < 1e-8 * (np.max(pop_fitness) - np.min(pop_fitness) + 1e-8):
                # Reinitialize half of population (excluding best)
                num_restart = NP // 2
                # Keep the best individual
                best_idx = np.argmin(pop_fitness)
                indices = [i for i in range(NP) if i != best_idx]
                if len(indices) >= num_restart:
                    restart_indices = rng.choice(indices, size=num_restart, replace=False)
                    for idx in restart_indices:
                        pop[idx] = rng.uniform(lb, ub, size=dim)
                        # Evaluate new individual? Better to evaluate on the fly during generation
                        # We'll evaluate later in generation loop

            generation += 1

            # Generation loop
            for i in range(NP):
                if calls >= budget:
                    break
                # Mutation: current-to-best/1
                indices = [j for j in range(NP) if j != i]
                r1, r2 = rng.choice(indices, size=2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.uniform() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # Clip to bounds
                trial = np.clip(trial, lb, ub)
                # Evaluate
                val = func(trial)
                calls += 1
                # Greedy selection
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        return best_val, best_x