import numpy as np

class ALS_SAR:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        max_evals = self.budget

        # Parameters
        H = 5                     # memory size
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = 4
        N = N_init
        archive_max = 2 * N_init
        max_stagnation = max(int(0.1 * max_evals), 100)

        # Latin hypercube initialization
        samples = np.random.uniform(0, 1, (N, dim))
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive
        archive = np.empty((0, dim))

        # Memory
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.5
        memory_idx = 0

        # Stagnation tracking
        best_f_last = self.f_opt
        stagnation_counter = 0

        while n_evals < max_evals:
            # Check for restart
            if stagnation_counter >= max_stagnation and N > N_min:
                # Keep best, reinitialize rest
                n_new = N - 1
                new_samples = np.random.uniform(0, 1, (n_new, dim))
                new_pop = lb + new_samples * (ub - lb)
                # Evaluate new individuals
                new_fitness = np.full(n_new, np.inf)
                for i in range(n_new):
                    new_fitness[i] = func(new_pop[i])
                    n_evals += 1
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_pop[i].copy()
                # Combine with best
                pop = np.vstack((self.x_opt.reshape(1, -1), new_pop))
                fitness = np.hstack((self.f_opt, new_fitness))
                # Reset archive and memory
                archive = np.empty((0, dim))
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                stagnation_counter = 0
                best_f_last = self.f_opt
                continue

            # Adaptive pbest ratio
            p = 0.2 - 0.1 * (n_evals / max_evals)

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
                # Select distinct r1
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)

                # r2 from union pop and archive
                if archive.shape[0] > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])

                # pbest selection
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)

                # Sample F and CR from memory
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.standard_normal(), 0, 1)

                # Mutation (current-to-pbest/1 with archive)
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # Bound handling: random reinitialization for out-of-bound components
                oob_low = trial < lb
                oob_high = trial > ub
                if np.any(oob_low) or np.any(oob_high):
                    random_replace = np.random.uniform(lb, ub, size=trial.shape)
                    trial = np.where(oob_low | oob_high, random_replace, trial)

                # Evaluate
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add replaced individual to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / np.sum(delta_f)
                MF[memory_idx] = np.sum(w * S_F**2) / np.sum(w * S_F)
                MCR[memory_idx] = np.sum(w * S_CR**2) / np.sum(w * S_CR)
                memory_idx = (memory_idx + 1) % H

            # Linear population size reduction
            N_new = round(N_init - (N_init - N_min) * (n_evals / max_evals))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                N = N_new

            # Update stagnation
            if self.f_opt < best_f_last:
                best_f_last = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt