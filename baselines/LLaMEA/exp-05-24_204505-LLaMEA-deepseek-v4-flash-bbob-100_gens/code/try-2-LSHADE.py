import numpy as np

class LSHADE:
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

        # Population size: initial and minimum
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = 4
        N = N_init

        # Latin hypercube initialization
        samples = np.random.uniform(0, 1, (N, dim))
        samples = lb + samples * (ub - lb)
        pop = samples.copy()
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive for inferior solutions
        archive = np.empty((0, dim))
        archive_max = N_init

        # Success-history memory parameters (increased size)
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.5
        memory_idx = 0

        # Main loop
        while n_evals < max_evals:
            p = 0.2 - 0.1 * (n_evals / max_evals)

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
                # r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)

                # r2 from union of population and archive
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])

                # pbest selection
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)

                # Sample F and CR from memory with adaptive scaling
                mem = np.random.randint(H)
                # Cauchy scale reduces over generations
                cauchy_scale = 0.1 * (1 - n_evals / max_evals) + 0.05
                F = MF[mem] + cauchy_scale * np.random.standard_cauchy()
                # Truncate to [0,1] and ensure F > 0
                F = np.clip(F, 0, 1)
                while F <= 0:
                    F = MF[mem] + cauchy_scale * np.random.standard_cauchy()
                    F = np.clip(F, 0, 1)
                # CR with normal distribution (std = 0.1)
                CR = MCR[mem] + 0.1 * np.random.randn()
                CR = np.clip(CR, 0, 1)

                # Mutation (current-to-pbest/1 with archive)
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2

                # Crossover (binomial)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # Enhanced bound handling: quasi-reflection then random repair
                # Reflect off bounds once
                trial = np.where(trial < lb, 2 * lb - trial, trial)
                trial = np.where(trial > ub, 2 * ub - trial, trial)
                # If still out-of-bounds, randomly reinitialize that component
                out_lb = trial < lb
                out_ub = trial > ub
                if np.any(out_lb):
                    trial[out_lb] = lb[out_lb] + np.random.uniform(0, 0.05 * (ub[out_lb] - lb[out_lb]))
                if np.any(out_ub):
                    trial[out_ub] = ub[out_ub] - np.random.uniform(0, 0.05 * (ub[out_ub] - lb[out_ub]))
                # Final clamp for safety
                trial = np.clip(trial, lb, ub)

                # Evaluate
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()

                if trial_f < fitness[i]:
                    # Success: record F, CR and fitness improvement
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial

                    # Add the replaced individual to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory if successful candidates exist
            if len(S_F) > 0:
                # Sort indices by delta_f descending
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order]
                w_sum = np.sum(w) + 1e-30  # avoid division by zero
                w = w / w_sum

                # Lehmer mean for F
                MF[memory_idx] = np.sum(w * S_F**2) / (np.sum(w * S_F) + 1e-30)
                # Weighted mean for CR
                MCR[memory_idx] = np.sum(w * S_CR**2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Linear population size reduction
            N_new = round(N_init - (N_init - N_min) * (n_evals / max_evals))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                # Reduce archive size proportionally
                if archive.shape[0] > archive_max:
                    archive = archive[:archive_max]
                N = N_new

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt