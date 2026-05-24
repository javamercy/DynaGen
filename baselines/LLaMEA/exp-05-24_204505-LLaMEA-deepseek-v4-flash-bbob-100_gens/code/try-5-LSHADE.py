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

        # Population size parameters
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
        archive_max = N

        # Success-history memory parameters
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation and diversity detection
        best_since_last_restart = self.f_opt
        evals_since_last_improvement = 0
        restart_threshold = 0.1 * max_evals
        diversity_threshold = 0.05 * (ub - lb).mean()  # average dimension range scaled
        n_restarts = 0

        # Main loop
        while n_evals < max_evals:
            p = 0.2 - 0.1 * (n_evals / max_evals)  # pbest fraction

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

                # r2 from union of population and archive, ensure not i and not r1
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                # avoid picking i or r1 (if they are in union)
                while True:
                    r2 = np.random.randint(union.shape[0])
                    if union.shape[0] == pop.shape[0]:  # no archive
                        if r2 != i and r2 != r1:
                            break
                    else:
                        # archive individuals are not in pop; no need to exclude i/r1
                        break

                # pbest selection
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)

                # Sample F and CR from memory
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0.1, 1.0)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0.0, 1.0)

                # Mutation (current-to-pbest/1 with archive)
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2

                # Crossover (binomial)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # Bound handling: reflect then random reinit for remaining outliers
                trial = np.where(trial < lb, 2 * lb - trial, trial)
                trial = np.where(trial > ub, 2 * ub - trial, trial)
                mask = (trial < lb) | (trial > ub)
                if np.any(mask):
                    trial[mask] = lb[mask] + (ub[mask] - lb[mask]) * np.random.rand(np.sum(mask))

                # Evaluate
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    evals_since_last_improvement = 0
                else:
                    evals_since_last_improvement += 1

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
                w = np.array(delta_f)[sorted_order] / np.sum(delta_f)

                # Lehmer mean for F
                MF[memory_idx] = np.sum(w * S_F**2) / np.sum(w * S_F)
                # Weighted mean for CR
                MCR[memory_idx] = np.sum(w * S_CR**2) / np.sum(w * S_CR)
                memory_idx = (memory_idx + 1) % H

            # Non-linear population size reduction (keep larger pop earlier)
            N_new = round(N_init - (N_init - N_min) * (n_evals / max_evals)**0.8)
            if N_new < N:
                # Keep best N_new individuals
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                # Shrink archive if needed (archive_max = current pop size)
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Diversity measure for restart trigger
            if N >= 2:
                centroid = pop.mean(axis=0)
                distances = np.sqrt(((pop - centroid)**2).sum(axis=1)).mean()
                low_diversity = distances < diversity_threshold
            else:
                low_diversity = False

            # Restart if stagnation or low diversity and enough budget remains
            if (evals_since_last_improvement > restart_threshold or low_diversity) and n_evals < max_evals * 0.8:
                # Keep the best solution, reinitialize rest with guided randomization
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]

                # Generate new population: half around best with Gaussian perturbation, half uniform random
                new_N = N  # keep current size
                pop_new = np.empty((new_N, dim))
                fitness_new = np.full(new_N, np.inf)
                # Keep best
                pop_new[0] = best_ind
                fitness_new[0] = best_fit
                # Gaussian perturbations around best (scale = 0.1 * range)
                scale = 0.1 * (ub - lb)
                n_gauss = (new_N - 1) // 2
                for j in range(1, n_gauss+1):
                    trial = best_ind + scale * np.random.randn(dim)
                    trial = np.clip(trial, lb, ub)
                    pop_new[j] = trial
                    fitness_new[j] = func(trial)
                    n_evals += 1
                    if fitness_new[j] < self.f_opt:
                        self.f_opt = fitness_new[j]
                        self.x_opt = trial.copy()
                # Uniform random for the rest
                for j in range(n_gauss+1, new_N):
                    trial = lb + (ub - lb) * np.random.rand(dim)
                    pop_new[j] = trial
                    fitness_new[j] = func(trial)
                    n_evals += 1
                    if fitness_new[j] < self.f_opt:
                        self.f_opt = fitness_new[j]
                        self.x_opt = trial.copy()

                pop = pop_new
                fitness = fitness_new
                # Reset archive and memory
                archive = np.empty((0, dim))
                archive_max = new_N
                MF[:] = 0.5
                MCR[:] = 0.8
                memory_idx = 0
                evals_since_last_improvement = 0
                n_restarts += 1
                # Reevaluate diversity threshold after restart
                if N >= 2:
                    centroid = pop.mean(axis=0)
                    distances = np.sqrt(((pop - centroid)**2).sum(axis=1)).mean()
                    diversity_threshold = max(0.02 * (ub - lb).mean(), distances * 0.5)  # adapt slightly

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt