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

        # Archive for inferior solutions (size = current population size)
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory parameters
        H = 20
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.5
        memory_idx = 0

        # Stagnation and diversity detection
        best_since_last_restart = self.f_opt
        evals_since_last_improvement = 0
        restart_threshold = 0.1 * max_evals
        n_restarts = 0

        # Main loop
        while n_evals < max_evals:
            p = 0.15 + 0.15 * (1 - n_evals / max_evals)  # from 0.3 to 0.15

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

                # Crossover (binomial)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # Bound handling: bounce-back (improved)
                for j in range(dim):
                    if trial[j] < lb:
                        trial[j] = lb + np.random.random() * (base[j] - lb)
                    elif trial[j] > ub:
                        trial[j] = ub - np.random.random() * (ub - base[j])
                # Ensure within bounds (safety)
                trial = np.clip(trial, lb, ub)

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
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Restart triggers: stagnation or low diversity
            do_restart = False
            if evals_since_last_improvement > restart_threshold and n_evals < max_evals * 0.8:
                do_restart = True
            # Also check diversity (standard deviation of normalized positions)
            if not do_restart and n_evals > 0.3 * max_evals:
                norm_pop = (pop - lb) / (ub - lb)
                diversity = np.mean(np.std(norm_pop, axis=0))
                if diversity < 0.05:  # very low diversity
                    do_restart = True
            if do_restart:
                restarts_remaining = max_evals - n_evals
                if restarts_remaining > N_init * 0.5:
                    best_idx = np.argmin(fitness)
                    best_ind = pop[best_idx].copy()
                    best_fit = fitness[best_idx]
                    # Reinitialize population (size N) with Latin hypercube, keep best
                    new_N = N
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                    pop = samples.copy()
                    fitness = np.full(new_N, np.inf)
                    pop[0] = best_ind
                    fitness[0] = best_fit
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    # Reset archive and memory
                    archive = np.empty((0, dim))
                    archive_max = new_N
                    MF[:] = 0.5
                    MCR[:] = 0.5
                    memory_idx = 0
                    evals_since_last_improvement = 0
                    n_restarts += 1

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt