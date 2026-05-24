import numpy as np

class AdaptiveDEWithEnhancedLS:
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
        N_min = max(4, int(dim / 5))
        N = N_init

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
        archive_max = N

        # Success-history memory
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation & diversity
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals
        diversity_threshold = 0.02  # normalized average distance to best

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0
        improvement_budget = 0

        # Helper: compute diversity (mean distance to best)
        def compute_diversity():
            if N < 2:
                return 0.0
            best_idx = np.argmin(fitness)
            best = pop[best_idx]
            dists = np.mean(np.abs(pop - best), axis=1)
            return np.mean(dists) / np.mean(ub - lb)

        # Enhanced pattern search with parabolic interpolation
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)
            iterations = 0
            used = 0
            while used < max_local_evals and iterations < dim * 6:
                iterations += 1
                improved = False
                # Coordinate search
                for d in range(dim):
                    if used >= max_local_evals - 2:
                        break
                    # Positive direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] + step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        # Parabolic interpolation (3 points: pos, new_pos, additional)
                        mid = (pos[d] + new_pos[d]) / 2.0
                        mid_pos = pos.copy()
                        mid_pos[d] = np.clip(mid, lb[d], ub[d])
                        mid_val = func(mid_pos)
                        used += 1
                        # Fit parabola: points: a=pos[d], b=new_pos[d], c=mid
                        a, b, c = pos[d], new_pos[d], mid
                        fa, fb, fc = val, new_val, mid_val
                        # Find minimum of parabola if a!=b and fa,fb,fc known
                        denom = (a - b) * (a - c) * (b - c)
                        if abs(denom) > 1e-12:
                            x_min = (0.5 * ((a**2 - b**2) * fc + (b**2 - c**2) * fa + (c**2 - a**2) * fb) / denom)
                            x_min = np.clip(x_min, lb[d], ub[d])
                            if abs(x_min - pos[d]) > 1e-12 and abs(x_min - new_pos[d]) > 1e-12:
                                cand = pos.copy()
                                cand[d] = x_min
                                cand_val = func(cand)
                                used += 1
                                if cand_val < new_val:
                                    new_pos = cand.copy()
                                    new_val = cand_val
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # Negative direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        mid = (pos[d] + new_pos[d]) / 2.0
                        mid_pos = pos.copy()
                        mid_pos[d] = np.clip(mid, lb[d], ub[d])
                        mid_val = func(mid_pos)
                        used += 1
                        a, b, c = pos[d], new_pos[d], mid
                        fa, fb, fc = val, new_val, mid_val
                        denom = (a - b) * (a - c) * (b - c)
                        if abs(denom) > 1e-12:
                            x_min = (0.5 * ((a**2 - b**2) * fc + (b**2 - c**2) * fa + (c**2 - a**2) * fb) / denom)
                            x_min = np.clip(x_min, lb[d], ub[d])
                            if abs(x_min - pos[d]) > 1e-12 and abs(x_min - new_pos[d]) > 1e-12:
                                cand = pos.copy()
                                cand[d] = x_min
                                cand_val = func(cand)
                                used += 1
                                if cand_val < new_val:
                                    new_pos = cand.copy()
                                    new_val = cand_val
                        pos = new_pos
                        val = new_val
                        improved = True
                # Random orthogonal direction search (improves on rotated functions)
                if used < max_local_evals - 3 and not improved:
                    d = np.random.uniform(-1, 1, dim)
                    d = d / (np.linalg.norm(d) + 1e-30)
                    step_vec = step_size * d
                    new_pos = np.clip(pos + step_vec, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        # Try a bigger step in the same direction
                        step_vec2 = step_size * d * 1.5
                        new_pos2 = np.clip(pos + step_vec2, lb, ub)
                        new_val2 = func(new_pos2)
                        used += 1
                        if new_val2 < val:
                            pos = new_pos2
                            val = new_val2
                if improved:
                    # Pattern move: accelerate along direction of improvement
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # Expand step size
                    step_size *= 1.2
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # Contract step size
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # Boundary handling: reflection
                for _ in range(10):
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, 2 * lb - trial, trial)
                    trial = np.where(out_high, 2 * ub - trial, trial)
                trial = np.clip(trial, lb, ub)
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1

                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            pop = new_pop
            fitness = new_fitness

            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population reduction
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 2
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Local search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Diversity check and restart
            diversity = compute_diversity()
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8) or (diversity < diversity_threshold and n_evals < max_evals * 0.8):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Generate new population: half opposition, half random, plus best
                new_pop = np.empty((new_N, dim))
                mid = (lb + ub) / 2.0
                for j in range(new_N):
                    if j == 0:
                        new_pop[0] = best_ind
                    elif j < new_N // 2:
                        # Opposition from random point
                        rand_point = lb + np.random.uniform(0, 1, dim) * (ub - lb)
                        opp = mid - (rand_point - mid)  # reflection about center
                        new_pop[j] = np.clip(opp, lb, ub)
                    else:
                        new_pop[j] = lb + np.random.uniform(0, 1, dim) * (ub - lb)
                pop = new_pop
                fitness = np.full(new_N, np.inf)
                fitness[0] = best_fit
                for j in range(1, new_N):
                    fitness[j] = func(pop[j])
                    n_evals += 1
                    if fitness[j] < self.f_opt:
                        self.f_opt = fitness[j]
                        self.x_opt = pop[j].copy()
                N = new_N
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt