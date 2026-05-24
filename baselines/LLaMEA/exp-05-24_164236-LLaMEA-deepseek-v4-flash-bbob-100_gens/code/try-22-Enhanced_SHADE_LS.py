import numpy as np

class Enhanced_SHADE_LS:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed()
        lb = np.full(self.dim, -5.0)
        ub = np.full(self.dim,  5.0)
        dim = self.dim
        budget = self.budget

        # ----- Population size (L-SHADE style) -----
        N_init = max(5, int(18 * np.sqrt(dim)))  # slightly larger initial
        N_min = 4
        pop_size = N_init
        max_gen = int(budget / pop_size * 1.5)   # more generations

        # ----- SHADE memory -----
        mem_size = 6
        mem_F  = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # ----- Strategy memory (for ensemble) -----
        # 0: current-to-pbest/1, 1: rand/1, 2: best/1
        strategy_prob = np.array([0.6, 0.3, 0.1])
        strategy_success = np.ones(3)
        strategy_attempts = np.ones(3)

        # ----- Latin Hypercube Sampling -----
        lhs = np.random.rand(pop_size, dim)
        for j in range(dim):
            lhs[:, j] = (np.argsort(lhs[:, j]) + 0.5) / pop_size
        pop = lb + lhs * (ub - lb)

        # ----- Evaluate initial population -----
        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # ----- Archive (set of parents replaced) -----
        archive = []
        archive_size = pop_size

        # ----- Tracking -----
        best_old = self.f_opt
        stagnation = 0
        gen = 0

        # ----- Main loop -----
        while evals < budget:
            gen += 1

            # ----- Linear population size reduction -----
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            # ----- pbest rate (time-dependent) -----
            p = 0.2 * (gen / max_gen) ** 1.5 + 0.1   # more aggressive early
            p = min(p, 0.5)

            success_F  = []
            success_CR = []
            weight     = []

            for i in range(pop_size):
                if evals >= budget:
                    break

                # ----- Select strategy based on success rates -----
                strat = np.random.choice(3, p=strategy_prob)

                # ----- Sample F and CR from memory -----
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # ----- pbest selection -----
                pbest_size = max(2, int(p * pop_size))
                best_idx = np.argsort(fitness)[:pbest_size]
                pbest = pop[np.random.choice(best_idx)]

                # ----- Choose r1, r2 from union of pop and archive (exclude i) -----
                union_indices = list(range(pop_size)) + list(range(len(archive)))
                union_indices.remove(i)
                if len(union_indices) >= 2:
                    r1, r2 = np.random.choice(union_indices, 2, replace=False)
                    def get_ind(idx):
                        return pop[idx] if idx < pop_size else archive[idx - pop_size]
                    x1 = get_ind(r1)
                    x2 = get_ind(r2)
                else:
                    alt = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(alt, 2, replace=False)
                    x1, x2 = pop[r1], pop[r2]

                # ----- Mutation -----
                if strat == 0:  # current-to-pbest/1
                    mutant = pop[i] + F * (pbest - pop[i]) + F * (x1 - x2)
                elif strat == 1:  # rand/1
                    mutant = x1 + F * (x2 - pop[np.random.choice([j for j in range(pop_size) if j != i])])
                else:  # best/1
                    best_vec = pop[np.argmin(fitness)]
                    mutant = best_vec + F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                # ----- Crossover: binomial -----
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])

                # ----- Evaluate trial -----
                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # ----- Add parent to archive (replace if full) -----
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # Replace the archive member with worst fitness
                        # (re-evaluate fitness of archive needed? use original fitness? Not stored.)
                        # Simpler: replace the member with smallest distance to parent
                        dists = np.linalg.norm(np.array(archive) - pop[i], axis=1)
                        idx_remove = np.argmin(dists)
                        archive[idx_remove] = pop[i].copy()

                    success_F.append(F)
                    success_CR.append(CR)
                    imp = fitness[i] - f_trial
                    weight.append(max(imp, 1e-12))

                    # Update strategy success count
                    strategy_success[strat] += 1
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                else:
                    strategy_attempts[strat] += 1

            # ----- Update memory (weighted Lehmer for F, weighted mean for CR) -----
            if len(success_F) > 0:
                w = np.array(weight)
                w /= (np.sum(w) + 1e-30)
                F_lehmer  = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean   = np.sum(w * np.array(success_CR))
                mem_F[mem_idx]  = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ----- Update strategy probabilities (softmax) -----
            strategy_prob = (strategy_success + 1e-6) / (strategy_attempts + 1e-6)
            strategy_prob /= np.sum(strategy_prob)

            # ----- Local search: Bidirectional random search (BRS) -----
            # Allocate a small budget (adaptive)
            remaining = budget - evals
            ls_budget = int(0.08 * remaining)  # 8% of remaining
            if ls_budget >= dim + 1 and (gen % 3 == 0 or stagnation >= 4):
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                # Adaptive step size
                step = 0.1 * (1 - gen / max_gen) + 0.01
                used = 0
                while used < ls_budget and evals < budget:
                    # Sample a random direction
                    d = np.random.randn(dim)
                    d /= (np.linalg.norm(d) + 1e-30)
                    # Try forward
                    x_plus = np.clip(x_best + step * d, lb, ub)
                    f_plus = func(x_plus)
                    evals += 1; used += 1
                    # Try backward
                    x_minus = np.clip(x_best - step * d, lb, ub)
                    f_minus = func(x_minus)
                    evals += 1; used += 1
                    # Best direction
                    if f_plus < f_best or f_minus < f_best:
                        if f_plus < f_minus:
                            x_best = x_plus
                            f_best = f_plus
                        else:
                            x_best = x_minus
                            f_best = f_minus
                        # Double step size on success
                        step = min(step * 1.1, 1.0)
                    else:
                        # Reduce step size
                        step = step * 0.8
                    # Update global best
                    if f_best < self.f_opt:
                        self.f_opt = f_best
                        self.x_opt = x_best.copy()

                # Inject best into population if better than worst
                worst_idx = np.argmax(fitness)
                if self.f_opt < fitness[worst_idx]:
                    pop[worst_idx] = self.x_opt.copy()
                    fitness[worst_idx] = self.f_opt

            # ----- Stagnation detection and restart -----
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation = 0
            else:
                stagnation += 1

            if stagnation > max(8, int(0.06 * max_gen)):
                # Restart: replace 70% of population with local + global samples
                n_restart = max(1, int(0.7 * pop_size))
                x_best = self.x_opt.copy()
                # Use Sobol-like sequence (fallback to LHS)
                try:
                    from scipy.stats import qmc
                    sampler = qmc.Sobol(d=dim, scramble=True)
                    sob = sampler.random(n_restart)
                except:
                    sob = np.random.rand(n_restart, dim)
                    for j in range(dim):
                        sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.05 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = x_best + np.random.randn(dim) * scale
                    else:
                        pop[idx] = lb + sob[idx] * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                mem_idx = 0
                strategy_success[:] = 1.0
                strategy_attempts[:] = 1.0
                strategy_prob = np.array([0.6, 0.3, 0.1])
                archive.clear()
                stagnation = 0

        return self.f_opt, self.x_opt