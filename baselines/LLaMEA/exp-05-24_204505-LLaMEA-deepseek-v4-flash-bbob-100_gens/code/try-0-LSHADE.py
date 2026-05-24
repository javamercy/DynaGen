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
        
        # Population size initial and minimum
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = 4
        N = N_init
        
        # Initialize population with Latin Hypercube Sampling
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
        
        # Archive (set of inferior solutions)
        archive = np.empty((0, dim))
        archive_max = N_init
        
        # Success-history memory
        H = 5
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.5
        memory_idx = 0
        
        # Main loop
        while n_evals < max_evals:
            # pbest proportion: linearly decreasing from 0.2 to 0.1
            p = 0.2 - 0.1 * (n_evals / max_evals)
            
            # Generate offspring
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []
            
            for i in range(N):
                # Choose r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                
                # Choose r2 from union of population and archive
                union = np.vstack((pop, archive)) if archive.size > 0 else pop
                union_indices = list(range(union.shape[0]))
                r2 = np.random.choice(union_indices)
                
                # pbest selection: pick top p*N individuals
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)
                
                # Select memory entry (random from history)
                mem = np.random.randint(H)
                F = np.clip(np.random.cauchy(MF[mem], 0.1), 0, 1)
                while F <= 0:
                    F = np.clip(np.random.cauchy(MF[mem], 0.1), 0, 1)
                CR = np.clip(np.random.normal(MCR[mem], 0.1), 0, 1)
                
                # Mutation (current-to-pbest/1 with archive)
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                
                # Crossover (binomial)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                
                # Bound handling: mirror
                trial = np.where(trial < lb, lb + (lb - trial), trial)
                trial = np.where(trial > ub, ub - (trial - ub), trial)
                
                # Evaluate
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                
                if trial_f < fitness[i]:
                    # Success: record
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    
                    # Add the replaced individual to archive
                    archive = np.vstack((archive, pop[i].reshape(1,-1)))
                    if archive.shape[0] > archive_max:
                        # Random removal
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)
            
            # Update population and fitness
            pop = new_pop
            fitness = new_fitness
            
            # Update memories if successful candidates exist
            if len(S_F) > 0:
                # Sort by delta_f descending
                sorted_indices = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_indices]
                S_CR = np.array(S_CR)[sorted_indices]
                w = np.array(delta_f)[sorted_indices] / np.sum(delta_f)
                
                # Lehmer mean for F
                MF[memory_idx] = np.sum(w * S_F**2) / np.sum(w * S_F)
                # Weighted mean for CR
                MCR[memory_idx] = np.sum(w * S_CR**2) / np.sum(w * S_CR)
                memory_idx = (memory_idx + 1) % H
            
            # Linear population size reduction
            N_new = round(N_init - (N_init - N_min) * (n_evals / max_evals))
            if N_new != N:
                # Sort by fitness and keep best N_new individuals
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive = archive[: min(archive_max, archive.shape[0])]  # maybe reduce archive? Keep same size
                N = N_new
            
            # Check termination condition (budget exhausted)
            if n_evals >= max_evals:
                break
        
        return self.f_opt, self.x_opt