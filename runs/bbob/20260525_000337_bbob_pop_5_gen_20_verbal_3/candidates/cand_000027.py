import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        rng = self.rng
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        
        # PSO parameters
        pop_size = max(10, 2 * dim)
        w = 0.9  # inertia weight
        c1 = 2.0  # cognitive
        c2 = 2.0  # social
        v_max = (ub - lb) / 3.0  # max velocity
        
        # Initialize swarm
        pop = rng.uniform(lb, ub, (pop_size, dim))
        vel = rng.uniform(-v_max, v_max, (pop_size, dim))
        pbest = pop.copy()
        pbest_val = np.full(pop_size, np.inf)
        gbest_x = None
        gbest_val = np.inf
        evals = 0
        
        # Evaluate initial population
        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            pbest_val[i] = val
            if val < gbest_val:
                gbest_val = val
                gbest_x = pop[i].copy()
                report_best(gbest_val, gbest_x)
        
        # Main loop
        patience = max(1, int(0.1 * budget))
        no_improve_evals = 0
        
        while evals < budget:
            # Check for restart
            if no_improve_evals >= patience:
                # Reinitialize all particles except the global best
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    pop[i] = rng.uniform(lb, ub, dim)
                    vel[i] = rng.uniform(-v_max, v_max, dim)
                    val = func(pop[i])
                    evals += 1
                    pbest_val[i] = val
                    if val < gbest_val:
                        gbest_val = val
                        gbest_x = pop[i].copy()
                        no_improve_evals = 0
                        report_best(gbest_val, gbest_x)
                no_improve_evals = 0
                continue
            
            # Update particles
            for i in range(pop_size):
                if evals >= budget:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (gbest_x - pop[i])
                vel[i] = np.clip(vel[i], -v_max, v_max)
                pop[i] = pop[i] + vel[i]
                pop[i] = np.clip(pop[i], lb, ub)
                val = func(pop[i])
                evals += 1
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = pop[i].copy()
                if val < gbest_val:
                    gbest_val = val
                    gbest_x = pop[i].copy()
                    no_improve_evals = 0
                    report_best(gbest_val, gbest_x)
                else:
                    no_improve_evals += 1
        
        return gbest_val, gbest_x