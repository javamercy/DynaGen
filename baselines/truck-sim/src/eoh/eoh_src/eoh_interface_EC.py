import re
import time
import random
import warnings

import numpy as np
import concurrent.futures

from joblib import Parallel, delayed

from .eoh_evolution import Evolution
from .evaluator_accelerate import add_numba_decorator


class InterfaceEC:
    def __init__(self,
                 pop_size,
                 m,
                 api_endpoint,
                 api_key,
                 llm_model,
                 debug_mode,
                 interface_prob,
                 select,
                 n_p,
                 timeout,
                 use_numba):

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(
            api_endpoint,
            api_key,
            llm_model,
            debug_mode,
            prompts)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        
        self.timeout = timeout
        self.use_numba = use_numba
        
    def code2file(self, code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    # def population_management(self,pop):
    #     # Delete the worst individual
    #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
    #     return pop_new
    
    # def parent_selection(self,pop,m):
    #     ranks = [i for i in range(len(pop))]
    #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    #     parents = random.choices(pop, weights=probs, k=m)
    #     return parents

    def population_generation(self, extended_init=False, strict_init=False):
        n_create = 3
        population = []

        for i in range(n_create):
            # print(f"init algorithm {i+1} / {n_create}")
            _, pop = self.get_algorithm([],'i1', strict_init)
            for p in pop:
                population.append(p)

        if extended_init:
            # print(f"extended init algorithm {i+1} / {n_create}")
            _, pop = self.get_algorithm([],'i2', strict_init)

            if strict_init:
                for p in pop:
                    population.append(p)
            else:
                # randomly select only half the elements
                random.shuffle(pop)
                for p in pop[:len(pop)//2]:
                    population.append(p)

        return population
    
    def population_generation_seed(self, seeds, n_p):
        population = []
        all_res = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)
        fitness = [r[0] for r in all_res]
        results = [r[1] for r in all_res]

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initialization finished! Get "+str(len(seeds))+" seed algorithms")

        return population
    

    def _get_alg(self, pop, operator):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None,
        }

        match operator:
            case "i1":
                parents = None
                [offspring['code'], offspring['algorithm']] =  self.evol.i1()
            case "i2":
                parents = None
                [offspring['code'], offspring['algorithm']] =  self.evol.i2()
            case "e1":
                parents = self.select.parent_selection(pop, self.m)
                [offspring['code'], offspring['algorithm']] = self.evol.e1(parents)
            case "e2":
                parents = self.select.parent_selection(pop, self.m)
                [offspring['code'], offspring['algorithm']] = self.evol.e2(parents)
            case "m1":
                parents = self.select.parent_selection(pop, 1)
                [offspring['code'], offspring['algorithm']] = self.evol.m1(parents[0])
            case "m2":
                parents = self.select.parent_selection(pop, 1)
                [offspring['code'], offspring['algorithm']] = self.evol.m2(parents[0])
            case "m3":
                parents = self.select.parent_selection(pop, 1)
                [offspring['code'], offspring['algorithm']] = self.evol.m3(parents[0])
            case _:
                print(f"Evolution operator [{operator}] has not been implemented ! \n")
                exit()

        return parents, offspring

    def get_offspring(self, pop, operator, strict=False):
        try:
            for _ in range(3):
                p, offspring = self._get_alg(pop, operator)

                if self.use_numba:
                    # Regular expression pattern to match function definitions
                    pattern = r"def\s+(\w+)\s*\(.*\):"

                    # Search for function definitions in the code
                    match = re.search(pattern, offspring['code'])
                    function_name = match.group(1)

                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']

                if self.check_duplicate(pop, offspring['code']):
                    if self.debug:
                        print("duplicated code, wait 1 second and retrying ... ")
                        time.sleep(1)
                else:
                    break

            #self.code2file(offspring['code'])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.interface_eval.evaluate, code)
                fitness, results = future.result(timeout=self.timeout)
                offspring['objective'] = np.round(fitness, 5)
                future.cancel()        
                # fitness = self.interface_eval.evaluate(code)

            print(results['missed'])
            if strict and ('missed' in results) and sum(results['missed']) > 0:
                p = None
                offspring = {
                    'algorithm': None,
                    'code': None,
                    'objective': None,
                    'other_inf': None
                }

        except:
            p = None
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }

        # Round the objective values
        return p, offspring


    # def process_task(self,pop, operator):
    #     result =  None, {
    #             'algorithm': None,
    #             'code': None,
    #             'objective': None,
    #             'other_inf': None
    #         }
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future = executor.submit(self.get_offspring, pop, operator)
    #         try:
    #             result = future.result(timeout=self.timeout)
    #             future.cancel()
    #             #print(result)
    #         except:
    #             future.cancel()
                
    #     return result

    def get_algorithm(self, pop, operator, strict=False):
        results = []
        try:
            results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(
                delayed(self.get_offspring)(pop, operator, strict) for _ in range(self.pop_size))
        except Exception as e:
            if self.debug:
                print(f"Error: {e}")
            print("Parallel time out .")
            
        time.sleep(2)

        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off

    # def get_algorithm(self,pop,operator, pop_size, n_p):
    #     # perform it pop_size times with n_p processes in parallel
    #     p,offspring = self._get_alg(pop,operator)
    #     while self.check_duplicate(pop,offspring['code']):
    #         if self.debug:
    #             print("duplicated code, wait 1 second and retrying ... ")
    #         time.sleep(1)
    #         p,offspring = self._get_alg(pop,operator)
    #     self.code2file(offspring['code'])
    #     try:
    #         fitness= self.interface_eval.evaluate()
    #     except:
    #         fitness = None
    #     offspring['objective'] =  fitness
    #     #offspring['other_inf'] =  first_gap
    #     while (fitness == None):
    #         if self.debug:
    #             print("warning! error code, retrying ... ")
    #         p,offspring = self._get_alg(pop,operator)
    #         while self.check_duplicate(pop,offspring['code']):
    #             if self.debug:
    #                 print("duplicated code, wait 1 second and retrying ... ")
    #             time.sleep(1)
    #             p,offspring = self._get_alg(pop,operator)
    #         self.code2file(offspring['code'])
    #         try:
    #             fitness= self.interface_eval.evaluate()
    #         except:
    #             fitness = None
    #         offspring['objective'] =  fitness
    #         #offspring['other_inf'] =  first_gap
    #     offspring['objective'] = np.round(offspring['objective'],5) 
    #     #offspring['other_inf'] = np.round(offspring['other_inf'],3)
    #     return p,offspring
