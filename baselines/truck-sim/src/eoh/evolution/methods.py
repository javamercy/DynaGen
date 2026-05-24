from ..eoh_src.eoh import EOH
from .selection import prob_rank, equal, tournament, roulette_wheel
from .management import ls_sa, ls_greedy, pop_greedy


class Methods:
    def __init__(self, paras, problem) -> None:
        self.paras = paras      
        self.problem = problem

        match paras.selection:
            case "prob_rank":
                self.select = prob_rank
            case "equal":
                self.select = equal
            case "roulette_wheel":
                self.select = roulette_wheel
            case "tournament":
                self.select = tournament
            case _:
                print(f"selection method {paras.selection} has not been implemented!")
                exit()

        match paras.management:
            case "pop_greedy":
                self.manage = pop_greedy
            case "ls_greedy":
                self.manage = ls_greedy
            case "ls_sa":
                self.manage = ls_sa
            case _:
                print("management method "+paras.management+" has not been implemented !")
                exit()

    def get_method(self):
        return EOH(self.paras, self.problem, self.select, self.manage)
