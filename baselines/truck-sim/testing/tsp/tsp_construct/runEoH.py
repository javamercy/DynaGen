import pickle

from src.eoh import eoh
from src.eoh.utils.getParas import Paras


if __name__ == '__main__':
    # Parameter initialization
    paras = Paras()

    # Set parameters
    api_key = ""
    paras.set_paras(method = "eoh",
                    problem = "tsp_construct",
                    llm_api_endpoint = "api.openai.com",
                    llm_api_key = api_key,
                    llm_model = "gpt-3.5-turbo",
                    ec_pop_size = 5, # number of samples in each population
                    ec_n_pop = 10,  # number of populations
                    exp_n_proc = 4,  # multi-core parallel
                    exp_debug_mode = False)

    # initialization
    evolution = eoh.EVOL(paras)

    n_test_ins = 8
    instance_file_name = f"../training_data/instances.pkl"
    with open(instance_file_name, 'rb') as f:
        instance_dataset = pickle.load(f)

    # run
    evolution.run(instance_dataset, 50, n_test_ins)
