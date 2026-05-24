import os
import pickle


# TODO: merge this in the startup for testing.py (check that truck_num_scaling and the dataset selected match)

path = "/home/koitu/git/truck-sim/testing/dvrptw/testing_data_1"

if __name__ == '__main__':
    # problem_size = [10, 20, 50, 100, 200]
    # for size in problem_size:
    #     instance_file_name = os.path.join(path, f"instance_data_{size}.pkl")
    #     with open(instance_file_name, 'rb') as f:
    #         instance_dataset = pickle.load(f)
    #
    #     print(len(instance_dataset[0][-1]['routes']))

    path = "/home/koitu/git/truck-sim/testing/vrptw/training_data_1/instances.pkl"
    with open(path, "rb") as f:
        instances = pickle.load(f)
    for i in range(8):
        print(instances[0][-1]['missed'])

