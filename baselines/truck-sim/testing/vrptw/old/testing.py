import os
import json

from o1mini.oneshot1_0 import vrptw
# from baseline.ortool import vrptw

if __name__ == '__main__':
    with open(os.path.join('', 'testing.json'), 'r', encoding='utf8') as f:
        test_samples = json.load(f)

    for sample in test_samples:
        output = vrptw(**sample['input'])
        print(output)
        print(sample['output'])

        print("\n\n\n")
