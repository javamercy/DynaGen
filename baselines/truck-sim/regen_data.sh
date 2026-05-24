#!/bin/bash

export PYTHONPATH=/home/koitu/git/truck-sim

#cd $PYTHONPATH
#cd testing/dvrp
#python gen_data.py

cd $PYTHONPATH
cd testing/dvrptw
python gen_data.py

#cd $PYTHONPATH
#cd testing/vrp
#python gen_data.py

cd $PYTHONPATH
cd testing/vrptw
python gen_data.py

