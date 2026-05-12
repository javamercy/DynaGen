#!/bin/bash
python testing.py tsp_construct 3 -m gpt-3.5-turbo
python testing.py tsp_construct 3 -m gpt-4o
python testing.py tsp_construct 3 -m gpt-4-turbo

python testing.py vrp_construct 3 -m gpt-3.5-turbo
python testing.py vrp_construct 3 -m gpt-4o
python testing.py vrp_construct 3 -m gpt-4-turbo

python testing.py vrptw_construct 3 -m gpt-3.5-turbo
python testing.py vrptw_construct 3 -m gpt-4o
python testing.py vrptw_construct 3 -m gpt-4-turbo

python testing.py dvrptw_construct 3 -m gpt-3.5-turbo
python testing.py dvrptw_construct 3 -m gpt-4o
python testing.py dvrptw_construct 3 -m gpt-4-turbo

