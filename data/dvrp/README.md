# DVRP Data

This directory contains the DVRP train/test split used by the paper implementation.

- `train/instances.pkl`: size-50 training instances; Dynagen uses the first 8 instances.
- `test/instance_data_*.pkl`: 64 held-out instances for each size 10, 20, 50, 100, and 200.

The pickle records are `(coordinates, arrival_times, ortools_reference)`. Dynagen evaluates generated policies online using only currently available customers. The primary DVRP score is TTT: the last-truck return time, matching the paper's maximum route-duration definition. The OR-Tools `max_distance` reference is used only to report percentage gaps against that TTT reference.
