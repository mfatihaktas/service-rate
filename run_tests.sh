#!/bin/bash

if [ $1 = 's' ]; then
  # pytest -rA -v tests/test_service_rate_inspector.py
  # pytest -rA -v tests/test_service_rate_inspector.py -k "test_is_in_cap_region"
  # pytest -rA -v tests/test_service_rate_inspector.py -k "test_min_max_functions"
  pytest -rA -v tests/test_service_rate_inspector.py -k "test_w_frac_of_demand_vectors_in_cap_region"

elif [ $1 = 'x' ]; then
  pytest -rA -v tests/test_service_rate_inspector_on_redundancy_w_two_xors.py

elif [ $1 = 'c' ]; then
  pytest -rA -v tests/test_service_rate_inspector_w_csv.py

elif [ $1 = 'l' ]; then
  pytest -rA -v tests/test_load_across_nodes.py
  # pytest -rA -v tests/test_load_across_nodes.py -k "test_load_across_nodes"

else
  echo "Unexpected arg= $1"
fi
