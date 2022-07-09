#!/bin/bash

if [ $1 = 's' ]; then
  # pytest -rA -v tests/test_service_rate_inspector.py
  # pytest -rA -v tests/test_service_rate_inspector.py -k "test_is_in_cap_region"
  pytest -rA -v tests/test_service_rate_inspector.py -k "test_min_cost_dist"

elif [ $1 = 't' ]; then
  pytest -rA -v tests/test_service_rate_inspector_on_redundancy_w_two_xors.py

else
  echo "Unexpected arg= $1"
fi
