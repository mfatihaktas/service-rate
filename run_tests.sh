#!/bin/bash

PYTEST="pytest -rA -v --color=yes"

if [ $1 = "s" ]; then
  # ${PYTEST} tests/test_service_rate_inspector.py
  # ${PYTEST} tests/test_service_rate_inspector.py -k "test_is_in_cap_region"
  # ${PYTEST} tests/test_service_rate_inspector.py -k "test_plot_cap_2d"
  # ${PYTEST} tests/test_service_rate_inspector.py -k "test_min_max_functions"

  ${PYTEST} "tests/service_rate/test_plot_capacity_region.py::test_plot_cost_capacity_region_2d_alternative"

elif [ $1 = "x" ]; then
  ${PYTEST} tests/test_service_rate_inspector_on_redundancy_w_two_xors.py

elif [ $1 = "c" ]; then
  ${PYTEST} tests/test_service_rate_inspector_w_csv.py

elif [ $1 = "l" ]; then
  # ${PYTEST} tests/test_load_across_nodes.py
  ${PYTEST} tests/test_load_across_nodes.py -k "test_load_across_nodes"
  # ${PYTEST} tests/test_load_across_nodes.py -k "test_load_on_first_node"

elif [ $1 = "i" ]; then
  # ${PYTEST} "tests/test_integrality_theorem.py::test_integer_programming_w_or_constraint"
  ${PYTEST} "tests/test_integrality_theorem.py::test_w_integer_programming_refined"
  # ${PYTEST} "tests/test_integrality_theorem.py::test_w_integer_programming_2"
  # ${PYTEST} "tests/test_integrality_theorem.py::test_mixed_integer_quadratic_program"

elif [ $1 = "o" ]; then
  ${PYTEST} "tests/test_storage_optimizer.py::test_StorageOptimizerReplication"

elif [ $1 = "p" ]; then
  ${PYTEST} "tests/test_popularity.py::test_PopModel_wZipf"

elif [ $1 = "m" ]; then
  ${PYTEST} "tests/storage_overlap/math_utils.py::test_prob_cum_demand_leq_cum_supply"

else
  echo "Unexpected arg= $1"
fi
