#!/bin/bash

if [ $1 = 's' ]; then
  # pytest -rA -v tests/test_service_rate_inspector.py -k "test_is_in_cap_region"
  pytest -rA -v tests/test_service_rate_inspector.py --show-capture=stderr

elif [ $1 = '?' ]; then
  pytest ?

else
  echo "Unexpected arg= $1"
fi
