#!/bin/bash

PY=python3


if [ $1 = 'e' ]; then
  $PY exp.py

elif [ $1 = 'es' ]; then
  $PY exp_w_schedule.py

elif [ $1 = 'c' ]; then
  echo ""

else
  echo "Unexpected arg= $1"
fi
