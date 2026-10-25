#!/bin/bash
# Download the official MMAR / MMSU scoring scripts at the commits used for the paper.
# (These upstream repos have no code license, so the scripts are fetched rather than vendored.
#  The MMAU scorer is Apache-2.0 and is included in MMAU/code/.)
set -e
cd "$(dirname "$0")"
curl -fsSL -o MMAR/code/evaluation.py \
  https://raw.githubusercontent.com/ddlBoJack/MMAR/3bce090a967db576c8ad433b290a8da582d6d2a0/code/evaluation.py
curl -fsSL -o MMSU/code/mmsu_evaluation.py \
  https://raw.githubusercontent.com/dingdongwang/MMSU/217d1d3f5ad59203ed16f44c0fd913f4f8add2b4/mmsu_evaluation.py
echo "Fetched MMAR/code/evaluation.py and MMSU/code/mmsu_evaluation.py"
