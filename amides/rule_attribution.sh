#!/bin/bash

# Creating rule attribution results for 'process_creation' rules using SOCBED data
cat models/process_creation/train_rslt_misuse_svc_rules_f1_0_info.json | jq .estimator_params > bin/config/process_creation/params.json
python3 bin/train.py --config bin/config/process_creation/train_attr_svc_rules.json
python3 bin/eval_attr.py --config bin/config/process_creation/eval_attr.json
python3 bin/plot_attr.py --config bin/config/process_creation/attr_plot.json
