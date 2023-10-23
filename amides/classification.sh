#!/bin/bash

# Creating misuse classification results for 'process_creation' rules using SOCBED data
python3 bin/train.py --config bin/config/process_creation/train_misuse_svc_rules.json
python3 bin/train.py --config bin/config/process_creation/train_misuse_svc_matches.json
python3 bin/validate.py --config bin/config/process_creation/validate_misuse_svc_rules.json
python3 bin/validate.py --config bin/config/process_creation/validate_misuse_svc_matches.json
python3 bin/eval_mcc_scaling.py --config bin/config/process_creation/eval_misuse_svc_rules.json
python3 bin/eval_mcc_scaling.py --config bin/config/process_creation/eval_misuse_svc_matches.json
python3 bin/plot_pr.py --config bin/config/process_creation/prt_plot_misuse_rules_matches.json
