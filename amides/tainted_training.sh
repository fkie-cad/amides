#!/bin/bash

# Creating tainted classification results for 'process_creation' events using SOCBED data
python3 bin/train.py --config bin/config/process_creation/train_misuse_svc_rules_tainted_10.json
python3 bin/train.py --config bin/config/process_creation/train_misuse_svc_rules_tainted_20.json
python3 bin/train.py --config bin/config/process_creation/train_misuse_svc_rules_tainted_30.json
python3 bin/validate.py --config bin/config/process_creation/validate_misuse_svc_rules_tainted_10.json
python3 bin/validate.py --config bin/config/process_creation/validate_misuse_svc_rules_tainted_20.json
python3 bin/validate.py --config bin/config/process_creation/validate_misuse_svc_rules_tainted_30.json
python3 bin/eval_mcc_scaling.py --config bin/config/process_creation/eval_misuse_svc_rules_tainted.json
python3 bin/plot_multi_tainted.py --config bin/config/process_creation/pr_plot_tainted.json
