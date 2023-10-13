#!/bin/bash

# Creating misuse classification results for 'process_creation' rules using SOCBED data
python3 bin/train.py --config bin/config/process_creation/train_misuse_svc_rules.json
python3 bin/train.py --config bin/config/process_creation/train_misuse_svc_matches.json
python3 bin/validate.py --config bin/config/process_creation/validate_misuse_svc_rules.json
python3 bin/validate.py --config bin/config/process_creation/validate_misuse_svc_matches.json
python3 bin/eval_mcc_scaling.py --config bin/config/process_creation/eval_misuse_svc_rules.json
python3 bin/eval_mcc_scaling.py --config bin/config/process_creation/eval_misuse_svc_matches.json
python3 bin/plot_pr.py --config bin/config/process_creation/prt_plot_misuse_rules_matches.json

# Creating rule attribution results for 'process_creation' rules using SOCBED data
cat models/process_creation/train_rslt_misuse_svc_rules_f1_0_info.json | jq .estimator_params > bin/config/process_creation/params.json
python3 bin/train.py --config bin/config/process_creation/train_attr_svc_rules.json
python3 bin/eval_attr.py --config bin/config/process_creation/eval_attr.json
python3 bin/plot_attr.py --config bin/config/process_creation/attr_plot.json

# Creating tainted classification results for 'process_creation' events using SOCBED data
python3 bin/train.py --config bin/config/process_creation/train_misuse_svc_rules_tainted_10.json
python3 bin/train.py --config bin/config/process_creation/train_misuse_svc_rules_tainted_20.json
python3 bin/train.py --config bin/config/process_creation/train_misuse_svc_rules_tainted_30.json
python3 bin/validate.py --config bin/config/process_creation/validate_misuse_svc_rules_tainted_10.json
python3 bin/validate.py --config bin/config/process_creation/validate_misuse_svc_rules_tainted_20.json
python3 bin/validate.py --config bin/config/process_creation/validate_misuse_svc_rules_tainted_30.json
python3 bin/eval_mcc_scaling.py --config bin/config/process_creation/eval_misuse_svc_rules_tainted.json
python3 bin/plot_multi_tainted.py --config bin/config/process_creation/pr_plot_tainted.json


# Creating misuse classification results for 'powershell', 'proxy_web', and 'registry' rules
python3 bin/train_new_types.py --config bin/config/powershell/train_misuse_svc_rules.json
python3 bin/train_new_types.py --config bin/config/proxy_web/train_misuse_svc_rules.json
python3 bin/train_new_types.py --config bin/config/registry/train_misuse_svc_rules.json

python3 bin/validate_new_types.py --config bin/config/powershell/validate_misuse_svc_rules.json
python3 bin/validate_new_types.py --config bin/config/proxy_web/validate_misuse_svc_rules.json
python3 bin/validate_new_types.py --config bin/config/registry/validate_misuse_svc_rules.json

python3 bin/eval_mcc_scaling.py --config bin/config/powershell/eval_misuse_svc_rules.json
python3 bin/eval_mcc_scaling.py --config bin/config/proxy_web/eval_misuse_svc_rules.json
python3 bin/eval_mcc_scaling.py --config bin/config/registry/eval_misuse_svc_rules.json
python3 bin/plot_pr.py --config bin/config/pr_plot_powershell_proxy_registry.json




