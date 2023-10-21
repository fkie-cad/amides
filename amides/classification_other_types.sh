#!/bin/bash

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
