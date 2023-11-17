<h1 align="left">Adaptive Misuse Detection System (AMIDES)</h1>

The Adaptive Misuse Detection System (AMIDES) extends conventional rule matching of SIEM systems by applying machine learning components that aim to detect attacks evading existing SIEM rules as well as otherwise undetected attack variants. It learns from SIEM rules and historical benign events and can thus estimate which SIEM rule was tried to be evaded. A brief overview of AMIDES is given in [Overview](#overview).

This repository contains the source code of the `amides` Python package. The package contains the modules and scripts that enable to train and validate models for AMIDES, evaluate the model's classification performance, and create meaningful visualizations that help users to assess the evaluation results. Additionally, the repository contains initial training and validation data that enables to build and evaluate models for AMIDES.

For operational use, AMIDES is integrated into [Logprep](https://logprep.readthedocs.io/en/latest/user_manual/configuration/processor.html#amides), a pipeline-based log message preprocessor also written in Python. The package also contains additional scripts that help to prepare models for the operational use with Logprep. For more information on how to prepare AMIDES models for Logprep, please read [here](#preparing-models-for-logprep).

## Overview

![amides_architecture](./docs/amides.png)

AMIDES is trained using a set of SIEM detection rules and historical benign events taken from an organization's corporate network.
During operation, incoming events are passed to the rule matching component and the feature extraction component, which transforms the events into feature vectors. The features required for vectorization have been learned during the training phase. The feature vectors are then passed to the misuse classification component, which classifies events as malicious or benign. In case of a malicious result, the feature vector is passed to the rule attribution component, which generates a ranked list of SIEM rules potentially evaded by the event. In the final step, potential alerts of the rule matching and both machine learning components are merged into a single alert by the alert generation component.

## System Requirements

AMIDES was developed and tested on Linux using Python 3.10. Before attempting to use `amides`, make sure you have

- Physical or virtual host with a Linux-based OS
- A minimum of 8 GB of RAM
- At least 2 GB of HDD space
- Python 3.10 (or newer)
- jq

The repository contains a `Dockerfile` that creates a quickstart environment for the `amides` package. For testing purposes, we highly recommend to use the quickstart environment. Building and using the environment has been tested with `docker 20.10`.

## Accessing Code and Initial Data

In order to access source code and initial data, change into the target location on your system and clone the repository by executing

```bash
git clone https://github.com/fkie-cad/amides.git
```

or

```bash
git clone git@github.com:fkie-cad/amides.git 
```

in case you prefer to use SSH.

Alternatively, you can get the repository by downloading the `.zip` file from the repository's main page and unpack it into your target location.

The `amides` package is located in the `amides` directory. Initial data to train and validate models for AMIDES is provided in the `data` directory.

### SOCBED Datasets

The [SOCBED](https://github.com/fkie-cad/socbed) testbed was used to generate benign datasets for each of the four different SIEM rule and event types that AMIDES was tested with. The `data/socbed` folder contains a subfolder with a small dataset for each of the event types. The samples in the `train` and `validation` files of each subfolder have already been split and normalized for the usage with training and validation scripts. The `all` file holds both training and validation samples in a non-normalized format. The following datasets are provided:

- `windows/process_creation` - The dataset in this folder consists of process command lines taken from the `CommandLine` field of  Sysmon `ProcessCreation` events.
- `proxy_web` - This folder contains full URLs observed in web proxy logs.
- `windows/registry` - Samples in this folder are registry keys extracted from the `TargetObject` field of Sysmon `RegistryEvent (Value Set)` and `RegistryEvent (Object Create and Delete)` events. For `Value Set` events, the samples also hold the corresponding `Details` value.
- `windows/powershell` - The samples of this dataset are `ScriptBlockText` field values extracted from `Microsoft-Windows-PowerShell/Operational 4104` events.

### Sigma Rules, Matches, and Evasions

The SIEM detection rules provided in this repository are converted Sigma rules. The rules have been converted using the Sigma rule converter and are located in `data/sigma/rules`. The rule types match the event types of the benign SOCBED datasets. The converted rules in `data/sigma` are thus organized in a similar folder structure as the benign datasets.

Corresponding matches, i.e., SIEM events triggering the detection rules, and a small number of evasions, i.e., matches adapted such that the executed commands achieve the exact same goal without triggering the respective rule, already revealed in the corresponding [research paper](#documentation) are provided in `data/sigma/events`. Both matches and evasions of a specific rule are organized in single `.json` files. Files with matches carry the pattern `_Match_` in their name, evasions the pattern `_Evasion_`.

## Getting Started

In order to just run the [experiments](#running-experiments), we highly recommend using the quickstart environment where the `amides` package and all its dependencies are already installed. The quickstart environment can also be used if experiments on your own datasets should be carried out. Installing `amides` onto your local system (or using a virtual environment) is also possible.

### Building the Quickstart Environment

Using the quickstart environment requires a `docker` installation. Building and running the environment was tested using `docker 20.10`, but it should also be compatible with other Docker versions.

In order to build the `amides:base` image for the quickstart environment, change into the project root directory and execute

```bash
./build_image.sh 
```

This will execute the corresponding `docker build` command. The image is based on the `python:3.11-slim-bookworm` image. If the quickstart environment image is no longer needed at some point, it can be removed by executing the `remove_image.sh` script from the project root directory.

### Installation

In case you want to use `amides` without the quickstart environment, the package can also be locally installed like other Python packages. We highly recommend to use a dedicated virtual environment for `amides` though. Virtual environments are created using `venv`. To create a dedicated virtual environment, execute

```bash
python3 -m venv <VIRTUAL-ENVIRONMENT-LOCATION>
```

After the environment has been created, activate it by executing

```bash
source <VIRTUAL-ENVIRONMENT-LOCATION>/bin/activate
```

To install the `amides` package and all its dependencies, change into the `amides` directory and execute

```bash
pip install -r requirements.txt
pip install .
```

Now, the `amides` modules and scripts should be usable in your virtual environment.

### Testing

`tox` can be used to execute the unit tests of the `amides` package, located in `amides/tests`. `tox` can be installed from PyPI via

```bash
pip install tox
```

To get an overview of the configured test environments, execute

```bash
tox -av
```

in the package root folder. The `tox.ini` file currently provides configurations to run unit tests on Python 3.10 and 3.11. For example, executing

```bash
tox -e py310-tests
```

will execute unit tests using Python 3.10.

## Running Experiments

The `amides` package comes with a bash script named `experiments.sh`, which is located in the package root folder. Executing the script from the package root folder will use the SOCBED and Sigma rule data in `data` and carry out the four experiments that we performed to evaluate the major claims of the corresponding [research paper](#documentation). Each of the experiments is described in more detail below.

Since the benign datasets in this repository are generated using SOCBED and not taken from the same enterprise network as in the research paper, the generated experiment results will look different. Hence, we provide the additional document `Paper Supplement.pdf` that shows the correct output that is to be expected when using the SOCBED datasets.
Due to ethical concerns, most of the evasions are not available in this repository. Thus, the results produced with the small amount of evasions in this repository are different again.

### Classification Performance

This experiment compares AMIDES' misuse classification performance to the benchmark approach that learns from matches (i.e., attack events matching SIEM rules) instead of the Sigma rule filters. This experiment can be carried out alone  by executing `classification.sh` from the package root folder, either using your local installation or the environment container. A look into the script file reveals the scripts and configuration files used for this experiment.

When finished, the file `figure_3_c1_c2_misuse_classification.pdf` (showing precision, recall, F1 score, and MCC for the threshold range from 0 to 1) is located in the `amides/plots/process_creation` folder.

### Rule Attribution

This experiment evaluates the AMIDES rule attribution model's attribution performance by assessing the position of the correct detection rule in the ranked list of possibly evaded detection rules. This experiment can be run by changing into the `amides` package directory and executing the `rule_attribution.sh` script. The Python scripts used and the corresponding configuration files are revealed in the script file.

After execution, figure `figure_4_c3_rule_attribution.pdf` in `amides/plots/process_creation` visualizes the distribution and cumulative distribution of the correct rule rankings returned by the rule attribution model.

### Tainted Training Data

The influence of different fractions of tainted training data onto AMIDES' misuse classification model is evaluated in this experiment. During the experiment, 10%, 20%, and 30% of the Sigma rule evasions are used to taint benign samples for the training of AMIDES' misuse classification model. During the experiment, the training data is tainted ten times for each fraction of tainted data. This specific experiment can be re-run by changing into the `amides` package folder and executing the `tainted_training.sh` script.

Precision and recall of all 30 training runs are shown in `figure_5_c4_tainted_training.pdf`, also located in the `amides/plots/process_creation` folder.

### Other Rule and Event Types

The classification performance of the AMIDES misuse classification model for Windows PowerShell, Windows Registry, and Web Proxy datasets is evaluated in this experiment. The experiment can be carried out by executing `classification_other_types.sh` from the `amides` package root folder. Precision and recall of the models trained on the given SOCBED data are shown in `figure_6_c5_classification_other_types.pdf`, located in `amides/plots`.

## Running Experiments using the Quickstart Environment

After the image of the quickstart environment has been successfully created, change into the project root directory and execute

```bash
./run_experiments.sh
```

This will run the `amides-experiments` container that executes the `experiments.sh` script of the `amides` package. The container is configured to use the bind mounts `amides/models` and `amides/plots` for results generated during the experiments, as well as the `data` mount as source for input data used for the experiments. This means that after the container's execution, models and plots generated by the experiments are accessible via the `amides/models` and `amides/plots` directories in the project root folder. The default input data used for model training and validation is taken from the `data` directory.

Optionally, to start the quickstart environment for running custom experiments, change into the project root directory and execute

```bash
./start_env.sh
```

This script creates and starts the `amides-env` container, which is created from the same base image as the `amides-experiments` container. When being started, the `amides-env` container is configured to immediately start a bash inside the container. The shell allows to use and configure the modules and scripts of the `amides` package to run your own experiments. Supporting the same bind mounts as the `amides-results` container, the `amides-env` container enables to build and evaluate models using your own data.

Both containers are run using the `--rm`-flag, which means they will be automatically removed once they finish execution.

Executing `cleanup.sh` from the same location will remove the base image as well as all models and plots placed in the default `amides/plots` and `amides/models` bind mount directories.

## Running Custom Experiments

The `amides` package enables to create models for AMIDES from your own datasets. The scripts in `amides/bin` are ready to train, validate, and evaluate models for both the misuse classification and rule attribution components of AMIDES. The current training, validation, and evaluation processes using these scripts are described below in more detail.

Training, validation, and evaluation allow to specify different configuration parameters that are usually provided as command line arguments and options.  Using the `-h` or `--h` flag on a script reveals the supported command line options and arguments.

Due to the amount of configuration parameters supported by many scripts, almost all of them support the usage of configuration files. Options and configuration parameters are placed in a `.json` file, where options are specified as keys, and parameters are placed in values. Config files are provided via the `--config` flag.

### Creating Misuse Classification Models

The training of misuse classification models is performed using `train.py`. First, the script takes the given benign samples and SIEM rule filters and converts them into feature vectors. The location of benign samples is specified by the `--benign-samples` flag. Currently, benign training data need to be provided in .txt- or .jsonl-files, containing one sample per line. Prior to vectorization, benign samples are normalized. Data can either be provided normalized or still needs to be normalized. In the latter case, the `--normalize` flag has to be set.

In case your data should be normalized beforehand, you can use the `normalize.py` script. Samples need to be provided in the same format as for `train.py`. The script applies the normalization currently used by AMIDES and stores them into output files. Assuming your dataset is located at `data/socbed/process_creation/all`, normalize it by executing

```bash
./bin/normalize.py "../data/socbed/process_creation/all" --out-file "../data/socbed/process_creation/train/all_normalized"
```

Locations of the SIEM detection rules and corresponding matches and evasions are defined by `--rules-dir` and `--events-dir`. SIEM rule filters and events are loaded by the `RuleSetDataset` class of the `amides.sigma` module. The `--malicious-samples-type` flag determines the type of malicious samples used for training. `rule_filters` uses the SIEM rule filters, `matches` takes the rule-triggering attack events.

After normalization, the feature extractor converts all samples into TF-IDF vectors. With the `--vectorization` option, other feature extraction and vectorization methods are available. The vectors are later used to fit the SVM model. The `--search-params` flag determines if the hyper-parameters of the SVM should be optimized, or the SVM should just be fitted on default parameters. Currently, the optimization is performed by the `GridSearchCV` class of `scikit-learn`. `GridSearchCV` exhaustively generates candidates from a grid of parameter values. The currently pre-set range of parameter values has been discovered throughout various experiments performed during the AMIDES development. The cross validation splitting strategy is a Stratified-K-Fold approach. The `--cv` flag determines the number of folds. The score function used to evaluate a parameter setting is specified by `--scoring`. The default score function is 'f1-score', but nearly all score functions of `sklearn.metrics` are compatible.

After parameters have been established and the model has been fit, an additional output scaler is created. The `--mcc-scaling` flag determines if the scaler range is determined by the MCC values on the benign training data. The `--mcc-threshold` determines the threshold value that is applied symmetrically to determine the required value range.

By executing

```bash
./bin/train.py --benign-samples "../data/socbed/process_creation/train"  --events-dir "../data/sigma/events/windows/process_creation" --rules-dir "../data/sigma/rules/windows/process_creation" --type "misuse" --malicious-samples-type "rule_filters" --search-params  --cv 5 --mcc-scaling --mcc-threshold 0.5  --result-name "misuse_model"  --out-dir "models/process_creation"
```

a misuse classification models is trained using the benign command lines in `../data/socbed/process_creation/train` and the SIEM rule filters in `./data/sigma/events/windows/process_creation`.

The final model is encapsulated into a `TrainingResult` object, together with the transformed training data vectors, the feature extractor, and the scaler. The object gets pickled into the location specified by the `--out-dir` flag. An additional JSON file containing basic information on model parameters etc. is also generated in the same location.

After training, the model needs to be validated. The `validate.py` script loads model and data from the pickled `TrainingResult` object and calculates decision function values on the specified validation dataset.

Benign validation is provided in the same way as for model training, using the `--benign-samples` option. The `--malicious-samples-type` flag determines whether malicious samples should be `evasions` or `matches`.

By executing

```bash
./bin/validate.py --result-path "models/process_creation/train_rslt_misuse_model.zip" --benign-samples "../data/socbed/process_creation/validation" --events-dir "../data/sigma/events/windows/process_creation" --rules-dir "../data/sigma/rules/windows/process_creation" --malicious-samples-type "evasions" --out-dir "models/process_creation"
```

the previously trained model is validated using the benign data in `data/socbed/process_creation/validation` and the evasions located in `data/sigma/events/windows/process_creation`. The final result is bundled into a `ValidationResult` object, which is pickled into the specified output location.

After the misuse model has been validated, its classification performance is evaluated. The `evaluate_mcc_scaling.py` script loads the validated model and calculates precision, recall, F1 score, and MCC values for the decision function value range that is determined by a specified MCC threshold value.
The number of evaluation thresholds (or iterations) in the target value range is specified by the  `--num-eval-thresholds` flag. By executing

```bash
./bin/eval_mcc_scaling.py --valid-results "models/process_creation/valid_rslt_misuse_model.zip" --num-eval-thresholds 50 --out-dir "models/process_creation"
```

the classification performance of the loaded model is evaluated for 50 evenly spaced threshold values of the dynamically determined threshold interval. The evaluation results are collected by a `BinaryEvaluationResult` object, which is also pickled.

To visualize the evaluation results, the `plot_pr.py`-script is used to create a precision-recall-thresholds plot:

```bash
./bin/plot_pr.py --result "models/process_creation/eval_rslt_misuse_model.zip" --type "prt" --out-dir "plots" 
```

### Performing Tainted Training

Tainted training is performed in the same way as training misuse classification models. For tainted training, the `--tainted-benign-samples` and `--tainted-seed` options are provided to `train.py`. The `tainted-benign-samples` option takes a value between 0 and 100 and defines the fraction of evasions that are used as benign training samples. In order to recreate tainted training results, the `tainted-seed` parameter can be provided. The seed value determines the set of evasions that are used for tainting. Executing

```bash
./bin/train.py --benign-samples "../data/socbed/process_creation/train" --events-dir "../data/sigma/events/windows/process_creation" --rules-dir "../data/sigma/rules/windows/process_creation" --type "misuse" --malicious-samples-type "rule_filters" --tainted-benign-samples 10.0 --tainted-seed 42 --search-params --cv 5 --mcc-scaling --mcc-threshold 0.5 --result-name "misuse_model_tainted" --out-dir "models/process_creation/tainted/10"
```

trains and optimizes a misuse classification  model using 10% of the evasions as benign samples. The seeding to determine the set of evasions that are used for tainting is 42.

Tainted share and tainted seed values are held by `TrainingResult` objects. When the model is validated, `validate.py` takes the tainted seed and share values to remove the evasions already used for training. Evaluation of tainted training models is performed by `eval_mcc_scaling.py` in the same way as other validation results.

Visualising precision and recall of the `EvaluationResult` objects of multiple tainted training results can be done with the `plot_multi_tainted.py` script. An optional base result without any tainting can be tainted using the `--base-result` flag

```bash
./bin/plot_multi_tainted.py --base-result "models/process_creation/valid_rslt_misuse_model.zip" --low-tainted "models/process_creation/tainted/10/eval_rslt_misuse_model_tainted.zip" --out-dir "plots"
```

### Creating Rule Attribution Models

Rule attribution models are also generated using `train.py`. Creating a rule attribution model basically consists of creating a misuse classification model for each of the SIEM rules of the rule dataset that you provide. Only the compilation of datasets used for training is different.

To build a rule attribution model, the script is started with the `--mode=attribution` option. The process of training rule attribution models can be parallelized. `train.py` supports the `--num-subprocesses` option to specify the number of sub-processes used for training the single rule models. To create a rule attribution model of the benign command lines and the SIEM rule data in `data/`, execute

```bash
./bin/train.py --benign-samples "../data/socbed/process_creation/train" --events-dir "../data/sigma/events/windows/process_creation" --rules-dir "../data/sigma/rules/windows/process_creation" --type "attribution" --malicious-samples-type "rule_filters" --search-params --search-method "GridSearch" --mcc-scaling --mcc-threshold 0.5 --result-name "attr_model" --out-dir "models/process_creation"
```

The rule models are gathered by a `MultiTrainingResult` object, where each entry is a `TrainingResult` object itself.

The evaluation of the rule attribution performance is done by the `eval_attr.py` script. For the rule attribution evaluation, a mapping of rules and their corresponding evasions is required.
The mapping can be provided as .json file by the `--rules-evasions` flag. In this JSON file rule names should be used as keys and the corresponding evasions are grouped into a list value.

```json
{
    "New RUN Key Pointing to Suspicious Folder": [
        "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\foo\\\\%%windir%%\\Temp\\calc.exe"
    ]
}
```

Alternatively, the  mapping is automatically built from the evasion and rule data specified by `events_dir` and `rules_dir` by executing

```bash
./bin/eval_attr.py --multi-result "models/process_creation/multi_train_rslt_attr_model.zip" --events-dir ../data/sigma/events/windows/process_creation --rules-dir "../data/sigma/rules/windows"
```

Results of the rule attribution evaluation are encapsulated in `RuleAttributionEvaluationResult` instances, which are also pickled.

Visualizing the rule attribution evaluation results is performed by the `plot_attr.py` script. The `--plot` option allows to choose between the normal distribution, the cumulative distribution, and a combination of both. To get both attributions into the same plot, choose the `combined` option:

```bash
./bin/plot_attr.py --eval-result "models/process_creation/rl_attr.zip" --plot "combined" --title "rule_attribution_eval_socbed" --out-dir "plots"
```

The generated plot type is the same as in the [rule attribution](#rule-attribution) experiment.

### Preparing Models for Logprep

Models for the operational use of AMIDES' misuse classification and rule attribution components need to be combined into a single `.zip` file, which is then provided to the Logprep instance. The models for the misuse classification and rule attribution components are bundled using the `combine_models.py` script. The pickled `TrainingResult` (or `ValidationResult`) containing the misuse classification model is specified by the `--single` option, the pickled `MultiTrainingResult` containing models for the rule attribution component is determined with the `--multi` flag. Here is an example:

```bash
./bin/combine_models.py --single "models/process_creation/valid_rslt_misuse_model.zip" --multi "models/process_creation/multi_train_rslt_attr_model.zip" --out-dir "models/operational"
```

## Documentation

The corresponding research paper describes AMIDES in more detail:

R. Uetz, M. Herzog, L. Hackländer, S. Schwarz, and M. Henze, "You Cannot Escape Me: Detecting Evasions of SIEM Rules in Enterprise Networks,"
in *Proceedings of the 33rd USENIX Security Symposium (USENIX Security)*, 2024. [[Conference Website](https://www.usenix.org/conference/usenixsecurity24/presentation/uetz)] [[Prepublication PDF](https://www.usenix.org/system/files/sec23winter-prepub-112-uetz.pdf)]

## License

The files in this repository are licensed under the GNU General Public License Version 3. See [LICENSE](LICENSE) for details.

If you are using AMIDES for your academic work, please cite the paper under [Documentation](#documentation).
