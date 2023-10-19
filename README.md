# Adaptive Misuse Detection System (AMIDES)

The Adaptive Misuse Detection System (AMIDES) extends conventional rule matching of SIEM systems by applying machine learning components that aim to detect attacks evading existing SIEM rules as well as otherwise undetected attack variants. It learns from SIEM rules and historical benign events and can thus estimate which SIEM rule was tried to be evaded. An overview of AMIDES is depicted in the figure below.

![amides_architecture](./docs/amides.png)

Incoming events are transformed into feature vectors by the feature extraction component. During operation, features learned during the training phase will be re-used by the feature extraction component. Feature vectors are then passed to the Misuse Classification component, which classifies events as malicious or benign. In case of a malicious result, the feature vector is passed to the Rule Attribution component, which generates a ranked list of SIEM rules potentially evaded by the event.

This repository contains the source code used for model training, validation, and evaluation, as well as some initial training and validation data that enable to build and evaluate models for AMIDES.
For operational use, AMIDES is integrated into [Logprep](https://logprep.readthedocs.io/en/latest/user_manual/configuration/processor.html#amides).

## System Requirements

AMIDES was developed and tested on Linux using Python 3.10. Before attempting to use AMIDES, make sure you have

- Physical or virtual host with a Linux-based OS
- A minimum of 8 GB of RAM
- At least 1 GB of HDD space
- Python 3.10 (or newer)

The repository contains a `Dockerfile` that creates a quickstart environment for AMIDES. For testing purposes, we highly recommend to use the quickstart environment. Building and using the environment has been tested with Docker 20.10.

## Accessing Code and Initial Data

In order to access the AMIDES source code and initial data, change into the target location on your system and clone the repository by executing

    git clone https://github.com/fkie-cad/amides.git

or

    git clone git@github.com:fkie-cad/amides.git 

in case you prefer to use SSH.

Alternatively, you can get the repository by downloading the `.zip`-file from the repository's main page and unpack it into your target location.

The `amides` package in the `amides` directory contains modules and scripts that enable to train and validate models for AMIDES, evaluate the model's classification performance, and create meaningful visualizations that help users to assess the models classification performance. The package also contains additional scripts and classes that help to prepare generated models for operational use with Logprep and perform further testing.

Initial data to train and validate models for AMIDES is provided in the `data` directory. The [SOCBED](https://github.com/fkie-cad/socbed) framework was used to generate a small set of benign  data for each of the four different rule types that AMIDES was tested with. The `socbed` folder contains sub-folders  with training and validation data for each of the rule types:

- `windows/process_creation` - The process command-lines in this folder are taken from Sysmon ProcessCreation (ID 1) events.
- `proxy_web` - This folder contains full URLs observed in web-proxy logs.
- `windows/registry` - Samples in this folder are registry keys extracted from Sysmon RegistryEvent (Value Set) (ID 12) and RegistryEvent (Object Create and Delete) events. For Value Set events the samples also hold the corresponding key values.
- `windows/powershell` - The samples in this folder are ScriptBlockText field values extracted from Microsoft-Windows-PowerShell (ID 4104) events.

Samples in the `train` and `validation` files have already been normalized for the usage with training and validation scripts. Each sub-folder additionally contains a file named `all` which contains all training and validation samples in a non-normalized format.

Converted Sigma rules, matches, and a small number of evasions already revealed in the corresponding [academic research paper](#documentation) are located in the `data/sigma` folder. Converted rules required for model training are located in `data/sigma/rules`, matches and evasions required for model validation  are located in `data/sigma/events`. The sub-folders show the same structure as the benign data in the `data/socbed` folder.

## Building and Using the Quickstart Environment

We highly recommend using the AMIDES quickstart environment where the `amides` package and all its dependencies have already been installed. Using the quickstart environment requires a Docker installation. Building and running the environment was tested using Docker 20.10, but it should also work with other Docker versions.

In order to build the `amides:base` image for the quickstart environment, execute the

    ./build_image.sh 

script located in the project's root folder. This will execute the corresponding `docker build` command. The image is based on the `python:3.11-slim-bookworm` image. If the quickstart environment image is no longer needed at some point, it can be removed by executing the `remove_image.sh` script.

After the image has been successfully created, execute

    ./run_experiments.sh

in the project's root folder to create the `amides-experiments` container that executes the `experiments.sh` script inside the `bin` folder of the `amides` package (See [experiments](#running-experiments)). The container is configured to use the bind mounts `amides/models` and `amides/plots` for models and plots generated during the experiments, as well as the `data` mount as source for input data used during the experiments. This means that after the container's execution, models and plots generated by the experiments are accessible via the `amides/models` and `amides/plots` directories of the projects root folder. The default input data used for model training and validation is taken from the `data` directory of the project's root folder.

To start the actual quickstart environment for AMIDES, execute

    ./start_env.sh

in the project's root folder. The script creates and starts the `amides-env` container which is created from the same base image as the `amides-experiments` container. When being started, the `amides-env` container is configured to immediately start a bash inside the container. The shell allows to use and configure the modules and scripts of the `amides` package for further experiments. Supporting the same bind mounts as the `amides-results` container, the `amides-env` container enables to build and evaluate models using your own data.

Executing both scripts, containers are run using the `--rm`-flag, which means the corresponding containers will be automatically removed once they finish execution.

Executing `cleanup.sh` will remove the base image as well as all models and plots placed in the default `amides/plots` and `amides/models` bind mount directories.

## Installing

In case you want to run AMIDES without the quickstart environment, the `amides` package can also be locally installed like other Python packages. We highly recommend to use a dedicated virtual environment for AMIDES though. Virtual environments are created using `venv`. To create a dedicated virtual environment for AMIDES, execute

    python3 -m venv <VIRTUAL-ENVIRONMENT-LOCATION>

After the environment has been created, activate it by executing

    source <VIRTUAL-ENVIRONMENT-LOCATION>/bin/activate

To install the `amides` package and all it's dependencies, change into the `amides` directory and execute

    pip install -r requirements.txt
    pip install .

Now, the `amides` modules and scripts should be usable in your virtual environment.

## Running Experiments

The `amides` package comes with a bash script named `experiments.sh` which is located in the package's root folder. Executing the script will use the given SOCBED and rule data in `data` and carry out the same experiments in the corresponding academic research paper, stating the major claims. Each of the experiments is described in the following sections.

Since the benign data in the repository are generated by SOCBED and not taken from the same enterprise network, the generated results will look different than in the corresponding research paper. Hence, we provide the correct output in the document `Paper Supplement.pdf` in this repository.
Due to responsible disclosure, the complete set of rule evasions is not publicly available and is thus missing in this repository. The results produced with the small amount of evasions provided in this repository are different again.

### Classification Performance

This experiment compares AMIDES's classification performance to the benchmark approach that learns from attack events ("matches") instead of SIEM rules. The Precision-Recall-Thresholds plot `figure_3_c1_c2_misuse_classification.pdf` showing Precision, Recall, F1-Score, and MCC against a threshold range from 0 to 1 is located in the `amides/plots/process_creation` folder.

Using your own local installation or the quickstart environment, this experiment can be carried out in an isolated manner by executing `classification.sh` in the `amides` package's root folder. A look into the script file reveals the `amides` scripts and configuration files used.

### Rule Attribution

This experiment evaluates the AMIDES rule attributors performance. Figure `figure_4_c3_rule_attribution.pdf` in `amides/plots/process_creation` visualizes the distribution and cumulative distribution of the rule attribution evaluation. The rule attribution evaluation can be re-run by the `rule_attribution.sh` script in the `amides` folder. Again, a look into the bash script file reveals scripts and modules used.

### Tainted Training Data

The influence of different fractions of tainted training data onto AMIDES' classification performance is evaluated by this experiment. During the experiment, 10%, 20%, and 30% of the evasions are used as benign data during the training process of AMIDES' misuse classification model. The experiment is re-run ten times. Precision and Recall of the final models are shown in `figure_5_c4_tainted_training.pdf`, also located in the `amides/plots/process_creation` folder. This specific experiment can be re-run by executing the `tainted_training.sh` script in the `amides` folder.

### Classification Performance for new Rule and Event Types

The classification performance of AMIDES for Windows PowerShell, Windows Registry, and Web-Proxy data is evaluated in this experiments. Precision and Recall of the models trained on the given SOCBED data are shown in `figure_6_c5_classification_new_types.pdf`, located in `amides/plots`. The experiment can be carried out by executing the `classification_new_types.sh`.

## Running Your Own Experiments

The `amides` package contains scripts and modules that enable to create models for AMIDES. The scripts provided in `amides/bin` are ready to train, validate, and evaluate models for both the misusue classification and rule attribution components of AMIDES. This section (and sub-sections) show the current training, validation, and evaluation process of the AMIDES modules using the data provided in `data`.
Not all modules and classes of the `amides` package are currently used. Many of them were developed, but later substituted by better performing ones. Most of them are still compatible and usable.

Training, validation, and evaluation processes allow the specification of different configuration parameters. The required configuration parameters are usually provided as command-line arguments and options.  Using the `-h` or `--h` flag reveals the command-line arguments and options supported by each script.

Due to the amount of configuration parameters supported by many of the scripts, almost all of them support the usage of configuration files. Options and configuration parameters are placed in a `.json` file, where flags are specified as keys, and parameters are placed in values. The config file is given to the script via the `--config` flag and is parsed during execution.

### Creating Misuse Classification Models

The training of misuse classification models is performed using `train.py`. The script loads the specified benign data samples and SIEM rule filters and converts them into feature vectors. The location of benign data is specified by the `--benign-samples` flag. Currently, benign training data needs to be provided in .txt- or .jsonl-files, containing one sample per line. Benign samples can be provided normalized, or still need to be normalized. In latter case, the `--normalize` flag has to be set.

Location of the SIEM evasions and rules are provided by `--events-dir` and `--rules-dir`. The `--malicious-samples-type` flag determines the type of malicious samples used for training. `rule_filters` uses the SIEM rule filters, `matches` takes the actual attack events.

The feature extractor converts all samples into TF-IDF vectors, which are then used too fit a SVM model. The `--search-params` flag determines if the hyper-parameters of the SVM should be optimized, or the SVM should just be fitted on default parameters. Currently, the optimization is performed by `GridSearchCV` of `scikit-learn` . The `--cv` flag determines the number of folds for the Stratified-K-Fold used by the cross-validation. The scoring function used for optimization is specified by `--scoring`.

After parameters have been established and the model has been fit, the output scaler is calibrated. The `--mcc-scaling` flag determines if the scaler range is determined by the MCC values of the training data. The `--mcc-threshold` determines the threshold value that is applied symmetrically to determine the required value range.

Executing

    ./bin/train.py --benign-samples "../data/socbed/process_creation/train" 
                   --events-dir "../data/sigma/events/windows/process_creation"
                   --rules-dir "../data/sigma/rules/windows/process_creation"
                   --type "misuse" 
                   --malicious-samples-type "rule_filters" 
                   --search-params 
                   --cv 5
                   --mcc-scaling 
                   --mcc-threshold 0.5 
                   --result-name "misuse_model" 
                   --out-dir "models/process_creation"

trains a misuse classification model using then benign process command-lines in `data/socbed/process_creation` and the SIEM rule filters in `data/sigma/windows/process_creation`.

The final model is encapsulated into a `TrainingResult` object, together with the transformed training data, the feature extractor, and the scaler. The object gets pickled into the location specified by the `--out-dir` flag. An additional JSON-File containing basic information on model parameters, etc. is also generated in the same location.

After training, the trained model is validated using `validation.py`. The script loads the trained model from the pickled TrainingResult object and calculates decision function values on the given validation data.

Benign validation data needs to be provided in the same way as for model training. The `--malicious-samples-type` flag determines whether malicious samples should be `evasions` or `matches`.

By executing

    ./bin/validate.py --result-path "models/process_creation/train_rslt_misuse_model.zip"
                      --benign-samples "../data/socbed/process_creation/validation
                      --events-dir "../data/sigma/events/windows/process_creation"
                      --rules-dir "../data/sigma/rules/windows/process_creation"
                      --malicious-samples-type "evasions"
                      --out-dir "models/process_creation"

the trained model is validated using the benign validation data in `data/socbed/process_creation/validation` and the evasions located in `data/sigma/events/windows/process_creation`. The final result is bundled into a `ValidationResult` object and pickled into the specified output location.

After the misuse model has been validated, it's classification performance is evaluated. The `evaluate_mcc_scaling.py` script loads the validated model and calculates Precision, Recall, F1-Score, and MCC for the decision function value range that is determined by a specified MCC threshold value.
The number of evaluation thresholds (or iterations) in the target decision function value range is specified by the  `--num-eval-thresholds` flag. By executing

    ./bin/eval_mcc_scaling.py --valid-results "models/process_creation/valid_rslt_misuse_model.zip"
                              --num-eval-thresholds 50
                              --out-dir "models/process_creation"

the specified result is evaluated for 50 threshold values. The evaluation results are collected in a `BinaryEvaluationResult` object, which is also pickled.

To visualise the evaluation results, the `plot_pr.py`-script can be used to create a precision-recall-thresholds plot:

    ./bin/plot_pr.py --result "models/process_creation/eval_rslt_misuse_model.zip"
                     --type "prt"
                     --out-dir "plots" 

### Performing Tainted Training

Tainted training is done the same way as training misuse classification models. For tainted training, the `--tainted-share` and `--tainted-seed` options need to be provided to `train.py`. The `tainteds-share` option is a value between 0 and 100 and  defines the fraction of evasions that are used as benign training samples. In order to re-create tainted training results, the `tainted-seed` parameter determines the evasions split: The seeding fixes the set of evasions that go into the benign data. By executing

    ./bin/train.py --benign-samples "../data/socbed/process_creation/train"
                   --events-dir "../data/sigma/events/windows/process_creation"
                   --rules-dir "../data/sigma/rules/windows/process_creation"
                   --type "misuse"
                   --malicious-samples-type "rule_filters"
                   --tainted-share 10.0
                   --tainted-seed 42
                   --search-params
                   --cv 5
                   --mcc-scaling
                   --mcc-threshold 0.5
                   --result-name "misuse_model_tainted"
                   --out-dir "models/process_creation/tainted/10"

10% of the evasions are moved into the set of benign samples. The seeding to determine the set of evasions that are used for tainting is 42.

Tainted share and tainted seed are added to the `TrainingResult` object that is pickled. When the result is used during validation, the tainted seed and share are re-used to remove the evasions already used for training. Tainted training models are evaluated using `eval_mcc_scaling.py` the same way as other validation results.

Visualising precision and recall of the `EvaluationResult` objects of the tainted training results can be done with the `plot_multi_tainted.py` script. Evaluation results of different tainted training runs are provided for the different fraction of tainted training data. An optional base result without any tainting can be tainted using the `--base-result` flag

    ./bin/plot_multi_tainted.py --base-result "models/process_creation/valid_rslt_misuse_model.zip"
                                --low-tainted "models/process_creation/tainted/10/eval_rslt_misuse_model_tainted.zip"
                                --out-dir "plots"

### Creating Rule Attribution Models

Rule attribution models are generated by `train.py`. Creating a rule attribution model basically consists of creating a misuse classification model for each of the SIEM rules in the rule dataset.
To achieve this, the cript is started with the `--mode=attribution` option.

`train.py` supports the parallelization of the rule attribution training process. The `--num-subprocesses` option specifies the number of sub-processes used for training the SIEM rule models.
To create a rule attribution model of the benign command-lines and the SIEM rule data in `data/`, execute

    ./bin/train.py --benign-samples "../data/socbed/process_creation/train"
                   --events-dir "../data/sigma/events/windows/process_creation"
                   --rules-dir "../data/sigma/rules/windows/process_creation"
                   --type "attribution"
                   --malicious-samples-type "rule_filters"
                   --search-params
                   --search-method "GridSearch"
                   --mcc-scaling
                   --mcc-threshold 0.5
                   --result-name "attr_model"
                   --out-dir "models/process_creation"

The rule models are pickled in a `MultiTrainingResult` object, where each entry is a `TrainingResult` object itself.

The evaluation of the rule attribution performance is done by the `eval_attr.py` script. For the rule attribution evaluation, a mapping of rules and their corresponding evasions is required.
The mapping can be provided as .json-file by the `--rules-evasions` flag. In this JSON file rule names are should be used as keys, and the corresponding evasions are grouped into a list value. Alternatively, the  mapping is automatically built from the evasion and rule data specified by `events_dir` and `rules_dir`. Executing

    ./bin/eval_attr.py --multi-result "models/process_creation/multi_train_rslt_attr_model.zip"
                       --events-dir ../data/sigma/events/windows/process_creation
                       --rules-dir "../data/sigma/rules/windows"

Results of the rule attribution evaluation are encapsulated in `RuleAttributionEvaluationResult` instances, which are again pickled.

Visualising the rule attribution evaluation results is performed by executing the `plot_attr.py` script.
The `--plot` option allows to choose between the normal distribution, the cumulative distribution, and a combination of both. To get both attributions into the same plot, choose the `combined` option

    ./bin/plot_attr.py --eval-result "models/process_creation/rl_attr.zip"
                       --plot "combined"
                       --title "rule_attribution_eval_socbed"
                       --out-dir "plots"

The generated plot type is the same as in the [rule attribution](#rule-attribution) experiment.

## Documentation

The corresponding academic research paper will be published in the proceedings of the 33rd USENIX Security Symposium:

R. Uetz, M. Herzog, L. Hackländer, S. Schwarz, and M. Henze, “You Cannot Escape Me: Detecting Evasions of SIEM Rules in Enterprise Networks,”
in *Proceedings of the 33rd USENIX Security Symposium (USENIX Security)*, 2024.[[DOI]()] [[arXiv]()]

## License

The files in this repository are licensed under the GNU General Public License Version 3. See [LICENSE](LICENSE) for details.

If you are using AMIDES for your academic work, please cite the paper under [Documentation](#documentation).
