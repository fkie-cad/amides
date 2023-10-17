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

*`windows/process_creation` - This process command-lines in this folder are taken from Sysmon ProcessCreation (ID 1) events.
*`proxy_web` - This folder contains full URLs observed in web-proxy logs.
*`windows/registry` - Samples in this folder are registry keys extracted from Sysmon RegistryEvent (Value Set) (ID 12) and RegistryEvent (Object Create and Delete) events. For Value Set events the samples also hold the corresponding key values.
*`windows/powershell` - The samples in this folder are ScriptBlockText field values  extracted from Microsoft-Windows-PowerShell (ID 4104) events.

Samples in the `train` and `validation` files have already been normalized for training and validation scripts. Each sub-folder additionally contains a file named `all` which contains alltraining and validation samples in a non-normalized format.

Converted Sigma rules, matches, and a small number of evasions already revealed in the corresponding [academic research paper](#documentation) are located in the `data/sigma` folder. Converted rules required for model training are located in `data/sigma/rules`, matches and evasions required for model validation  are located in `data/sigma/events`. The subfolders show the same structure as the benign data in the `data/socbed` folder.

## Building and Using the Quickstart Environment

We highly recommend the AMIDES quickstart environment where the `amides` package and all its dependencies are already installed. Using the quickstart environment requires a Docker installation. Building and running the environment was tested using Docker 20.10, but it should also work with other Docker versions.

In order to build the `amides:base` image of the quickstart environment, execute the

    ./build_image.sh 

script located in the project's root folder. This will execute the corresponding `docker build` command. The image is based on the `python:3.11-slim-bookworm` image. If the image is no longer needed at some point, it can be removed by executing the `remove_image.sh` script.

After the image has been successfully created, execute the

    ./create_containers.sh

script to create the two containers `amides-experiments` and `amides-env`.

The `amides-experiments` container is specifically created to execute the `experiments.sh` script inside the `bin` folder of the `amides` package (See [experiments](#running-experiments)). Start the execution of `experiments.sh` by executing

    ./run_experiments.sh

in the project's root folder. The container is configured to use separate bind mounts for models and plots generated by AMIDES, as well as the input data used by AMIDES. This means that after the container's execution, the generated models and plots are accessible via the `amides/models` and `amides/plots` directories of the projects root folder. The input data used for model training and validation is the sample data located in the `data` directory of the project's root folder.

The `amides-env` container provides the actual quickstart environment for AMIDES. Starting the container by executing

    ./start_env.sh

runs a bash script inside the container, which can then be used to execute several scripts of the `amides` package, including training and validation, plotting results, etc. The `amides-env` container supports the same bind mounts as the `amides-results` container. This means event data from the local system are accessible within the container. Models, evaluation results, and plots are accessible from the local file system.

If both containers and their results are no longer required, executing `cleanup.sh` will remove the Docker image and containers, as well as all models and plots produced by the containers.

## Installing

In case you want to run AMIDES without the quickstart environment, the `amides` package can also be locally installed like other Python packages. We highly recommend to use a dedicated virtual environment for AMIDES though. Virtual environments are created either using the `venv` or `virtualenv` package. To create a dedicated virtual environment for AMIDES, execute

    python3 -m venv <VIRTUAL-ENVIRONMENT-LOCATION>

or

    python3 -m virtualenv <VIRTUAL-ENVIRONMENT-LOCATION>

in case you want to use `virtualenv`. After the environment has been created, activate it by executing

    source <VIRTUAL-ENVIRONMENT-LOCATION>/bin/activate

To install the `amides` package and all it's dependencies, change into the `amides` directory and execute

    pip install -r requirements.txt
    pip install .

Now, AMIDES modules and scripts should be usable in your virtual environment.

## Running Experiments

The `amides` package comes with a bash script named `experiments.sh` which is located in the package's root folder. Executing the script will use the given SOCBED and rule data in the `data` directory and carry out the same experiments and produce the same figures as in the corresponding academic research paper, stating our major claims. This includes:

*Classification performance - This experiment compares AMIDES's classification performance to the benchmark approach that learns from attack events ("matches") instead of SIEM rules. The Precision-Recall-Thresholds plot showing Precision, Recall, F1-Score, and MCC is named `figure_3_c1_c2_misuse_classification.pdf` and is located in the `amides/plots/process_creation` folder.
*Rule Attribution - Figure `figure_4_c3_rule_attribution.pdf` in `amides/plots/process_creation`, visualizes the distribution and cumulative distribution of the rule attribution evaluation.
*Tainted training data - The influence of different fractions of tainted training data onto AMIDES' classification performance is shown in `figure_5_c4_tainted_training.pdf`, also in the `amides/plots/process_creation`-folder.
*Classification performance for new rule and event types - The classification performance results for new rule and event types are shown in `figure_6_c5_classification_new_types.pdf`, located in `amides/plots`.

Since the benign data in the repository are generated by SOCBED and not taken from the same enterprise network, the generated results will look different than in the corresponding research paper. Hence, we provide the correct output in the document `Paper Supplement.pdf` in this repository.
As the complete set of rule evasions is not publicly available, the results produced with the small amount of evasions provided in this repository are different again.

## Documentation

The corresponding academic research paper will be published in the proceedings of the 33rd USENIX Security Symposium:

R. Uetz, M. Herzog, L. Hackländer, S. Schwarz, and M. Henze, “You Cannot Escape Me: Detecting Evasions of SIEM Rules in Enterprise Networks,”
in *Proceedings of the 33rd USENIX Security Symposium (USENIX Security)*, 2024.[[DOI]()] [[arXiv]()]


## License

The files in this repository are licensed under the GNU General Public License Version 3. See [LICENSE](LICENSE) for details.

If you are using AMIDES for your academic work, please cite the paper under [Documentation](#documentation).
