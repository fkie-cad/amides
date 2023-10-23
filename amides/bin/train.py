#!/usr/bin/env python3
"""
This script is used to train models for the AMIDES misuse classification and rule attribution components. Models are 
trained using feature vectors extracted from benign samples of enterprise networks  and Sigma rule filters or matches 
(i.e. events triggering SIEM detectio rules) serving as malicious samples.

Benign samples are provided in .txt-files, one sample per line. Sigma rule data(rule filters, matches, evasions) are provided
in folders with .json files, one element per file.

The trained model, converted training data, the feature extractor, as well as the scaler are pickled and saved into a single .zip-archive.
The archive is accompanied by a JSON-file holding meta information on the produced results in human readable format.
"""

import os
import sys
import argparse
import multiprocessing
import numpy as np
import random
import json

from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer,
)
from sklearn.pipeline import Pipeline

from ast import literal_eval

from amides.persist import Dumper
from amides.sigma import RuleDatasetError, RuleSetDataset
from amides.data import (
    DataBunch,
    TrainTestValidSplit,
    TrainingResult,
    MultiTrainingResult,
)
from amides.models.selection import HyperParameterOptimizer, GridSearch
from amides.features.extraction import CommandlineExtractor
from amides.features.normalize import normalize, Normalizer
from amides.features.deduplicate import deduplicate_samples
from amides.features.tokenization import TokenizerFactory
from amides.scale import (
    create_symmetric_mcc_min_max_scaler,
    create_symmetric_min_max_scaler,
)
from amides.utils import (
    execution_time,
    get_logger,
    read_json_file,
    set_log_level,
    load_args_from_file,
)

set_log_level("info")
_logger = get_logger("train")

base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))
sigma_dir = os.path.join(base_dir, "data/sigma")
events_dir = os.path.join(sigma_dir, "events/windows/process_creation")
rules_dir = os.path.join(sigma_dir, "rules/windows/process_creation")

benign_samples_path = os.path.join(base_dir, "data/socbed/process_creation/train")

num_benign_samples = 0
benign_samples = None
deduplicate_benign_samples = False
normalize_benign_samples = False

dumper = None
output_dir = None

model_type = None
malicious_samples_type = "rule_filter"

vectorization = "tfidf"
tokenization = "comma_separation"
ngram_mode = "word"
ngram_range = (1, 1)

search_params = False
params_path = None
model_params = {
    "kernel": ["linear"],
    "C": np.logspace(-2, 1, num=50),
    "class_weight": ["balanced", None],
}

scorings = {"f1": f1_score, "mcc": matthews_corrcoef}
scoring = "f1"
cv_schema = 5
num_optimization_jobs = 3

mcc_scaling = True
mcc_threshold = 0.1
num_mcc_samples = 50

tainted_benign_samples = 0.0
tainted_random_seed = 42
tainted_sample_seedings = []

num_subprocesses = 1
num_iterations = 1

save_data = True
result_name = "misuse_svc"


def get_search_method():
    if model_type == "misuse":
        return GridSearchCV
    return GridSearch


def init_dumper():
    global dumper

    try:
        if not dumper:
            dumper = Dumper(output_dir)
    except OSError as err:
        _logger.error(err)
        sys.exit(1)


def count_benign_samples(benign_samples):
    return sum(1 for _ in benign_samples)


def save_result(result):
    dumper.save_object(result)


def load_rules_data():
    try:
        rule_set = RuleSetDataset()
        rule_set.load_rule_set_data(events_path=events_dir, rules_path=rules_dir)

        return rule_set
    except FileNotFoundError as err:
        _logger.error(err)
        sys.exit(1)


def load_model_params():
    global model_params

    try:
        model_params = read_json_file(params_path)
    except (FileNotFoundError, PermissionError) as err:
        _logger.error(err)
        sys.exit(1)


def prepare_multi_train_result(train_results, iteration):
    multi_train_result = MultiTrainingResult(
        name=f"{result_name}_{iteration}", timestamp=""
    )

    for result in train_results.values():
        multi_train_result.add_result(result)

    return multi_train_result


def prepare_benign_sample_tainting():
    if tainted_benign_samples < 0.0 or tainted_benign_samples > 100.0:
        _logger.error("Share of tainted benign samples must be within 0.0 and 100.0")
        sys.exit(1)

    global tainted_sample_seedings
    random.seed(tainted_random_seed)

    for _ in range(num_iterations):
        tainted_sample_seedings.append(random.randint(0, 100))


@execution_time
def search_best_model_params(train_data):
    hp_optimizer = HyperParameterOptimizer(
        SVC(cache_size=2000),
        search_method=get_search_method(),
        param_grid=model_params,
        cv_schema=cv_schema,
        n_jobs=num_jobs,
        scoring=make_scorer(scorings[scoring]),
    )
    hp_optimizer.search_best_parameters(train_data)

    return hp_optimizer.best_estimator


@execution_time
def fit_model(train_data):
    estimator = SVC(
        C=model_params["C"],
        kernel=model_params["kernel"],
        class_weight=model_params["class_weight"],
        cache_size=2000,
    )

    estimator.fit(train_data.samples, train_data.labels)

    return estimator


def create_vectorizer():
    tokenizer = TokenizerFactory.create(tokenization)

    if vectorization == "count":
        return CountVectorizer(
            tokenizer=tokenizer, analyzer=ngram_mode, ngram_range=ngram_range
        )
    elif vectorization == "tfidf":
        return TfidfVectorizer(
            tokenizer=tokenizer, analyzer=ngram_mode, ngram_range=ngram_range
        )
    elif vectorization == "hashing":
        return HashingVectorizer(
            tokenizer=tokenizer, analyzer=ngram_mode, ngram_range=ngram_range
        )
    elif vectorization == "binary_count":
        return CountVectorizer(
            tokenizer=tokenizer,
            analyzer=ngram_mode,
            ngram_range=ngram_range,
            binary=True,
        )
    elif vectorization == "scaled_count":
        return Pipeline(
            steps=[
                (
                    "count",
                    CountVectorizer(
                        tokenizer=tokenizer,
                        analyzer=ngram_mode,
                        ngram_range=ngram_range,
                    ),
                ),
                ("scaling", MaxAbsScaler()),
            ]
        )


def create_labels_array(
    num_benign_samples: int, num_tainted_samples: int, num_malicious_samples: int
):
    labels = []

    labels.extend((num_benign_samples + num_tainted_samples) * [0])
    labels.extend(num_malicious_samples * [1])

    return np.array(labels)


def benign_samples_iterator():
    for sample in open(benign_samples_path, "r", encoding="utf-8"):
        yield sample.rstrip("\n")


def normalized_benign_samples_iterator():
    normalizer = Normalizer()
    for sample in open(benign_samples_path, "r", encoding="utf-8"):
        yield normalizer.normalize(json.loads(sample.rstrip("\n")))


def prepare_benign_samples_for_fitting():
    if deduplicate_benign_samples:
        deduplicated = deduplicate_samples(benign_samples_path, Normalizer())
        if normalize_benign_samples:
            return normalize(deduplicated.samples)
        else:
            return deduplicated.samples
    else:
        if normalize_benign_samples:
            return normalized_benign_samples_iterator
        else:
            return benign_samples_iterator


def create_malicious_sample_list(rule_set_data: RuleSetDataset):
    if malicious_samples_type == "matches":
        events = rule_set_data.matches.data
        samples = CommandlineExtractor.extract_commandline(events)
    else:
        samples = rule_set_data.extract_field_values_from_filter(
            search_fields=["process.command_line"]
        )

    return normalize(samples)


def create_tainted_sample_list(rule_set_data: RuleSetDataset, seed: int):
    if tainted_benign_samples > 0.0:
        evasions = rule_set_data.evasions

        taint_evasions, _ = evasions.create_random_split(
            [tainted_benign_samples, 1.0 - tainted_benign_samples],
            seed=seed,
        )

        tainted_samples = CommandlineExtractor.extract_commandline(taint_evasions.data)

        return normalize(tainted_samples)
    return []


def train_samples(
    benign_samples: list, tainted_samples: list, malicious_samples: list
) -> dict:
    for sample in benign_samples:
        yield sample

    for sample in tainted_samples:
        yield sample

    for sample in malicious_samples:
        yield sample


@execution_time
def prepare_training_data(
    rule_set_data: RuleSetDataset, taint_seed: int
) -> tuple[DataBunch, BaseEstimator]:
    malicious_samples = create_malicious_sample_list(rule_set_data)
    tainted_samples = create_tainted_sample_list(rule_set_data, taint_seed)
    vectorizer = create_vectorizer()

    benign_samples = prepare_benign_samples_for_fitting()
    feature_vectors = vectorizer.fit_transform(
        train_samples(benign_samples(), tainted_samples, malicious_samples)
    )

    labels = create_labels_array(
        count_benign_samples(benign_samples()),
        len(tainted_samples),
        len(malicious_samples),
    )

    train_data = DataBunch(
        samples=feature_vectors, labels=labels, label_names=["benign", "malicious"]
    )

    return train_data, vectorizer


@execution_time
def create_scaler(estimator: BaseEstimator, train_data: DataBunch):
    df_values = estimator.decision_function(train_data.samples)

    if mcc_scaling:
        return create_symmetric_mcc_min_max_scaler(
            df_values, train_data.labels, num_mcc_samples, mcc_threshold
        )
    else:
        return create_symmetric_min_max_scaler(df_values)


def train_model(
    train_data: DataBunch, vectorizer: BaseEstimator, taint_seed: int
) -> TrainingResult:
    if search_params:
        estimator = search_best_model_params(train_data)
    else:
        estimator = fit_model(train_data)

    scaler = create_scaler(estimator, train_data)

    train_result = TrainingResult(
        estimator,
        scaler=scaler,
        tainted_share=tainted_benign_samples,
        tainted_seed=taint_seed,
        timestamp="",
    )
    train_result.data = TrainTestValidSplit(
        train_data=train_data if save_data else None, name=result_name
    )
    train_result.add_feature_extractor(vectorizer)

    return train_result


def _train_models(dataset_queue: multiprocessing.Queue, train_results: dict):
    while True:
        rule_dataset, taint_seed = dataset_queue.get()
        if not rule_dataset:
            break

        if rule_dataset.evasions.size == 0:
            continue

        _logger.info("Training model for %s", rule_dataset.name)
        try:
            train_data, feature_extractor = prepare_training_data(
                rule_dataset, taint_seed
            )
            train_result = train_model(train_data, feature_extractor, taint_seed)
        except (RuleDatasetError, ValueError) as err:
            _logger.error(err)
            continue

        train_result.name = rule_dataset.name
        train_results[rule_dataset.name] = train_result


@execution_time
def create_attribution_model(
    rule_set_data: RuleSetDataset, iteration: int
) -> MultiTrainingResult:
    manager = multiprocessing.Manager()
    train_results = manager.dict()
    dataset_queue = multiprocessing.Queue()
    taint_seed = tainted_sample_seedings[iteration]
    workers = []

    for _ in range(num_subprocesses):
        workers.append(
            multiprocessing.Process(
                target=_train_models, args=(dataset_queue, train_results)
            )
        )
        workers[-1].start()

    for rule_dataset in rule_set_data.rule_datasets.values():
        dataset_queue.put((rule_dataset, taint_seed))

    for _ in range(num_subprocesses):
        dataset_queue.put((None, None))

    for worker in workers:
        worker.join()

    multi_train_result = prepare_multi_train_result(dict(train_results), iteration)

    return multi_train_result


@execution_time
def create_misuse_model(
    rule_set_data: RuleSetDataset, iteration: int
) -> TrainingResult:
    _logger.info("Creating misuse model for %s", rule_set_data)
    taint_seed = tainted_sample_seedings[iteration]
    train_data, feature_extractor = prepare_training_data(rule_set_data, taint_seed)
    train_result = train_model(train_data, feature_extractor, taint_seed)
    train_result.name = f"{result_name}_{iteration}"

    return train_result


def create_model():
    prepare_benign_sample_tainting()

    rules_data = load_rules_data()

    if params_path is not None:
        load_model_params()

    for i in range(num_iterations):
        _logger.info("Creating Model - Iteration: %s", i)
        if model_type == "misuse":
            result = create_misuse_model(rules_data, i)
        else:
            result = create_attribution_model(rules_data, i)

        save_result(result)


def parse_args_and_options(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    if args.config:
        args = load_args_from_file(parser, args.config)
        if not args:
            _logger.error("Error loading parameters from file. Exiting")
            sys.exit(1)

    if args.benign_samples:
        global benign_samples_path
        benign_samples_path = args.benign_samples

    if args.deduplicate:
        global deduplicate_benign_samples
        deduplicate_benign_samples = True

    if args.normalize:
        global normalize_benign_samples
        normalize_benign_samples = True

    if args.save_data:
        global save_data
        save_data = True

    if args.events_dir:
        global events_dir

        events_dir = args.events_dir

    if args.rules_dir:
        global rules_dir

        rules_dir = args.rules_dir

    if args.model_type:
        global model_type
        model_type = args.model_type
    else:
        _logger.error("No model specified. Exiting")
        sys.exit(1)

    if args.malicious_samples_type:
        global malicious_samples_type
        malicious_samples_type = args.malicious_samples_type

    if args.tainted_benign_samples:
        global tainted_benign_samples
        tainted_benign_samples = args.tainted_benign_samples / 100.0

    if args.tainted_seed:
        global tainted_random_seed
        tainted_random_seed = args.tainted_seed

    if args.vectorization:
        global vectorization
        vectorization = args.vectorization

    if args.ngram_mode:
        global selected_ngram_mode
        selected_ngram_mode = args.ngram_mode

    if args.ngram_range:
        global ngram_range
        ngram_range = literal_eval(args.ngram_range)

    if args.tokenization:
        global tokenization
        tokenization = args.tokenization

    if args.cv:
        global cv_schema
        cv_schema = args.cv

    if args.search_params:
        global search_params
        search_params = True

    if args.scoring:
        global scoring
        scoring = args.scoring

    if args.model_params:
        global params_path
        params_path = args.model_params

    if args.mcc_scaling:
        global mcc_scaling
        mcc_scaling = True

    if args.mcc_threshold:
        global mcc_threshold
        mcc_threshold = args.mcc_threshold

    global output_dir
    if args.out_dir:
        output_dir = args.out_dir
    else:
        output_dir = os.path.join(os.getcwd(), "models")
        _logger.warning(
            "No output dir for results data specified. Using %s", output_dir
        )

    init_dumper()

    if args.num_jobs:
        global num_jobs
        num_jobs = args.num_jobs

    if args.num_subprocesses:
        global num_subprocesses
        num_subprocesses = args.num_subprocesses

    if args.num_iterations:
        global num_iterations
        num_iterations = args.num_iterations

    if args.result_name:
        global result_name
        result_name = args.result_name


def main():
    parser = argparse.ArgumentParser(
        description="Train misuse and rule attribution models for AMIDES"
    )
    parser.add_argument(
        "--benign-samples",
        type=str,
        nargs="?",
        action="store",
        help="Path of the benign training samples file (.txt)",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Perform deduplication of benign samples before model training",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        nargs="?",
        action="store",
        help="Normalize benign samples before training",
    )
    parser.add_argument(
        "--events-dir",
        type=str,
        nargs="?",
        action="store",
        help="Path of the directory with Sigma rule matches and evasions (.json)",
    )
    parser.add_argument(
        "--rules-dir",
        type=str,
        nargs="?",
        action="store",
        help="Path of the directory with Sigma detection rules (.yml)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        action="store",
        choices=["misuse", "attribution"],
        help="Specifies the type of model which should be created",
    )
    parser.add_argument(
        "--malicious-samples-type",
        type=str,
        action="store",
        choices=["rule_filters, matches"],
        help="Specifies the type  of malicious samples used for training",
    )
    parser.add_argument(
        "--tainted-benign-samples",
        type=float,
        action="store",
        help="Fraction (0-100) of evasions that are used for benign samples tainting",
    )
    parser.add_argument(
        "--tainted-seed",
        type=int,
        action="store",
        help="Seeding value to init benign sample tainting",
    )
    parser.add_argument(
        "--vectorization",
        type=str,
        action="store",
        choices=["count", "binary_count", "tfidf", "hashing", "scaled_count"],
        help="Specifies the type of vectorization used to create feature vectors",
    )
    parser.add_argument(
        "--tokenization",
        type=str,
        action="store",
        choices=[
            "default",
            "split",
            "ws_ast",
            "ws_ast_sla_min",
            "ws_ast_sla_min_eq",
            "comma_separation",
        ],
        help="Specifiecs the sample tokenizer given to the vectorizer (if ngram_mode == 'word')",
    )
    parser.add_argument(
        "--ngram-mode",
        type=str,
        action="store",
        choices=["word", "char"],
        help="Specifies the n-gram mode used by the vectorizer",
    )
    parser.add_argument(
        "--ngram-range",
        type=str,
        action="store",
        help="Specifies the n-gram range used by the vectorizer (Example: (1,1))",
    )
    parser.add_argument(
        "--search-params",
        action="store_true",
        help="Optimize the classifier by searching a given hyper parameter space",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        action="store",
        choices=["f1", "mcc"],
        help="Choose the scoring function used for candidate evaluation when perforing exhaustive parameter optimization",
    )
    parser.add_argument(
        "--cv",
        type=int,
        action="store",
        help="Number of cross-validation splits used for parameter optimization",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        action="store",
        help="Path to JSON-file containing parameters used for just fitting the model (No parameter optimization)",
    )
    parser.add_argument(
        "--mcc-scaling",
        action="store_true",
        help="Scale decision function values using MCC-Scaling",
    )
    parser.add_argument(
        "--mcc-threshold",
        action="store",
        type=float,
        help="Threshold value used for MCC-Scaling",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        action="store",
        help="Number of parallel jobs when search candidates are evaluated during parameter optimization",
    )
    parser.add_argument(
        "--num-subprocesses",
        type=int,
        action="store",
        help="Number of parallel processes used to use rule models for the rule attribution model",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        action="store",
        help="Number of model creation iterations (Mainly for performance testing purposes)",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Specify if the transformed training data (feature vectors!) should be added to TrainingResult",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        action="store",
        help="Output directory to save result files",
    )
    parser.add_argument(
        "--result-name",
        type=str,
        action="store",
        help="Specifies the result files base name",
    )
    parser.add_argument(
        "--config", type=str, action="store", help="Path to config file."
    )

    parse_args_and_options(parser)
    create_model()


if __name__ == "__main__":
    main()
