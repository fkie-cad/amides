#!/usr/bin/env python3

import json
import sys
import os
import argparse

from amides.features.extraction import CommandlineExtractor
from amides.features.normalize import Normalizer
from amides.sigma import RuleSetDataset
from amides.evaluation import RuleAttributionEvaluationResult
from amides.utils import get_logger, set_log_level, load_args_from_file
from amides.persist import Dumper, PersistError


set_log_level("info")
logger = get_logger("eval_attr")


base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../"))
sigma_dir = os.path.join(base_dir, "data/sigma-study")
events_dir = os.path.join(sigma_dir, "events/windows/process_creation")
rules_dir = os.path.join(sigma_dir, "rules/windows/process_creation")

benign_events = os.path.join(base_dir, "data/socbed/validation.txt")

dumper = None
out_dir = None
multi_result_path = None

rules_evasions = None

rule_attributor = {}

num_rules = 0


def init_dumper():
    global dumper

    try:
        if dumper is None:
            dumper = Dumper(output_path=out_dir)
    except OSError as err:
        logger.error(err)
        sys.exit(1)


def save_result(result):
    dumper.save_object(result)


def load_pickled_object(object_path):
    try:
        return dumper.load_object(object_path)
    except (TypeError, PersistError) as err:
        logger.error(err)
        sys.exit(1)


def samples(evasions, benign_samples=None):
    if benign_samples:
        for sample in open(benign_samples, "r"):
            yield json.loads(sample)

    for sample in evasions:
        yield sample


def create_rules_evasions_dict(rules_data):
    global num_rules
    rules_evasions_dict = {}
    extractor = CommandlineExtractor()

    for rule_data in rules_data.rule_datasets.values():
        if rule_data.evasions.size == 0:
            continue

        num_rules += 1
        rules_evasions_dict[rule_data.name] = list(
            extractor.extract_commandline(rule_data.evasions.data)
        )

    return rules_evasions_dict


def load_rule_set_data():
    try:
        rule_set = RuleSetDataset()
        rule_set.load_rule_set_data(events_dir, rules_dir)

        return rule_set
    except FileNotFoundError as err:
        logger.err(err)
        sys.exit(1)


def load_rules_evasions():
    with open(rules_evasions, "r", encoding="utf-8") as in_file:
        return json.load(in_file)


def load_rules_evasions_data():
    if rules_evasions is not None:
        return load_rules_evasions()
    else:
        rule_set_data = load_rule_set_data()
        return create_rules_evasions_dict(rule_set_data)


def create_evasions_rules_dict(rules_evasions_dict):
    evasions_rules_dict = {}
    for rule, evasions in rules_evasions_dict.items():
        for evasion in evasions:
            evasions_rules_dict[evasion] = rule

    return evasions_rules_dict


def unpack_clf_and_feature_extractor(result):
    clf = result.estimator
    feature_extractors = result.feature_extractors
    vectorizer = feature_extractors[0]
    scaler = result.scaler if result.scaler else None

    return {"clf": clf, "vectorizer": vectorizer, "scaler": scaler}


def prepare_rule_attribution():
    multi_result = load_pickled_object(multi_result_path)

    global rule_attributor
    for rule_name, result in multi_result.results.items():
        model = unpack_clf_and_feature_extractor(result)
        rule_attributor[rule_name] = model


def process_samples(evasions_rules_dict):
    samples = list(sorted(evasions_rules_dict.keys()))
    sample_results = [{} for _ in samples]
    normalizer = Normalizer(max_len_num_values=3)
    normalized = [normalizer.normalize(sample) for sample in samples]

    for rule, model in rule_attributor.items():
        transformed = model["vectorizer"].transform(normalized)
        df_values = model["clf"].decision_function(transformed)

        for i, _ in enumerate(samples):
            sample_results[i][rule] = df_values[i]

    results = []
    for i, sample in enumerate(samples):
        sorted_sample_results = sorted(
            sample_results[i].items(), key=lambda item: item[1], reverse=True
        )
        rule = evasions_rules_dict[sample]
        results.append((rule, sorted_sample_results))

    return results


def evaluate_rule_attribution():
    rules_evasions_dict = load_rules_evasions_data()
    prepare_rule_attribution()
    evasions_rule_dict = create_evasions_rules_dict(rules_evasions_dict)

    eval_rslt = RuleAttributionEvaluationResult(
        num_rules=len(rule_attributor.keys()), timestamp=""
    )

    results = process_samples(evasions_rule_dict)

    for rule, attributions in results:
        eval_rslt.evaluate_rule_attributions(rule, attributions)

    eval_rslt.calculate_top_n_hit_rates()
    save_result(eval_rslt)


def parse_args_and_options(parser):
    args = parser.parse_args()

    if args.config:
        args = load_args_from_file(parser, args.config)
        if not args:
            logger.error("Error loading parameters from file. Exiting")
            sys.exit(1)

    if args.multi_result:
        global multi_result_path
        multi_result_path = args.multi_result

    if args.benign_samples:
        global benign_samples
        benign_samples = args.benign_samples

    if args.rules_evasions:
        global rules_evasions
        rules_evasions = args.rules_evasions

    if args.events_dir:
        global events_dir
        events_dir = args.events_dir

    if args.rules_dir:
        global rules_dir
        rules_dir = args.rules_dir

    global out_dir
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(os.getcwd(), "models")
        logger.warning(f"No path for distribution plot specified. Using {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-result", type=str, action="store")
    parser.add_argument(
        "--benign-samples",
        "-b",
        type=str,
        action="store",
        help="File containing benign samples",
    )
    parser.add_argument(
        "--events-dir",
        "-s",
        type=str,
        action="store",
        help="Directory where rule set events are located",
    )
    parser.add_argument(
        "--rules-dir",
        "-r",
        type=str,
        action="store",
        help="Directory where rule set rules are located",
    )
    parser.add_argument(
        "--rules-evasions",
        "-m",
        type=str,
        action="store",
        help="File containing rule->malicious-samples-mapping",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        action="store",
        help="Location where plot should be saved",
    )
    parser.add_argument(
        "--config", type=str, action="store", help="Path to config file."
    )

    parse_args_and_options(parser)

    init_dumper()
    evaluate_rule_attribution()


if __name__ == "__main__":
    main()
