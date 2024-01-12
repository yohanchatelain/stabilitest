import warnings

import json
import numpy as np
from icecream import ic

import stabilitest
import stabilitest.model as model
import stabilitest.mri_loader.sample
import stabilitest.normality
import stabilitest.numpy_loader.sample
import stabilitest.parse_args as parse_args
import stabilitest.pprinter as pprinter
import stabilitest.statistics.stats
import stabilitest.help
from stabilitest.collect import Collector

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_sample_module(args):
    if args.domain == "smri":
        return stabilitest.mri_loader.sample
    if args.domain == "numpy":
        return stabilitest.numpy_loader.sample
    raise Exception(f"Domain not found {args.domain}")


def run_configurator(args, collector):
    sample_module = get_sample_module(args)
    example = sample_module.configurator(args)
    print(example)


def run_kta(args, collector):
    sample_module = get_sample_module(args)
    fvr = model.run_kta(args, sample_module, collector)
    parse_output(fvr)
    return fvr


def run_loo(args, collector):
    sample_module = get_sample_module(args)
    fvr = model.run_loo(args, sample_module, collector)
    parse_output(fvr)
    return fvr


def run_test(args, collector):
    sample_module = get_sample_module(args)
    fvr = model.run_single_test(args, sample_module, collector)
    parse_output(fvr)
    return fvr


def run_normality(args, collector):
    sample_module = get_sample_module(args)
    stabilitest.normality.run_normality_test(args, sample_module, collector)


def run_kfold(args, collector):
    sample_module = get_sample_module(args)
    fvr = model.run_kfold(args, sample_module, collector)
    parse_output(fvr)
    return fvr


def run_stats(args, collector):
    sample_module = get_sample_module(args)
    stabilitest.statistics.stats.compute_stats(args, sample_module, collector)


def run_distance(args, collector):
    sample_module = get_sample_module(args)
    stabilitest.mri_loader.distance.main(args, sample_module, collector)


cross_validation_models = {
    "kta": run_kta,
    "loo": run_loo,
    "kfold": run_kfold,
}


def run_cross_validation(args, collector):
    return cross_validation_models[args.model](args, collector)


tests = {
    "configurator": run_configurator,
    "single-test": run_test,
    "cross-validation": run_cross_validation,
    "normality": run_normality,
    "stats": run_stats,
    "distance": run_distance,
}


def parse_output(output):
    methods_ = set()
    confidences_ = set()
    runs_ = set()
    nb_tests = 0
    for run_id, run in enumerate(output):
        runs_.add(run_id)
        tests = 0
        for _id, results in run.items():
            tests += 1
            for (method, confidence), result in results.items():
                confidences_.add(confidence)
                methods_.add(method)
        nb_tests = max(nb_tests, tests)

    methods_ = {m: i for i, m in enumerate(methods_)}
    runs_ = {r: i for i, r in enumerate(runs_)}
    confidences_ = {c: i for i, c in enumerate(confidences_)}

    _to_print = np.ndarray(
        shape=(len(methods_), len(runs_), len(confidences_), nb_tests), dtype=object
    )

    methods = {}
    for nb_run, run in enumerate(output):
        for result_id, (_id, results) in enumerate(run.items()):
            for (method, confidence), result in results.items():
                methods[method, result.confidence] = methods.get(
                    (method, result.confidence), 0
                ) + int(result.passed)

                _to_print[methods_[method]][runs_[nb_run]][confidences_[confidence]][
                    result_id
                ] = int(result.passed)

    for method, method_id in methods_.items():
        print(f"Method {method}")
        nb_rounds = len(runs_)
        print(f"{nb_rounds} round{'s' if nb_rounds > 1 else ''}")
        for confidence, confidence_id in confidences_.items():
            print(f"{confidence:.3f} ", end="")
            for run, run_id in runs_.items():
                for passed in _to_print[method_id][run_id][confidence_id]:
                    msg = (
                        pprinter.as_success(".") if passed else pprinter.as_failure("x")
                    )
                    print(f"{msg}", end="")
                print("|", end="")
            print()
        print()

    for (method, confidence), passed in methods.items():
        nb_tests = len(output)
        ratio = passed / nb_tests
        print(
            f"Method: {method}, alpha: {1-confidence:.5f}, passed: {passed}, tests: {nb_tests}, ratio: {ratio:.2.2f}%"
        )


def main(args=None):
    ic.configureOutput(includeContext=True)

    parser, parsed_args = parse_args.parse_args(args)

    if parsed_args.verbose:
        pprinter.enable_verbose_mode()

    if parsed_args.analysis == "list-domain":
        ic(parser)
        ic(parsed_args)
        ic(dir(parser._subparsers))
        ic(dir(parsed_args))
        parser.print_help()
        return

    if parsed_args.help_info_list:
        info = stabilitest.help.info_list()
        print(info)
        return

    if parsed_args.help_info:
        info = stabilitest.help.main(parsed_args.help_info)
        print(info)
        return

    if parsed_args.analysis not in parse_args.analysis_modules:
        parser.print_help()
        return

    collector = Collector(parsed_args.output)
    tests[parsed_args.analysis](parsed_args, collector)
    collector.dump()
