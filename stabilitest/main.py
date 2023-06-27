import warnings

from icecream import ic

import stabilitest
import stabilitest.model as model
import stabilitest.mri_loader.sample
import stabilitest.normality
import stabilitest.numpy_loader.sample
import stabilitest.parse_args as parse_args
import stabilitest.pprinter as pprinter
from stabilitest.collect import stats_collect
import stabilitest.statistics.stats


warnings.simplefilter(action="ignore", category=FutureWarning)


def get_sample_module(args):
    if args.domain == "smri":
        return stabilitest.mri_loader.sample
    if args.domain == "numpy":
        return stabilitest.numpy_loader.sample
    raise Exception(f"Domain not found {args.domain}")


def run_kta(args):
    sample_module = get_sample_module(args)
    fvr = model.run_kta(args, sample_module)
    parse_output(fvr)
    return fvr


def run_loo(args):
    sample_module = get_sample_module(args)
    fvr = model.run_loo(args, sample_module)
    parse_output(fvr)
    return fvr


def run_test(args):
    sample_module = get_sample_module(args)
    fvr = model.run_one(args, sample_module)
    parse_output(fvr)
    return fvr


def run_normality(args):
    sample_module = get_sample_module(args)
    normality = stabilitest.normality.run_normality_test(args, sample_module)
    print(normality)


def run_kfold(args):
    sample_module = get_sample_module(args)
    fvr = model.run_kfold(args, sample_module)
    parse_output(fvr)
    return fvr


def run_stats(args):
    sample_module = get_sample_module(args)
    stabilitest.statistics.stats.compute_stats(args, sample_module)


def run_distance(args):
    sample_module = get_sample_module(args)
    stabilitest.mri_loader.distance.main(args, sample_module)


cross_validation_models = {
    "kta": run_kta,
    "loo": run_loo,
    "kfold": run_kfold,
}


def run_cross_validation(args):
    return cross_validation_models[args.model](args)


tests = {
    "test": run_test,
    "cross-validation": run_cross_validation,
    "normality": run_normality,
    "stats": run_stats,
    "distance": run_distance,
}


def parse_output(output):
    methods = {}
    for run in output:
        for _, results in run.items():
            for (method, confidence), result in results.items():
                methods[method, result.confidence] = methods.get(
                    (method, result.confidence), 0
                ) + int(result.passed)

    for (method, confidence), passed in methods.items():
        nb_tests = len(output)
        ratio = passed / nb_tests
        print(
            f"Method: {method}, alpha: {1-confidence:.5f}, passed: {passed}, tests: {nb_tests}, ratio: {ratio:.2f}"
        )


def main(args=None):
    ic.configureOutput(includeContext=True)

    parser, parsed_args = parse_args.parse_args(args)

    if parsed_args.verbose:
        pprinter.enable_verbose_mode()

    if parsed_args.analysis not in stabilitest.parse_args.analysis_modules:
        parser.print_help()
        return

    output = tests[parsed_args.analysis](parsed_args)

    stats_collect.set_name(parsed_args.output)
    stats_collect.dump()
