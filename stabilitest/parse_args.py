import argparse
from random import choices

import numpy as np
from icecream import ic

import stabilitest.mri_loader.parse_args
import stabilitest.numpy_loader.parse_args
import stabilitest.statistics.multiple_testing

cross_validation_models = ["kta", "loo", "kfold"]


def init_global_args(parser):
    parser.add_argument(
        "--output",
        action="store",
        default="output.pkl",
        help="Analysis output filename",
    )

    parser.add_argument("--verbose", action="store_true", help="verbose mode")

    parser.add_argument(
        "--cpus", default=1, type=int, help="Number of CPUs for multiprocessing"
    )
    parser.add_argument(
        "--cached", action="store_true", help="Use cached value if exist"
    )


def init_stats_args(parser):
    parser.add_argument(
        "--confidence",
        action="store",
        default=[0.95],
        type=float,
        help="Confidence",
        nargs="+",
    )
    parser.add_argument(
        "--distribution",
        choices=[
            "normal",
            "student",
            "normal-skew",
            "normal-general",
            "gaussian-mixture",
            "significant-digit",
        ],
        default="normal",
        help="Distribution type",
    )
    parser.add_argument(
        "--parallel-fitting", action="store_true", help="Parallel fitting"
    )


def init_multiple_comparision_tests_args(parser):
    parser.add_argument(
        "--multiple-comparison-tests",
        nargs="+",
        default=["fwe_bonferroni"],
        choices=stabilitest.statistics.multiple_testing.get_method_names(),
        help="Multiple comparison tests",
    )


# Global submodules (to load before domain submodules)
def init_module_test(parser):
    msg = """
    Submodule for single test
    """
    subparser = parser.add_parser("test", help=msg)
    init_global_args(subparser)
    init_stats_args(subparser)
    init_multiple_comparision_tests_args(subparser)

    return subparser


def init_module_normality(parser):
    msg = """
    Submodule for normality test
    """
    subparser = parser.add_parser("normality", help=msg)
    init_global_args(subparser)
    subparser.add_argument(
        "--confidence",
        action="store",
        default=[0.95],
        type=float,
        help="Confidence",
        nargs="+",
    )
    return subparser


def init_module_distance(parser):
    msg = """
    Submodule for computing various distances
    """
    subparser = parser.add_parser("distance", help=msg)
    init_global_args(subparser)
    return subparser


def init_module_stats(parser):
    msg = """
    Submodule for statistics (mean, std, sig)
    """
    subparser = parser.add_parser("stats", help=msg)
    init_global_args(subparser)
    subparser.set_defaults(output="stats")
    return subparser


def _init_kta_args(parser):
    msg = """
    
    Cross-validation that tests that the reference interval computed contains each
    reference observation. The reference interval is computed by using the all
    observations (keep-them-all), including the one being tested.
    """
    # parser.add_parser("kta", help=msg)
    parser.set_default(k_fold_rounds=None)


def _init_loo_args(parser):
    msg = """
    Sanity check that tests that the reference interval computed contains each
    reference observation. The reference interval is computed by using the all
    observations, excluding the one being tested.
    """
    # parser.add_parser("loo", help=msg)
    parser.set_default(k_fold_rounds=None)


def _init_kfold_args(parser):
    msg = """
    Sanity check that tests that the reference interval (train set)
    computed contain reference observations (test set).
    The train/test is splitted with a 80/20 ratio and
    is done K times.
    """
    # parser.add_parser("kfold", help=msg)
    pass


cross_validation_models = {
    "kta": _init_kta_args,
    "loo": _init_loo_args,
    "kfold": _init_kfold_args,
}


def init_module_cross_validation(parser):
    msg = """
    Submodule for cross-validation
    """
    subparser = parser.add_parser("cross-validation", help=msg)
    init_global_args(subparser)
    init_stats_args(subparser)
    init_multiple_comparision_tests_args(subparser)
    subparser.add_argument(
        "--model",
        required=True,
        choices=cross_validation_models,
        help="Model to perform",
    )
    subparser.add_argument(
        "--k-fold-rounds",
        action="store",
        type=int,
        default=5,
        help="Number of K-fold rounds to perform",
    )
    return subparser


domain_modules = {
    "smri": stabilitest.mri_loader.parse_args.init_module,
    "numpy": stabilitest.numpy_loader.parse_args.init_module,
}

analysis_modules = {
    "test": init_module_test,
    "cross-validation": init_module_cross_validation,
    "normality": init_module_normality,
    "stats": init_module_stats,
    "distance": init_module_distance,
}


def init_domain_modules(parser):
    domain_subparser = parser.add_subparsers(
        title="Domain submodules",
        help="Domain submodules",
        dest="domain",
        required=True,
    )
    for domain_init in domain_modules.values():
        domain_init(parser, domain_subparser)


def init_analysis_modules(parser):
    subparser = parser.add_subparsers(
        title="Analysis submodules", help="Analysis submodules", dest="analysis"
    )
    for analysis_init in analysis_modules.values():
        analysis_parser = analysis_init(subparser)
        init_domain_modules(analysis_parser)


def parse_args(args):
    parser = argparse.ArgumentParser(description="stabilitest", prog="stabilitest")
    init_global_args(parser)
    init_analysis_modules(parser)

    known_args, _ = parser.parse_known_args(args)

    return parser, known_args
