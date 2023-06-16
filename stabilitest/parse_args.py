import argparse

from icecream import ic

import stabilitest.mri.parse_args

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


def init_module_kta(parser):
    msg = """
    
    Cross-validation that tests that the reference interval computed contains each
    reference observation. The reference interval is computed by using the all
    observations (keep-them-all), including the one being tested.
    """
    subparser = parser.add_parser("kta", help=msg)
    init_global_args(subparser)


def init_module_loo(parser):
    msg = """
    Sanity check that tests that the reference interval computed contains each
    reference observation. The reference interval is computed by using the all
    observations, excluding the one being tested.
    """
    subparser = parser.add_parser("loo", help=msg)
    init_global_args(subparser)


def init_module_k_fold(parser):
    msg = """
    Sanity check that tests that the reference interval (train set)
    computed contain reference observations (test set).
    The train/test is splitted with a 80/20 ratio and
    is done K times.
    """
    subparser = parser.add_parser("k-fold", help=msg)
    subparser.add_argument(
        "--k-fold-rounds",
        action="store",
        type=int,
        default=5,
        help="Number of K-fold rounds to perform",
    )
    init_global_args(subparser)


def init_module_one(parser):
    msg = """
    Test that the target image is included into the reference
    interval computed from the reference sample.
    """
    subparser = parser.add_parser("one", help=msg)
    init_global_args(subparser)
    subparser.add_argument(
        "--target-prefix", action="store", required=True, help="Target prefix path"
    )
    subparser.add_argument(
        "--target-dataset", action="store", required=True, help="Dataset target"
    )
    subparser.add_argument(
        "--target-subject", action="store", required=True, help="Subject target"
    )
    subparser.add_argument(
        "--target-template", action="store", required=True, help="Target template"
    )


# Domain submodules
def init_module_smri(parser):
    msg = """
    Submodule for Structural MRI analysis
    """
    subparser = parser.add_parser("smri", help=msg)
    init_global_args(subparser)


def init_stats_args(parser):
    parser.add_argument(
        "--confidence",
        action="store",
        default=0.95,
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


# Global submodules (to load before domain submodules)
def init_module_test(parser):
    msg = """
    Submodule for single test
    """
    subparser = parser.add_parser("test", help=msg)
    init_global_args(subparser)
    init_stats_args(subparser)
    return subparser


def init_module_normality(parser):
    msg = """
    Submodule for normality test
    """
    subparser = parser.add_parser("normality", help=msg)
    init_global_args(subparser)
    return subparser


def init_module_snr(parser):
    msg = """
    Submodule for computing SNR
    """
    subparser = parser.add_parser("snr", help=msg)
    init_global_args(subparser)
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
    return subparser


def init_module_cross_validation(parser):
    msg = """
    Submodule for cross-validation
    """
    subparser = parser.add_parser("cross-validation", help=msg)
    init_global_args(subparser)
    init_stats_args(subparser)

    return subparser


domain_modules = {
    "smri": stabilitest.mri.parse_args.init_module,
}

analysis_modules = {
    "test": init_module_test,
    "cross-validation": init_module_cross_validation,
    "normality": init_module_normality,
    "stats": init_module_stats,
    "snr": init_module_snr,
    "distance": init_module_distance,
}


def init_domain_modules(parser):
    for domain_init in domain_modules.values():
        domain_init(parser)


def init_analysis_modules(parser):
    subparser = parser.add_subparsers(
        title="Analysis submodules", help="Analysis submodules", dest="analysis"
    )
    for analysis_init in analysis_modules.values():
        analysis_parser = analysis_init(subparser)
        init_domain_modules(analysis_parser)


def parse_args():
    parser = argparse.ArgumentParser(description="stabilitest", prog="stabilitest")
    init_global_args(parser)
    init_analysis_modules(parser)

    args, _ = parser.parse_known_args()

    ic(parser)
    ic(args)

    return parser, args
