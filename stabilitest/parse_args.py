import argparse
import json
from textwrap import indent

import stabilitest.mri_loader.parse_args
import stabilitest.numpy_loader.parse_args
import stabilitest.statistics.distribution
import stabilitest.statistics.multiple_testing

_default_confidence_values = [
    0.999,
    0.995,
    0.99,
    0.95,
    0.9,
    0.85,
    0.8,
    0.75,
    0.7,
    0.65,
    0.6,
    0.55,
    0.5,
]


def load_configuration_file(configuration_file):
    with open(configuration_file, "r") as f:
        return json.load(f)


def init_global_args(parser):
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--configuration-file", "-c", metavar="filename", help="Configuration file"
    )
    parser.add_argument(
        "--help-info",
        help="Print help information for a specific argument",
        metavar="argument",
        default="",
    )
    parser.add_argument(
        "--help-info-list",
        help="List all arguments available for --help-info",
        action="store_true",
    )


# def init_stats_args(parser):
#     parser.add_argument(
#         "--confidence",
#         action="store",
#         default=[0.95],
#         type=float,
#         metavar="confidence",
#         help="Confidence value (default: %(default)s)",
#         nargs="+",
#     )
#     parser.add_argument(
#         "--distribution",
#         choices=stabilitest.statistics.distribution.get_distribution_names(),
#         metavar="distribution",
#         default="normal",
#         help="Distribution type (default: %(default)s)\n%(choices)s",
#     )
#     parser.add_argument(
#         "--parallel-fitting", action="store_true", help="Parallel fitting"
#     )


# def init_multiple_comparision_tests_args(parser):
#     parser.add_argument(
#         "--multiple-comparison-tests",
#         nargs="+",
#         metavar="test",
#         default=["fwe-bonferroni"],
#         choices=stabilitest.statistics.multiple_testing.get_method_names(),
#         help="Multiple comparison tests (default: %(default)s))\n%(choices)s",
#     )


# Global submodules (to load before domain submodules)
def init_module_configurator(parser):
    msg = """
    Submodule for configurator
    """
    subparser = parser.add_parser("configurator", description=msg, help=msg)
    return subparser


def init_module_single_test(parser):
    msg = """
    Submodule for single test
    """
    subparser = parser.add_parser("single-test", description=msg, help=msg)
    init_global_args(subparser)
    # init_stats_args(subparser)
    # init_multiple_comparision_tests_args(subparser)

    return subparser


def init_module_normality(parser):
    msg = """
    Submodule for normality test
    """
    subparser = parser.add_parser("normality", description=msg, help=msg)
    init_global_args(subparser)
    subparser.add_argument(
        "--confidence",
        action="store",
        default=[0.95],
        type=float,
        nargs="+",
        help="Confidence value (default: %(default)s)",
    )
    return subparser


def init_module_distance(parser):
    msg = """
    Submodule for computing various distances
    """
    subparser = parser.add_parser("distance", description=msg, help=msg)
    init_global_args(subparser)
    return subparser


def init_module_stats(parser):
    msg = """
    Submodule for statistics (mean, std, sig)
    """
    subparser = parser.add_parser("stats", description=msg, help=msg)
    init_global_args(subparser)
    subparser.set_defaults(output="stats")

    return subparser


def init_module_domain_list(parser):
    msg = """
    Submodule for listing available domains
    """
    subparser = parser.add_parser(
        "list-domain", description=msg, help=msg, add_help=False
    )
    return subparser


_kta_description = """
Keep-them-all (KTA)
-----------------
Cross-validation that tests that the reference interval computed contains each
reference observation. The reference interval is computed by using the all
observations (keep-them-all), including the one being tested.
"""


_loo_description = """
Leave-one-out (LOO)
-------------------
Sanity check that tests that the reference interval computed contains each
reference observation. The reference interval is computed by using the all
observations, excluding the one being tested.
"""


_kfold_description = """
K-fold
-------------------
Sanity check that tests that the reference interval (train set)
computed contain reference observations (test set).
The train/test is splitted with a 80/20 ratio and
is done K times.
"""


cross_validation_models = ["kta", "loo", "kfold"]


def init_module_cross_validation(parser):
    msg = """
    Submodule for cross-validation
    """

    epilog = "---"
    epilog += """
Models
===================
"""

    epilog += "".join([_kta_description, _loo_description, _kfold_description])
    epilog += """
Multiple comparison tests
=========================
    """
    epilog += stabilitest.statistics.multiple_testing.get_description()
    epilog = indent(epilog, " " * 2)
    subparser = parser.add_parser(
        "cross-validation",
        description=msg,
        epilog=epilog,
        help=msg,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    init_global_args(subparser)
    # init_stats_args(subparser)
    # init_multiple_comparision_tests_args(subparser)

    subparser.add_argument(
        "--model",
        required=True,
        choices=cross_validation_models,
        metavar="model",
        help="Model to perform: %(choices)s",
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
    "list-domain": init_module_domain_list,
    "configurator": init_module_configurator,
    "single-test": init_module_single_test,
    "cross-validation": init_module_cross_validation,
    "normality": init_module_normality,
    "stats": init_module_stats,
    "distance": init_module_distance,
}


def init_domain_modules(parser, required=True):
    domain_subparser = parser.add_subparsers(
        title="Domain submodules",
        help="Domain submodules",
        dest="domain",
        metavar="",
        required=required,
    )
    for domain_init in domain_modules.values():
        domain_init(parser, domain_subparser)


def init_analysis_modules(parser):
    subparser = parser.add_subparsers(
        title="Analysis submodules",
        help="Analysis submodules",
        dest="analysis",
        metavar="",
    )
    for analysis_init in analysis_modules.values():
        analysis_parser = analysis_init(subparser)
        # init_domain_modules(
        #     analysis_parser, required=analysis_init != init_module_domain_list
        # )


def parse_args(args):
    usage = "stabiltiest <analysis> <domain> [options]"

    parser = argparse.ArgumentParser(
        description="stabilitest", prog="stabilitest", usage=usage
    )
    init_analysis_modules(parser)
    domain = parser.add_argument_group("Domain submodules")
    domain.add_argument(
        "domain",
        metavar="",
        help="Domain submodules.\nChoices: %(choices)s",
        choices=["smri", "numpy"],
    )
    init_global_args(parser)

    known_args, _ = parser.parse_known_args(args)

    return parser, known_args
