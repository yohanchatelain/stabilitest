#!/usr/bin/env python3
# coding: utf-8

import warnings

from icecream import ic

import stabilitest
import stabilitest.mri.sample
import stabilitest.model as model
import stabilitest.mri.distance as mri_distance
import stabilitest.normality
import stabilitest.parse_args as parse_args
import stabilitest.pprinter as pprinter
import stabilitest.snr as snr
from stabilitest.collect import stats_collect

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_reference_sample(args):
    if args.application == "mri":
        return stabilitest.mri.sample.get_reference_sample(args)
    if args.application == "numpy":
        return stabilitest.numpy.sample.get_reference_sample(args)


def get_target_sample(args):
    if args.application == "mri":
        return stabilitest.mri.sample.get_target_sample(args)
    if args.application == "numpy":
        return stabilitest.numpy.sample.get_target_sample(args)


def run_kta(args):
    sample = get_reference_sample(args)
    sample.load()
    fvr = model.run_kta(args, sample)
    return fvr


def run_loo(args):
    sample = get_reference_sample(args)
    sample.load()
    fvr = model.run_loo(args, sample)
    return fvr


def run_one(args):
    reference_sample = get_reference_sample(args)
    target_sample = get_target_sample(args)
    reference_sample.load()
    target_sample.load()
    fvr = model.run_one(args, reference_sample, target_sample)
    return fvr


def run_normality(args):
    sample = get_reference_sample(args)
    sample.load()
    normality = stabilitest.normality.run_normality_test(args, sample)
    print(normality)


def run_k_fold(args):
    sample = get_reference_sample(args)
    sample.load()
    fvr = model.run_kfold(args, sample)
    return fvr


def run_stats(args):
    sample = get_reference_sample(args)
    sample.load()
    stabilitest.statistics.stats.compute_stats(args, sample)


def run_distance(args):
    sample = get_reference_sample(args)
    sample.load()
    mri_distance.main(args, sample)


def run_snr(args):
    sample = get_reference_sample(args)
    sample.load()
    snr.main(args)


tests = {
    "kta": run_kta,
    "loo": run_loo,
    "k-fold": run_k_fold,
    "one": run_one,
    "normality": run_normality,
    "stats": run_stats,
    "distance": run_distance,
    "snr": run_snr,
}


def main():
    ic.configureOutput(includeContext=True)

    parser, parsed_args = parse_args.parse_args()

    if parsed_args.verbose:
        pprinter.enable_verbose_mode()

    if parsed_args.analysis not in stabilitest.parse_args.analysis_modules:
        parser.print_help()
        return

    tests[parsed_args.analysis](parsed_args)
    stats_collect.set_name(parsed_args.output)
    stats_collect.dump()


if "__main__" == __name__:
    main()
