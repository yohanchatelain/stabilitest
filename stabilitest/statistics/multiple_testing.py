import warnings

import stabilitest.pprinter as pprinter
import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


def pce_test(reject, tests, alpha):
    b = binomtest(k=reject, n=tests, p=alpha)
    return b.pvalue >= alpha


def pce(target, alpha, p_values):
    """
    Compute the Per-Comparison Error rate (uncorrected)
    """
    name = "PCE"
    size = p_values.size
    threshold = alpha
    reject = p_values < threshold
    nb_reject = np.ma.sum(reject)
    ratio = nb_reject / size
    passed = ratio <= alpha

    if pprinter.verbose():
        pprinter.print_name_method("Per-Comparison Error (Uncorrected)")
        print(f"- Alpha                    = {threshold:f}")
        print(f"- Card(Reject)             = {nb_reject}")
        print(f"- Card(tests)              = {size}")
        print(f"- Card(Reject)/Card(tests) = {ratio:.2e} [{ratio*100:f}%]")
    pprinter.print_result(target, nb_reject, size, alpha, passed, name)

    return nb_reject, size, passed


def pce_sig(target, alpha, reject):
    """
    Compute the Per-Comparison Error rate (uncorrected) for significant bits
    """
    name = "PCE-sig"
    size = reject.size
    nb_reject = np.ma.sum(reject)
    ratio = nb_reject / size
    passed = ratio <= alpha

    if pprinter.verbose():
        pprinter.print_name_method("Per-Comparison Error (Uncorrected)")
        print(f"- Card(Reject)         = {nb_reject}")
        print(f"- Card(tests)          = {size}")
        print(f"- Card(FP)/Card(tests) = {ratio:.2e} [{ratio*100:f}%]")
    pprinter.print_result(target, nb_reject, size, alpha, passed, name)

    return nb_reject, size, passed


def mct(target, alpha, p_values, method, short_name, long_name, success_test):
    """
    Generic method for compute Multiple Comparison tests rate.
    """
    name = short_name
    size = p_values.size

    reject = None
    corrected_threshold = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (
            reject,
            _,
            corrected_threshold_sidak,
            corrected_threshold_bonferroni,
        ) = multipletests(p_values, alpha=alpha, method=method, is_sorted=True)

        if method == "bonferroni":
            corrected_threshold = corrected_threshold_bonferroni
        elif method == "sidak":
            corrected_threshold = corrected_threshold_sidak

    nb_reject = np.ma.sum(reject)
    ratio = nb_reject / size

    if pprinter.verbose():
        pprinter.print_name_method(long_name)
        ct = corrected_threshold
        if ct is not None:
            print(f"- Alpha correction     = {ct:f} ({ct:.3e})")
        print(f"- Card(FP)             = {nb_reject}")
        print(f"- Card(tests)          = {size}")
        print(f"- Card(FP)/Card(tests) = {ratio:.2e} [{ratio*100:f}%]")

    pprinter.print_result(
        target, nb_reject, size, alpha, success_test(nb_reject, size, alpha), name
    )

    return nb_reject, size, success_test(nb_reject, size, alpha)


def fwe_bonferroni(target, alpha, p_values):
    """
    Compute the failing tests ratio using the Bonferonni correction
    """

    def success_test(reject, tests, alpha):
        return reject < 1

    return mct(
        target=target,
        p_values=p_values,
        alpha=alpha,
        method="bonferroni",
        short_name="FWE-Bon",
        long_name="FWE (Bonferroni)",
        success_test=success_test,
    )


def fwe_sidak(target, alpha, p_values):
    """
    Compute the failing tests ratio using the Sidak correction
    """

    def success_test(reject, tests, alpha):
        return reject < 1

    return mct(
        target=target,
        p_values=p_values,
        alpha=alpha,
        method="sidak",
        short_name="FWE-Sidak",
        long_name="FWE (Sidak)",
        success_test=success_test,
    )


def fwe_holm_sidak(target, alpha, p_values):
    """
    Compute the failing tests ratio using the Holm-Sidak correction
    """

    def success_test(reject, tests, alpha):
        return reject < 1

    return mct(
        target=target,
        p_values=p_values,
        alpha=alpha,
        method="holm-sidak",
        short_name="FWE-HS",
        long_name="FWE (Holm-Sidak)",
        success_test=success_test,
    )


def fwe_holm_bonferroni(target, alpha, p_values):
    """
    Compute the failing tests ratio using the Holm-Bonferonni correction
    """

    def success_test(reject, tests, alpha):
        return reject < 1

    return mct(
        target=target,
        p_values=p_values,
        alpha=alpha,
        method="holm",
        short_name="FWE-HB",
        long_name="FWE (Holm-Bonferroni)",
        success_test=success_test,
    )


def fwe_simes_hochberg(target, alpha, p_values):
    """
    Compute the failing tests ratio using the Simes-Hochberg correction
    """

    def success_test(reject, tests, alpha):
        return reject < 1

    return mct(
        target=target,
        p_values=p_values,
        alpha=alpha,
        method="simes-hochberg",
        short_name="FWE-SH",
        long_name="FWE (Simes-Hochberg)",
        success_test=success_test,
    )


def fdr_BH(target, alpha, p_values):
    """
    Compute the failing tests ratio using the False Discovery Rate correction (Benjamini-Hochberg)
    """

    def success_test(reject, tests, alpha):
        return reject < 1

    return mct(
        target=target,
        p_values=p_values,
        alpha=alpha,
        method="fdr_bh",
        short_name="FDR-BH",
        long_name="FDR (Benjamini-Hochberg)",
        success_test=success_test,
    )


def fdr_BY(target, alpha, p_values):
    """
    Compute the failing tests ratio using the False Discovery Rate correction (Benjamini-Yekutieli)
    """

    def success_test(reject, tests, alpha):
        return reject < 1

    return mct(
        target=target,
        p_values=p_values,
        alpha=alpha,
        method="fdr_by",
        short_name="FDR-BY",
        long_name="FDR (Benjamini-Yekutieli)",
        success_test=success_test,
    )


def fdr_TSBH(target, alpha, p_values):
    """
    Compute the failing tests ratio using the False Discovery Rate correction (Two-stage Benjamini-Hochberg)
    """

    def success_test(reject, tests, alpha):
        return reject < 1

    return mct(
        target=target,
        p_values=p_values,
        alpha=alpha,
        method="fdr_tsbh",
        short_name="FDR-TSBH",
        long_name="FDR (Two-Stage Benjamini-Hochberg)",
        success_test=success_test,
    )


def fdr_TSBY(target, alpha, p_values):
    """
    Compute the failing tests ratio using the False Discovery Rate correction (Two-Stage Benjamini-Yekutieli)
    """

    def success_test(reject, tests, alpha):
        return reject < 1

    return mct(
        target=target,
        p_values=p_values,
        alpha=alpha,
        method="fdr_tsbky",
        short_name="FDR-TSBY",
        long_name="FDR (Two-Stage Benjamini-Yekutieli)",
        success_test=success_test,
    )


___methods = {
    "pce": pce,
    "fdr-TSBY": fdr_TSBY,
    "fdr-TSBH": fdr_TSBH,
    "fdr-BY": fdr_BY,
    "fdr-BH": fdr_BH,
    "fwe-simes-hochberg": fwe_simes_hochberg,
    "fwe-holm-bonferroni": fwe_holm_bonferroni,
    "fwe-holm-sidak": fwe_holm_sidak,
    "fwe-sidak": fwe_sidak,
    "fwe-bonferroni": fwe_bonferroni,
}

__description = """
Per-Comparison Error (PCE)
----------------------------
The PCE is the ratio of the number of false positives to the number of tests.

Family-Wise Error rate (FWE)
----------------------------
Probability of making a Type I error among a family of tests

  - fwe-bonferroni: Bonferroni correction
  - fwe-sidak: Sidak correction
  - fwe-holm-bonferroni: Holm-Bonferroni correction
  - fwe-holm-sidak: Holm-Sidak correction
  - fwe-simes-hochberg: Simes-Hochberg correction

False Discovery Rate (FDR)
----------------------------
Expected proportion of "discoveries" (rejected null hypotheses) that are false (incorrect rejections of the null)

  - fdr-BH: Benjamini-Hochberg correction
  - fdr-BY: Benjamini-Yekutieli correction
  - fdr-TSBH: Two-Stage Benjamini-Hochberg correction
  - fdr-TSBY: Two-Stage Benjamini-Yekutieli correction
"""


def get_description():
    return __description


def get_method_names():
    return ___methods.keys()


def get_methods(args):
    return [___methods[name] for name in args.multiple_comparison_tests]
