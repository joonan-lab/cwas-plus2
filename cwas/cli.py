"""
Command line tool for Category-wide association study (CWAS)
"""
import sys

import cwas.factory
from cwas.utils.log import print_log

AVAILABLE_STEPS = [
    "start",
    "configuration",
    "preparation",
    "annotation",
    "categorization",
    "binomial_test",
    "permutation_test",
    "burden_shift",
    "effective_num_test",
    "correlation",
    "risk_score",
    "dawn",
    "extract_variant",
]


def main():
    print_log("LOG", "Category-Wide Association Study (CWAS)")
    if len(sys.argv) < 2:
        print(
            "Usage: cwas <step> [options]\n\n"
            "Available steps:\n  " + "\n  ".join(AVAILABLE_STEPS),
            file=sys.stderr,
        )
        sys.exit(1)
    cwas_factory = cwas.factory.create(sys.argv[1])
    cwas_args = cwas_factory.argparser().parse_args(sys.argv[2:])
    cwas_obj = cwas_factory.runnable
    print_log("LOG", f"Current step: {cwas_obj.__name__}")
    cwas_inst = cwas_obj(cwas_args)
    return cwas_inst
