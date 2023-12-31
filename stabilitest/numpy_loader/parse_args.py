import icecream as ic


def init_global_args(parser):
    parser.add_argument("--reference", required=True, help="Reference path")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize values between [0, 1]",
    )


def init_test_args(parser):
    parser.add_argument("--target", required=True, help="Target path")


def init_module(parser, subparser):
    numpy_parser = subparser.add_parser("numpy")
    init_global_args(numpy_parser)

    if parser.prog == "stabilitest test":
        init_test_args(numpy_parser)
