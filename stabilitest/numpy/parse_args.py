import icecream as ic


def init_global_args(parser):
    parser.add_argument("--reference", required=True, help="Reference path")
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize values between [0,1]"
    )


def init_test_args(parser):
    parser.add_argument("--target", required=True, help="Target path")


def init_module(parser):
    numpy_group = parser.add_argument_group("NumPy ndarray options", "")
    init_global_args(numpy_group)

    if parser.prog == "stabilitest test":
        init_test_args(numpy_group)

    ic(parser)
