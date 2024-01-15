def init_global_args(parser):
    parser.add_argument("--config-file", action="store", default=None)


def init_module(parser, subparser):
    msg = "Submodule for numpy data"
    numpy_parser = subparser.add_parser("numpy", description=msg, help=msg)
    init_global_args(numpy_parser)
