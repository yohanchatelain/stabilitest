def init_global_args(parser):
    parser.add_argument("--config-file", action="store", default=None)
    parser.add_argument(
        "--datatype", action="store", default="anat", choices=["anat"], help="Data type"
    )

    parser.add_argument(
        "--smooth-kernel",
        "--fwhm",
        action="store",
        metavar="fwhm",
        type=float,
        default=0.0,
        help="Size of the kernel smoothing",
    )

    parser.add_argument(
        "--mask-combination",
        action="store",
        type=str,
        choices=["union", "intersection", "identity"],
        default="union",
        help="Method to combine brain mask (default %(default)s)\n'identity' uses each voxel mask",
    )

    parser.add_argument(
        "--normalize",
        "--min-max-normalization",
        action="store_true",
        help="Min-max normalization to have voxel intensities between [0,1]",
    )

    parser.add_argument(
        "--mask-non-normal-voxels",
        action="store_true",
        help="Mask voxels that do not pass the normality test (Shapiro-Wilk)",
    )


def init_module(parser, subparser):
    msg = "Submodule for Structural MRI data"
    smri_parser = subparser.add_parser("smri", description=msg, help=msg)
    init_global_args(smri_parser)
