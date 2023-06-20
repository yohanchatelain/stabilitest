from icecream import ic


def init_global_args(parser):
    parser.add_argument(
        "--datatype", action="store", default="anat", choices=["anat"], help="Data type"
    )

    parser.add_argument(
        "--reference-prefix",
        action="store",
        required=True,
        help="Reference prefix path",
    )
    parser.add_argument(
        "--reference-dataset",
        action="store",
        required=True,
        help="Dataset reference",
    )
    parser.add_argument(
        "--reference-subject",
        action="store",
        required=True,
        help="Subject reference",
    )
    parser.add_argument(
        "--reference-template",
        action="store",
        required=True,
        help="Reference template",
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
        choices=["union", "intersection", "map"],
        default="union",
        help="Method to combine brain mask (map applies each brain mask to the image repetition)",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the T1w to have [0,1] intensities",
    )

    parser.add_argument(
        "--mask-non-normal-voxels",
        action="store_true",
        help="Mask voxels that do not pass the normality test (Shapiro-Wilk)",
    )


def init_test_args(parser):
    parser.add_argument(
        "--target-prefix", action="store", required=True, help="Target prefix path"
    )
    parser.add_argument(
        "--target-dataset", action="store", required=True, help="Dataset target"
    )
    parser.add_argument(
        "--target-subject", action="store", required=True, help="Subject target"
    )
    parser.add_argument(
        "--target-template", action="store", required=True, help="Target template"
    )


def init_module(parser, subparser):
    smri_parser = subparser.add_parser("smri")
    init_global_args(smri_parser)

    if parser.prog == "stabilitest test":
        init_test_args(smri_parser)
