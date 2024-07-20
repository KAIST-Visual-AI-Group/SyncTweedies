from pathlib import Path

import numpy as np
from PIL import Image

from synctweedies.renderer.visual_anagrams.views.view_inner_rotate import \
    InnerRotateView

from .view_flip import FlipView
from .view_identity import IdentityView
from .view_inner_circle import InnerCircleView
from .view_jigsaw import JigsawView
from .view_negate import NegateView
from .view_patch_permute import PatchPermuteView
from .view_rotate import (Rotate90CCWView, Rotate90CWView, Rotate180View,
                          RotateCWView)
from .view_skew import SkewView
from .view_square_hinge import SquareHingeView

VIEW_MAP = {
    "identity": IdentityView,
    "flip": FlipView,
    "rotate_cw": Rotate90CWView,
    "rotate_ccw": Rotate90CCWView,
    "rotate_180": Rotate180View,
    "negate": NegateView,
    "skew": SkewView,
    "patch_permute": PatchPermuteView,
    "pixel_permute": PatchPermuteView,
    "jigsaw": JigsawView,
    "inner_circle": InnerCircleView,
    "square_hinge": SquareHingeView,
    "inner_rotate": InnerRotateView,
}


def get_views(view_names, *arg):
    """
    Bespoke function to get views (just to make command line usage easier)
    """
    views = []
    for view_name in view_names:
        if view_name == "patch_permute":
            args = [8]
        elif view_name == "pixel_permute":
            args = [64]
        elif view_name == "skew":
            args = [1.5]
        elif view_name == "inner_rotate":
            args = arg[:1]
        else:
            args = []

        view = VIEW_MAP[view_name](*args)
        views.append(view)

    return views
