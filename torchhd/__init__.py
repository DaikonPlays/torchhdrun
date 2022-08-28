import torchhd.primitives as primitives
import torchhd.functional as functional
import torchhd.embeddings as embeddings
import torchhd.structures as structures
import torchhd.datasets as datasets
import torchhd.utils as utils

from torchhd.primitives import (
    bind,
    multibind,
    set_bind_method,
    bundle,
    multibundle,
    set_bundle_method,
    permute,
    set_permute_method,
)
from torchhd.functional import (
    identity_hv,
    random_hv,
    level_hv,
    circular_hv,
    unbind,
    cosine_similarity,
    dot_similarity,
)

from torchhd.version import __version__

__all__ = [
    "primitives",
    "functional",
    "embeddings",
    "structures",
    "datasets",
    "utils",
    "identity_hv",
    "random_hv",
    "level_hv",
    "circular_hv",
    "bind",
    "multibind",
    "set_bind_method",
    "unbind",
    "bundle",
    "multibundle",
    "set_bundle_method",
    "permute",
    "set_permute_method",
    "cosine_similarity",
    "dot_similarity",
]
