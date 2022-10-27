import torchhd.functional as functional
import torchhd.embeddings as embeddings
import torchhd.structures as structures
import torchhd.datasets as datasets
import torchhd.utils as utils

from torchhd.base import VSA_Model
from torchhd.bsc import BSC
from torchhd.map import MAP
from torchhd.hrr import HRR
from torchhd.fhrr import FHRR


from torchhd.primitives import (
    add,
    badd,
    mul,
    bmul,
    randsel,
    brandsel,
    shift,
    bind,
    multibind,
    bundle,
    multibundle,
    permute,
)
from torchhd.functional import (
    identity_hv,
    random_hv,
    level_hv,
    circular_hv,
    bind,
    unbind,
    bundle,
    permute,
    cos_similarity,
    dot_similarity,
    ham_similarity,
)

from torchhd.version import __version__

__all__ = [
    "__version__",
    "VSA_Model",
    "BSC",
    "MAP",
    "HRR",
    "FHRR",
    "add",
    "badd",
    "mul",
    "bmul",
    "randsel",
    "brandsel",
    "shift",
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
    "cosine_similarity",
    "dot_similarity",
    "ham_similarity",
]
