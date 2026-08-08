"""Welcome to the AI slop containment chamber submodule of torchalgos.

For now, all code outside of this submodule remains entirely hand-written.
But if a sloptimizer turns out good, it will be moved out of this submodule,
infecting the pristine lands.

So far no sloptizer has come even close to breaching the chamber.
There is one saving grace for all of those. Extensive hyperparameter tuning.
It is possible that one of those is solid, but has bad default hyperparameters."""
from .qnkfac import KFSRC, KronCBFGS
from .spectral_splus import SpectralSPlus
from .prism import Prism
from .hso import HSO
from .komo import KOMO
from .ortho_adam import OrthoAdam
from .aether import Aether