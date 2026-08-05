from .factor import Factor as Factor
from .factor import asfactor as asfactor
from .factor import gl as gl
from .format import round_scientific_notation as round_scientific_notation
from .format import truncate_name as truncate_name
from .linalg import cov2cor as cov2cor
from .lm import lm_fit as lm_fit
from .lm import lm_wfit as lm_wfit
from .logging import LOGGER as LOGGER
from .splines import ns as ns
from .splines import spline_design as spline_design
# stats.py needs the compiled stats_cpp extension (Cython, not vendored --
# see ../../README.md) and none of its symbols (dnbinom_mu/dnorm/pnorm/pt/
# rnbinom, all NB-distribution helpers for edgeR/DESeq2/ComBat-seq style
# count models) are used by limma, so it's omitted from this vendored copy.
