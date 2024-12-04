from numpy import __version__ as numpy_version
from numpy.lib import NumpyVersion

from .eigency_tests import *

if NumpyVersion(numpy_version) >= "2.0.0":
    from .eigency_np2 import *
