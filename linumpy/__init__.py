# Configure thread limits FIRST, before any numerical libraries are imported
import os as _os

from linumpy._thread_config import (
    apply_threadpool_limits as apply_threadpool_limits,
)
from linumpy._thread_config import (
    configure_all_libraries as configure_all_libraries,
)
from linumpy._thread_config import (
    configure_sitk as configure_sitk,
)
from linumpy._thread_config import (
    configure_thread_limits as configure_thread_limits,
)


def get_home():
    """Set a user-writeable file-system location to put files."""
    if "LINUMPY_HOME" in _os.environ:
        return _os.environ["LINUMPY_HOME"]
    return _os.path.join(_os.path.expanduser("~"), ".linumpy")


def get_root():
    return _os.path.realpath(f"{_os.path.dirname(_os.path.abspath(__file__))}/..")


LINUMPY_HOME = get_home()
LINUMPY_ROOT = get_root()

# Apply runtime thread pool limits (for libraries that don't respect env vars)
apply_threadpool_limits()

# Note: configure_sitk() must be called after SimpleITK is imported.
# Scripts that use SimpleITK should call:
#   from linumpy._thread_config import configure_all_libraries
#   configure_all_libraries()  # after all imports
