from .complex_real2 import from_real2, to_real2

try:
    # `matplotlib` is an optional dependency
    from . import plotting
except ModuleNotFoundError as exc:
    if "matplotlib" not in exc.msg:  # pylint: disable=unsupported-membership-test
        raise exc
