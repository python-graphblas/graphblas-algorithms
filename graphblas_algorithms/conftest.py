import pytest


@pytest.fixture(scope="session", autouse=True)
def ic():
    """Make `ic` available everywhere during testing for easier debugging"""
    try:
        import icecream
    except ImportError:
        return
    icecream.install()
    # icecream.ic.disable()  # do ic.enable() to re-enable
    return icecream.ic
