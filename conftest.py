import pytest


def pytest_addoption(parser):
    parser.addoption("--hf-auth", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--hf-auth"):
        skip = pytest.mark.skip(reason="pass --hf-auth to run")
        for item in items:
            if item.get_closest_marker("requires_hf_auth"):
                item.add_marker(skip)
