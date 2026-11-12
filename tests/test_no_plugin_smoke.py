import pytest

from scripts.no_plugin_smoke import run


@pytest.mark.no_plugins
def test_no_plugin_smoke_script_runs():
    assert run() == 0
