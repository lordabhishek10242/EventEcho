"""Smoke tests for the reproducible historical rebuild entry point."""


def test_historical_rebuild_entry_point_imports_without_running_network_calls():
    from scripts.rebuild_historical import main

    assert callable(main)
