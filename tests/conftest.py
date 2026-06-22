def pytest_configure(config):
    """Keep benchmark reports compact when pytest-benchmark is active."""
    columns = getattr(config.option, "benchmark_columns", None)
    if columns == ["min", "max", "mean", "stddev", "median", "iqr", "outliers", "ops", "rounds", "iterations"]:
        config.option.benchmark_columns = ["min", "mean", "stddev", "rounds"]

    if getattr(config.option, "benchmark_name", None) == "normal":
        config.option.benchmark_name = "short"

    if getattr(config.option, "benchmark_time_unit", None) is None:
        config.option.benchmark_time_unit = "us"
