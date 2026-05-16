try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("exo-toolkit")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except Exception:
    __version__ = "0.1.0"
