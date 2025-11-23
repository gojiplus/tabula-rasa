"""Version information for tabula-rasa."""

try:
    from importlib.metadata import version

    __version__ = version("tabula-rasa")
except ImportError:
    # Fallback for development or if package not installed
    __version__ = "0.1.0"

__version_info__ = tuple(int(i) for i in __version__.split(".") if i.isdigit())
