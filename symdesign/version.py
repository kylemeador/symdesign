"""Single source of version."""
import importlib.metadata

version = importlib.metadata.version('symdesign')
__version__ = version
