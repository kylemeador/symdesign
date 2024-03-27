"""
#### Model, design, and analyze proteins without explicit considerations for applying local and global symmetry.

Modules exported by this package:

- [flags][]: Setup program inputs to perform a job
- [metrics][]: Perform calculations on a protein pose
- [protocols][]: Implement a defined set of instructions for a protein pose
- [resources][]: Common methods and variable to connect job instructions to protocol implementation during job runtime
- [sequence][]: Handle biological sequences as python objects
- [structure][]: Handle biological structures as python objects
- [tools][]: Helper scripts, not quite yet a [protocols][protocols] worthy module, perhaps only accomplishes a sinlge task like file manipulation
- [utils][]: Miscellaneous functions, methods, and tools for all modules
- [visualization][]: Miscellaneous PyMol visualization helper functions and plotting config.
"""
import importlib.metadata
from pathlib import Path

source_dir = Path(__file__).parent
try:
    __version__ = importlib.metadata.version(str(source_dir.name))
except importlib.metadata.PackageNotFoundError as exc:
    first_error = f"Unable to find the package '{source_dir.name}'. Have you installed it?"
    try:  # to parse from packaging source
        with open(source_dir.parent / 'pyproject.toml', 'r') as f:
            package_info = f.read()

        name_str = 'name = '
        name_info = package_info[package_info.find(name_str) + len(name_str) + 1:]
        pyproject_source_name = name_info.split()[0].strip("'").strip('"')
        if source_dir.name != pyproject_source_name:
            # This isn't the right file
            raise FileNotFoundError
        version_str = 'version = '
        version_info = package_info[package_info.find(version_str) + len(version_str) + 1:]
        __version__ = version_info.split()[0].strip("'").strip('"')
    except FileNotFoundError as exc2:
        print(first_error)
        print("Falling back on pyproject.toml also failed")
        raise RuntimeError(
            f"Unable to parse '{source_dir.name}.version'..."
        ) from exc
version = __version__
