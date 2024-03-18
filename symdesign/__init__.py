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
from .version import version, __version__
