from pathlib import Path


_dependencies_dir = Path(__file__).parent
freesasa_config_path = _dependencies_dir / 'freesasa-2.0.config'
freesasa_debug_dir = _dependencies_dir / 'freesasa_debug'
orient_dir_path = _dependencies_dir / 'orient'
orient_exe_path = orient_dir_path / 'orient_oligomer'
orient_symmetry_files_dir = orient_dir_path / 'symm_files'
errat_exe_path = _dependencies_dir / 'errat' / 'errat'
errat_residue_path = _dependencies_dir / 'errat' / 'errat_every_residue.cpp'
stride_dir = _dependencies_dir / 'stride'
stride_exe_path = stride_dir / 'stride'
freesasa_dir = _dependencies_dir / 'freesasa'
freesasa_exe_path = freesasa_dir / 'src' / 'freesasa'
