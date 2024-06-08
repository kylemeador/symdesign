from __future__ import annotations

import os
from collections import defaultdict
from itertools import chain
from pathlib import Path

from symdesign.utils.path import config

# Rosetta scripts and miscellaneous dependency files
exe_make = config.get('rosetta_make')
main_dir = os.environ.get(config.get('rosetta_env'), None)
if main_dir is None:
    main_dir = 'main'
main_dir = Path(main_dir)
source_dir = main_dir / 'source'
# rosetta_default_bin_path = source_dir / 'bin'
scripts_dir = Path(__file__).parent
make_sdf_path = source_dir / 'src' / 'apps' / 'public' / 'symmetry' / 'make_symmdef_file.pl'
symmetry_definition_dir = scripts_dir / 'sdf'
dalphaball_exe_path = source_dir / 'external' / 'DAlpahBall' / 'DAlphaBall.gcc'  # Wrong spelling is correct
min_cores_per_job = 1  # Currently one for the MPI node, and 5 workers
num_thread_per_process = 2
current_energy_function = 'REF2015'
reference_average_residue_weight = 3  # For REF2015
run_cmds = defaultdict(
    list, {'default': [], 'python': [], 'cxx11thread': [], 'mpi': ['mpiexec', '--oversubscribe', '-np'],
           'cxx11threadmpi': ['mpiexec', '--oversubscribe', '-np', str(int(min_cores_per_job /
                                                                           num_thread_per_process))]})
extras_flags = defaultdict(
    list, {'default': [], 'python': [], 'cxx11thread': [f'-multithreading:total_threads {num_thread_per_process}',
                                                        '-multithreading:interaction_graph_threads '
                                                        f'{num_thread_per_process}'],
           'mpi': [], 'cxx11threadmpi': [f'-multithreading:total_threads {num_thread_per_process}']})
script_cmd = [os.path.join(main_dir, 'source', 'bin', f'rosetta_scripts.{exe_make}.linuxgccrelease'),
              '-database', os.path.join(main_dir, 'database')]
flags = extras_flags[exe_make] + \
    ['-ex1', '-ex2', '-extrachi_cutoff 5', '-ignore_unrecognized_res', '-ignore_zero_occupancy false',
     # '-overwrite',
     '-linmem_ig 10', '-out:file:scorefile_format json', '-output_only_asymmetric_unit true', '-no_chainend_ter true',
     '-write_seqres_records true', '-output_pose_energies_table false', '-output_pose_cache_data false',
     f'-holes:dalphaball_exe_path {dalphaball_exe_path}' if dalphaball_exe_path.exists() else '',
     # '' creates a new line if not used
     '-use_occurrence_data',  # Todo integrate into xml with Rosetta Source update
     '-preserve_header true', '-write_pdb_title_section_records true',
     '-chemical:exclude_patches LowerDNA UpperDNA Cterm_amidation SpecialRotamer VirtualBB ShoveBB VirtualNTerm '
     'VirtualDNAPhosphate CTermConnect sc_orbitals pro_hydroxylated_case1 N_acetylated C_methylamidated cys_acetylated'
     'pro_hydroxylated_case2 ser_phosphorylated thr_phosphorylated tyr_phosphorylated tyr_diiodinated tyr_sulfated'
     'lys_dimethylated lys_monomethylated lys_trimethylated lys_acetylated glu_carboxylated MethylatedProteinCterm',
     '-mute all', '-unmute protocols.rosetta_scripts.ParsedProtocol protocols.jd2.JobDistributor']
relax_pairs = ['-relax:ramp_constraints false', '-no_optH false', '-relax:coord_cst_stdev 0.5',
               '-nblist_autoupdate true', '-no_nstruct_label true', '-relax:bb_move false']  # Todo remove this one?
relax_singles = ['-constrain_relax_to_start_coords', '-use_input_sc', '-relax:coord_constrain_sidechains', '-flip_HNQ',
                 '-no_his_his_pairE']
relax_flags = relax_singles + relax_pairs
relax_flags_cmdline = relax_singles + list(chain.from_iterable(map(str.split, relax_pairs)))
sym_weights = 'ref2015_sym.wts_patch'
solvent_weights = 'ref2015_solvent.wts_patch'
solvent_weights_sym = 'ref2015_sym_solvent.wts_patch'
variables = [('scripts', str(scripts_dir)), ('sym_score_patch', sym_weights),
             ('solvent_sym_score_patch', solvent_weights_sym),
             ('solvent_score_patch', solvent_weights)]
design_profile = 'design_profile'
hbnet_design_profile = 'hbnet_design_profile'
refine = 'refine'
structure_background = 'structure_background'
scout = 'scout'
hbnet_scout = 'hbnet_scout'
protocol = 'protocol'
protocols_of_interest = {design_profile, structure_background, hbnet_design_profile}
# protocols_of_interest = ['combo_profile', 'limit_to_profile', 'no_constraint']  # Used for P432 models
