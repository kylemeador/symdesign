from __future__ import annotations

import os
from itertools import chain

from symdesign.utils.path import rosetta_main, rosetta_extras, dalphaball, rosetta_scripts_dir, sym_weights, \
    solvent_weights_sym, solvent_weights

min_cores_per_job = 1  # currently one for the MPI node, and 5 workers
num_thread_per_process = 2
reference_average_residue_weight = 3  # for REF2015
run_cmds = {'default': '',
            'python': '',
            'cxx11thread': '',
            'mpi': ['mpiexec', '--oversubscribe', '-np'],
            'cxx11threadmpi': ['mpiexec', '--oversubscribe', '-np',
                               str(int(min_cores_per_job / num_thread_per_process))]}  # TODO Optimize
extras_flags = {'default': [],
                'python': [],
                'cxx11thread': [f'-multithreading:total_threads {num_thread_per_process}',
                                f'-multithreading:interaction_graph_threads {num_thread_per_process}'],
                'mpi': [],
                'cxx11threadmpi': [f'-multithreading:total_threads {num_thread_per_process}']}
script_cmd = [os.path.join(rosetta_main, 'source', 'bin', f'rosetta_scripts.{rosetta_extras}.linuxgccrelease'),
              '-database', os.path.join(rosetta_main, 'database')]
rosetta_flags = extras_flags[rosetta_extras] + \
    ['-ex1', '-ex2', '-extrachi_cutoff 5', '-ignore_unrecognized_res', '-ignore_zero_occupancy false',
     # '-overwrite',
     '-linmem_ig 10', '-out:file:scorefile_format json', '-output_only_asymmetric_unit true', '-no_chainend_ter true',
     '-write_seqres_records true', '-output_pose_energies_table false', '-output_pose_cache_data false',
     f'-holes:dalphaball {dalphaball}' if os.path.exists(dalphaball) else '',  # This creates a new line if not used
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
rosetta_variables = [('scripts', rosetta_scripts_dir), ('sym_score_patch', sym_weights),
                     ('solvent_sym_score_patch', solvent_weights_sym),
                     ('solvent_score_patch', solvent_weights)]
