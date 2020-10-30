import os

# from multiprocessing import cpu_count
import PathUtils as PUtils

min_cores_per_job = 1  # currently one for the MPI node, and 5 workers
mpi = 6
num_thread_per_process = 2
hhblits_threads = 1
reference_average_residue_weight = 3  # for REF2015

run_cmds = {'default': '',
            'python': '',
            'cxx11thread': '',
            'mpi': ['mpiexec', '-np', str(int(mpi))],  # TODO Optimize
            'cxx11threadmpi': ['mpiexec', '-np', str(int(min_cores_per_job / num_thread_per_process))]}  # TODO Optimize

extras_flags = {'default': [],
                'python': [],
                'cxx11thread': ['-multithreading:total_threads ' + str(num_thread_per_process),
                                '-multithreading:interaction_graph_threads ' + str(num_thread_per_process)],
                'mpi': [],
                'cxx11threadmpi': ['-multithreading:total_threads ' + str(num_thread_per_process)]}

script_cmd = [os.path.join(PUtils.rosetta, 'source/bin/rosetta_scripts.' + PUtils.rosetta_extras + '.linuxgccrelease'),
              '-database', os.path.join(PUtils.rosetta, 'database')]
flags = extras_flags[PUtils.rosetta_extras] + \
        ['-ex1', '-ex2', '-extrachi_cutoff 5', '-ignore_unrecognized_res',  # '-run:timer true',
         '-ignore_zero_occupancy false', '-overwrite', '-linmem_ig 10', '-out:file:scorefile_format json',
         '-output_only_asymmetric_unit true', '-no_chainend_ter true', '-write_seqres_records true',
         '-output_pose_energies_table false', '-output_pose_cache_data false',
         '-chemical:exclude_patches LowerDNA UpperDNA Cterm_amidation SpecialRotamer VirtualBB ShoveBB VirtualNTerm '
         'VirtualDNAPhosphate CTermConnect sc_orbitals pro_hydroxylated_case1 N_acetylated C_methylamidated '
         'cys_acetylated pro_hydroxylated_case2 ser_phosphorylated thr_phosphorylated tyr_phosphorylated tyr_sulfated '
         'lys_dimethylated lys_monomethylated lys_trimethylated lys_acetylated glu_carboxylated MethylatedProteinCterm '
         'tyr_diiodinated']

# PUtils.stage = 1: refine, 2: design, 3: metrics, 4:analysis, 5:consensus. 6:rmsd_calculation
# 1 and 5 have the same flag options as both are relax
flag_options = {PUtils.stage[1]: ['-constrain_relax_to_start_coords', '-use_input_sc', '-relax:ramp_constraints false',
                                  '-no_optH false', '-relax:coord_constrain_sidechains', '-relax:coord_cst_stdev 0.5',
                                  '-no_his_his_pairE', '-flip_HNQ', '-nblist_autoupdate true', '-no_nstruct_label true',
                                  '-relax:bb_move false',  # '-out:suffix _' + PUtils.stage[1], '-o + PUtils.stage[1]],
                                  '-mute all', '-unmute protocols.rosetta_scripts.ParsedProtocol'],
                PUtils.stage[2]: ['-use_occurrence_data', '-out:suffix _' + PUtils.stage[2],  # '-o + PUtils.stage[2]],
                                  '-mute all', '-unmute protocols.rosetta_scripts.ParsedProtocol'],  # -holes:dalphaball
                PUtils.stage[3]: ['-no_nstruct_label true',  # '-out:suffix _' + PUtils.stage[2],
                                  '-mute all', '-unmute protocols.rosetta_scripts.ParsedProtocol']}  # -out:pdb false

process_scale = {PUtils.stage[1]: 2, PUtils.stage[2]: 1, PUtils.stage[3]: 1, PUtils.stage[5]: 2, PUtils.nano: 1,
                 PUtils.stage[6]: 1}
