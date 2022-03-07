import os
import math
from glob import glob
from copy import copy
from subprocess import list2cmdline

import numpy as np

import PathUtils as PUtils
import SymDesignUtils as SDUtils
from CommandDistributer import rosetta_flags, script_cmd, distribute
from DesignDirectory import relax_flags
from PDB import PDB
from PathUtils import monofrag_cluster_rep_dirpath, intfrag_cluster_rep_dirpath, intfrag_cluster_info_dirpath, \
    frag_directory
from Query.PDB import boolean_choice
from SequenceProfile import parse_hhblits_pssm, MultipleSequenceAlignment, read_fasta_file  # parse_pssm
from Structure import parse_stride
from SymDesignUtils import DesignError, unpickle, get_all_base_root_paths, start_log, dictionary_lookup
from classes.SymEntry import sdf_lookup
from utils.MysqlPython import Mysql
# import dependencies.bmdca as bmdca


# Globals
# for checking out the options to read and write Rosetta runs to a relational DB such as MySQL
# https://new.rosettacommons.org/docs/latest/rosetta_basics/options/Database-options
logger = start_log(name=__name__)
index_offset = 1


class Database:  # Todo ensure that the single object is completely loaded before multiprocessing... Queues and whatnot
    def __init__(self, oriented, oriented_asu, refined, full_models, stride, sequences, hhblits_profiles, sql=None,
                 log=logger):
        if sql:
            raise DesignError('SQL set up has not been completed!')

        self.log = log
        self.oriented = DataStore(location=oriented, extension='.pdb*', sql=sql, log=log)
        self.oriented_asu = DataStore(location=oriented_asu, extension='.pdb', sql=sql, log=log)
        self.refined = DataStore(location=refined, extension='.pdb', sql=sql, log=log)
        self.full_models = DataStore(location=full_models, extension='_ensemble.pdb', sql=sql, log=log)
        self.stride = DataStore(location=stride, extension='.stride', sql=sql, log=log)
        self.sequences = DataStore(location=sequences, extension='.fasta', sql=sql, log=log)
        self.alignments = DataStore(location=hhblits_profiles, extension='.sto', sql=sql, log=log)
        self.hhblits_profiles = DataStore(location=hhblits_profiles, extension='.hmm', sql=sql, log=log)
        # self.bmdca_fields = \
        #     DataStore(location=hhblits_profiles, extension='_bmDCA%sparameters_h_final.bin' % os.sep, sql=sql, log=log)
        # self.bmdca_couplings = \
        #     DataStore(location=hhblits_profiles, extension='_bmDCA%sparameters_J_final.bin' % os.sep, sql=sql, log=log)

    def load_all_data(self):
        """For every resource, acquire all existing data in memory"""
        #              self.oriented_asu, self.sequences,
        for source in [self.stride, self.alignments, self.hhblits_profiles, self.oriented, self.refined]:  # self.full_models
            try:
                source.get_all_data()
            except ValueError:
                raise ValueError('Issue from source %s' % source)
        # self.log.debug('The data in the Database is: %s'
        #                % '\n'.join(str(store.__dict__) for store in self.__dict__.values()))

    def source(self, name):
        """Return on of the various DataStores supported by the Database"""
        try:
            return getattr(self, name)
        except AttributeError:
            raise AttributeError('There is no Database source named \'%s\' found. Possible sources are: %s'
                                 % (name, ', '.join(self.__dict__)))

    def retrieve_data(self, source=None, name=None):
        """Return the data requested if loaded into source Database, otherwise, load into the Database from the located
        file. Raise an error if source or file/SQL doesn't exist

        Keyword Args:

        Returns:
            (object): The object requested will depend on the source
        """
        object_db = self.source(source)
        data = getattr(object_db, name, None)
        if not data:
            object_db.name = object_db.load_data(name, log=None)
            data = object_db.name  # store the new data as an attribute

        return data

    def retrieve_file(self, from_source=None, name=None):
        """Retrieve the specified file on disk for subsequent parsing"""
        object_db = getattr(self, from_source, None)
        if not object_db:
            raise DesignError('There is no source named %s found in the Design Database' % from_source)

        return object_db.retrieve_file(name)

    def preprocess_entities_for_design(self, all_entities, script_outpath=os.getcwd(), load_resources=False):
        """Assess whether the design files of interest require any processing prior to design calculations.
        Processing includes relaxation into the energy function and/or modelling of missing loops and segments

        Args:
            all_entities (list[Entity]):
        Keyword Args:
            script_outpath=os.getcwd() (str): Where should entity processing commands be written?
            load_resources=False (bool): Whether resources have already been specified to be loaded
        Returns:
            (Tuple[list, bool, bool]): Any instructions, then two booleans on whether designs are pre_refined, and whether
            they are pre_loop_modeled
        """
        global timestamp
        # oriented_asu_files = self.oriented_asu.retrieve_files()
        # oriented_asu_files = os.listdir(orient_asu_dir)
        self.refined.make_path()
        refine_names = self.refined.retrieve_names()
        refine_dir = self.refined.location
        # refine_dir = master_directory.refine_dir
        # refine_files = os.listdir(refine_dir)
        self.full_models.make_path()
        full_model_names = self.full_models.retrieve_names()
        full_model_dir = self.full_models.location
        # full_model_dir = master_directory.full_model_dir
        # full_model_files = os.listdir(full_model_dir)
        entities_to_refine, entities_to_loop_model, sym_def_files = [], [], {}
        # Identify thDe entities to refine and to model loops before proceeding
        for entity in all_entities:  # if entity is here, the file should've been oriented...
            # for entry_entity in entities:  # ex: 1ABC_1
            # symmetry = master_directory.sym_entry.sym_map[idx]
            if entity.symmetry == 'C1':
                sym_def_files[entity.symmetry] = sdf_lookup(None)
            else:
                sym_def_files[entity.symmetry] = sdf_lookup(entity.symmetry)
            # for entry_entity in entities:
            #     entry = entry_entity.split('_')
            # for orient_asu_file in oriented_asu_files:  # iterating this way to forgo missing "missed orient"
            #     base_pdb_code = os.path.splitext(orient_asu_file)[0]
            #     if base_pdb_code in all_entities:
            if entity.name not in refine_names:  # assumes oriented_asu entity name is the same
                entities_to_refine.append(entity)
            if entity.name not in full_model_names:  # assumes oriented_asu entity name is the same
                entities_to_loop_model.append(entity)

        info_messages = []
        # query user and set up commands to perform refinement on missing entities
        pre_refine = True
        if entities_to_refine:  # if files found unrefined, we should proceed
            logger.info('The following oriented oligomers are not yet refined and are being set up for refinement'
                        ' into the Rosetta ScoreFunction for optimized sequence design: %s'
                        % ', '.join(set(os.path.splitext(os.path.basename(orient_asu_file))[0]
                                        for orient_asu_file in entities_to_refine)))
            print('Would you like to refine them now? If you plan on performing sequence design with models '
                  'containing them, it is highly recommended you perform refinement')
            if not boolean_choice():
                print('To confirm, asymmetric units are going to be generated with unrefined coordinates. Confirm '
                      'with \'y\' to ensure this is what you want')
                if boolean_choice():
                    pre_refine = False
            # Generate sbatch refine command
            flags_file = os.path.join(refine_dir, 'refine_flags')
            if not os.path.exists(flags_file):
                flags = copy(rosetta_flags) + relax_flags
                flags.extend(['-out:path:pdb %s' % refine_dir, '-no_scorefile true'])
                flags.remove('-output_only_asymmetric_unit true')  # want full oligomers
                flags.extend(['dist=0'])  # Todo modify if not point groups used
                with open(flags_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(flags))

            # if sym != 'C1':
            refine_cmd = ['@%s' % flags_file, '-parser:protocol',
                          os.path.join(PUtils.rosetta_scripts, '%s.xml' % PUtils.stage[1])]
            # else:
            #     refine_cmd = ['@%s' % flags_file, '-parser:protocol',
            #                   os.path.join(PUtils.rosetta_scripts, '%s.xml' % PUtils.stage[1]),
            #                   '-parser:script_vars']
            refine_cmds = [script_cmd + refine_cmd + ['-in:file:s', entity.filepath, '-parser:script_vars'] +
                           ['sdf=%s' % sym_def_files[entity.symmetry],
                            'symmetry=%s' % 'make_point_group' if entity.symmetry != 'C1' else 'asymmetric']
                           for entity in entities_to_refine]
            commands_file = SDUtils.write_commands([list2cmdline(cmd) for cmd in refine_cmds],
                                                   name='%s-refine_oligomers' % timestamp, out_path=refine_dir)
            refine_sbatch = distribute(file=commands_file, out_path=script_outpath, scale='refine',
                                       log_file=os.path.join(refine_dir, 'refine.log'),
                                       max_jobs=int(len(refine_cmds) / 2 + 0.5),
                                       number_of_commands=len(refine_cmds))
            print('\n' * 2)
            refine_sbatch_message = \
                'Once you are satisfied%snter the following to distribute refine jobs:\n\tsbatch %s' \
                % (', you can run this script at any time. E' if load_resources else ', e', refine_sbatch)
            # logger.info('Please follow the instructions below to refine your input files')
            # logger.critical(sbatch_warning)
            # logger.info(refine_sbatch_message)
            info_messages.append(refine_sbatch_message)
            load_resources = True

        # query user and set up commands to perform loop modelling on missing entities
        pre_loop_model = True
        if entities_to_loop_model:
            logger.info('The following structures have not been modelled for disorder. Missing loops will '
                        'be built for optimized sequence design: %s' % ', '.join(entities_to_loop_model))
            print('Would you like to model loops for these structures now? If you plan on performing sequence '
                  'design with them, it is highly recommended you perform loop modelling to avoid designed clashes')
            if not boolean_choice():
                print('To confirm, asymmetric units are going to be generated without disordered loops. Confirm '
                      'with \'y\' to ensure this is what you want')
                if boolean_choice():
                    pre_loop_model = False
            # Generate sbatch refine command
            flags_file = os.path.join(full_model_dir, 'loop_model_flags')
            if not os.path.exists(flags_file):
                loop_model_flags = ['-remodel::save_top 0', '-run:chain A', '-remodel:num_trajectory 1']
                #                   '-remodel:run_confirmation true', '-remodel:quick_and_dirty',
                flags = copy(rosetta_flags) + loop_model_flags
                # flags.extend(['-out:path:pdb %s' % full_model_dir, '-no_scorefile true'])
                flags.extend(['-no_scorefile true', '-no_nstruct_label true'])
                # flags.remove('-output_only_asymmetric_unit true')  # NOT necessary -> want full oligomers
                with open(flags_file, 'w') as f:
                    f.write('%s\n' % '\n'.join(flags))

            loop_model_cmd = ['@%s' % flags_file, '-parser:protocol',
                              os.path.join(PUtils.rosetta_scripts, 'loop_model_ensemble.xml'),
                              '-parser:script_vars']
            # Make all output paths and files for each loop ensemble
            logger.info('Preparing blueprint and loop files for entity:')
            out_paths, blueprints, loop_files = [], [], []
            for entity in entities_to_loop_model:
                entity_out_path = os.path.join(full_model_dir, entity)
                SDUtils.make_path(entity_out_path)
                out_paths.append(entity_out_path)
                blueprints.append(all_entities[entity].make_blueprint_file(out_path=full_model_dir))
                loop_files.append(all_entities[entity].make_loop_file(out_path=full_model_dir))

            loop_model_cmds = []
            for idx, (entity, sym) in enumerate(entities_to_loop_model.items()):
                entity_cmd = script_cmd + loop_model_cmd + \
                             ['blueprint=%s' % blueprints[idx], 'loop_file=%s' % loop_files[idx],
                              '-in:file:s', os.path.join(refine_dir, '%s.pdb' % entity), '-out:path:pdb',
                              out_paths[idx]] + \
                             (['-symmetry:symmetry_definition', sym_def_files[sym]] if sym != 'C1' else [])

                multimodel_cmd = ['python', PUtils.models_to_multimodel_exe, '-d', out_paths[idx],
                                  '-o', os.path.join(full_model_dir, '%s_ensemble.pdb' % entity)]
                copy_cmd = ['scp', os.path.join(out_paths[idx], '%s_0001.pdb' % entity),
                            os.path.join(full_model_dir, '%s.pdb' % entity)]
                loop_model_cmds.append(
                    SDUtils.write_shell_script(list2cmdline(entity_cmd), name=entity, out_path=full_model_dir,
                                               additional=[list2cmdline(multimodel_cmd), list2cmdline(copy_cmd)]))

            loop_cmds_file = SDUtils.write_commands(loop_model_cmds, name='%s-loop_model_entities' % timestamp,
                                                    out_path=full_model_dir)
            loop_model_sbatch = distribute(file=loop_cmds_file, out_path=script_outpath,
                                           scale='refine', log_file=os.path.join(full_model_dir, 'loop_model.log'),
                                           max_jobs=int(len(loop_model_cmds) / 2 + 0.5),
                                           number_of_commands=len(loop_model_cmds))
            print('\n' * 2)
            loop_model_sbatch_message = \
                'Once you are satisfied%snter the following to distribute loop_modelling jobs:\n\tsbatch %s' \
                % (', run this script AFTER completion of the Entity refinement script. E' if load_resources else ', e',
                   loop_model_sbatch)
            # logger.info('Please follow the instructions below to model loops on your input files')
            # logger.critical(sbatch_warning)
            # logger.info(refine_sbatch_message)
            info_messages.append(loop_model_sbatch_message)
            # load_resources = True

        return info_messages, pre_refine, pre_loop_model


class DataStore:
    def __init__(self, location=None, extension='.txt', sql=None, log=logger):
        self.location = location
        self.extension = extension
        self.sql = sql
        self.log = log

        if '.pdb' in extension:
            self.load_file = PDB.from_file
        elif extension == '.fasta':
            self.load_file = read_fasta_file
        elif extension == '.stride':
            self.load_file = parse_stride
        elif extension == '.hmm':  # in ['.hmm', '.pssm']:
            self.load_file = parse_hhblits_pssm  # parse_pssm
        # elif extension == '.fasta' and msa:  # Todo if msa is in fasta format
        elif extension == '.sto':
            self.load_file = MultipleSequenceAlignment.from_stockholm  # parse_stockholm_to_msa
        elif extension == '_bmDCA%sparameters_h_final.bin' % os.sep:
            self.load_file = bmdca.load_fields
        elif extension == '_bmDCA%sparameters_J_final.bin' % os.sep:
            self.load_file = bmdca.load_couplings
        else:  # '.txt' read the file and return the lines
            self.load_file = self.read_file

    def make_path(self, condition=True):
        """Make all required directories in specified path if it doesn't exist, and optional condition is True

        Keyword Args:
            condition=True (bool): A condition to check before the path production is executed
        """
        if condition:
            os.makedirs(self.location, exist_ok=True)

    def store(self, name):
        """Return the path of the storage location given an entity name"""
        return os.path.join(self.location, '%s%s' % (name, self.extension))

    def retrieve_file(self, name):
        """Returns the actual location by combining the requested name with the stored .location"""
        path = os.path.join(self.location, '%s%s' % (name, self.extension))
        files = glob(path)
        if files:
            if len(files) > 1:
                self.log.error('Found more than one file at %s. Grabbing the first one: %s' % (path, files[0]))

            return files[0]
        else:
            self.log.error('No files found for \'%s\'. Attempting to incorporate into the Database' % path)

    def retrieve_files(self):
        """Returns the actual location of all files in the stored .location"""
        path = os.path.join(self.location, '*%s' % self.extension)
        files = glob(path)
        if files:
            return files
        else:
            self.log.error('No files found for \'%s\'. Attempting to incorporate into the Database' % path)

    def retrieve_names(self):
        """Returns the names of all objects in the stored .location"""
        path = os.path.join(self.location, '*%s' % self.extension)
        names = list(map(os.path.basename, [os.path.splitext(file)[0] for file in glob(path)]))
        if names:
            return names
        else:
            self.log.error('No files found for \'%s\'. Attempting to incorporate into the Database' % path)

    def retrieve_data(self, name=None):
        """Return the data requested if loaded into source Database, otherwise, load into the Database from the located
        file. Raise an error if source or file/SQL doesn't exist

        Keyword Args:
            name=None (str): The name of the data to be retrieved. Will be found with location and extension attributes
        Returns:
            (any[object, None]): If the data is available, the object requested will be returned, else None
        """
        data = getattr(self, name, None)
        if data:
            self.log.debug('Info %s%s was retrieved from DataStore' % (name, self.extension))
        else:
            setattr(self, name, self.load_data(name, log=None))  # attempt to store the new data as an attribute
            data = getattr(self, name)
            if data:
                self.log.debug('Database file %s%s was loaded fresh' % (name, self.extension))

        return data

    def load_data(self, name, **kwargs):
        """Return the data located in a particular entry specified by name

        Returns:
            (Union[None, Any])
        """
        if self.sql:
            dummy = True
        else:
            file = self.retrieve_file(name)
            if file:
                return self.load_file(file, **kwargs)
        return

    def get_all_data(self, **kwargs):
        """Return all data located in the particular DataStore storage location"""
        if self.sql:
            dummy = True
        else:
            for file in glob(os.path.join(self.location, '*%s' % self.extension)):
                # self.log.debug('Fetching %s' % file)
                data = self.load_file(file)
                setattr(self, os.path.splitext(os.path.basename(file))[0], data)

    @staticmethod
    def read_file(file, **kwargs):
        with open(file, 'r') as f:
            lines = f.readlines()

        return lines


class ClusterInfoFile:
    def __init__(self, infofile_path):
        # self.infofile_path = infofile_path
        self.name = os.path.splitext(os.path.basename(infofile_path))[0]
        self.size = None
        self.rmsd = None
        self.representative_filename = None
        self.central_residue_pair_freqs = []
        # self.central_residue_pair_counts = []
        # self.load_info()

    # def load_info(self):
        with open(infofile_path, 'r') as f:
            info_lines = f.readlines()

        is_res_freq_line = False
        for line in info_lines:
            # if line.startswith("CLUSTER NAME:"):
            #     self.name = line.split()[2]
            if line.startswith("CLUSTER SIZE:"):
                self.size = int(line.split()[2])
            elif line.startswith("CLUSTER RMSD:"):
                self.rmsd = float(line.split()[2])
            elif line.startswith("CLUSTER REPRESENTATIVE NAME:"):
                self.representative_filename = line.split()[3]
            elif line.startswith("CENTRAL RESIDUE PAIR COUNT:"):
                is_res_freq_line = False
            elif is_res_freq_line:
                res_pair_type = (line.split()[0][0], line.split()[0][1])
                res_pair_freq = float(line.split()[1])
                self.central_residue_pair_freqs.append((res_pair_type, res_pair_freq))
            elif line.startswith("CENTRAL RESIDUE PAIR FREQUENCY:"):
                is_res_freq_line = True

    def get_name(self):
        return self.name

    def get_size(self):
        return self.size

    def get_rmsd(self):
        return self.rmsd

    def get_representative_filename(self):
        return self.representative_filename

    def get_central_residue_pair_freqs(self):
        return self.central_residue_pair_freqs


class FragmentDB:
    def __init__(self, fragment_length=5):
        self.monofrag_representatives_path = monofrag_cluster_rep_dirpath
        self.cluster_representatives_path = intfrag_cluster_rep_dirpath
        self.cluster_info_path = intfrag_cluster_info_dirpath
        self.fragment_length = fragment_length
        self.reps = None
        self.paired_frags = None
        self.indexed_ghosts = {}
        self.info = None

    def get_monofrag_cluster_rep_dict(self):
        self.reps = {int(os.path.splitext(file)[0]):
                     PDB.from_file(os.path.join(root, file), solve_discrepancy=False, pose_format=False,
                                   entities=False, log=None)
                     for root, dirs, files in os.walk(self.monofrag_representatives_path) for file in files}

    def get_intfrag_cluster_rep_dict(self):
        ijk_cluster_representatives = {}
        for root, dirs, files in os.walk(self.cluster_representatives_path):
            if not dirs:
                i_cluster_type, j_cluster_type, k_cluster_type = map(int, root.split(os.sep)[-1].split('_'))

                if i_cluster_type not in ijk_cluster_representatives:
                    ijk_cluster_representatives[i_cluster_type] = {}
                if j_cluster_type not in ijk_cluster_representatives[i_cluster_type]:
                    ijk_cluster_representatives[i_cluster_type][j_cluster_type] = {}

                for file in files:
                    ijk_frag_cluster_rep_pdb = PDB.from_file(os.path.join(root, file), solve_discrepancy=False,
                                                             pose_format=False, entities=False, log=None)
                    # ijk_cluster_rep_mapped_chain = file[file.find('mappedchain') + 12:file.find('mappedchain') + 13]
                    ijk_cluster_rep_partner_chain = file[file.find('partnerchain') + 13:file.find('partnerchain') + 14]
                    ijk_cluster_representatives[i_cluster_type][j_cluster_type][k_cluster_type] = \
                        (ijk_frag_cluster_rep_pdb, ijk_cluster_rep_partner_chain)  # ijk_cluster_rep_mapped_chain,

        self.paired_frags = ijk_cluster_representatives
        if self.info:
            self.index_ghosts()

    def get_intfrag_cluster_info_dict(self):
        intfrag_cluster_info_dict = {}
        for root, dirs, files in os.walk(self.cluster_info_path):
            if not dirs:
                i_cluster_type, j_cluster_type, k_cluster_type = map(int, root.split(os.sep)[-1].split('_'))

                if i_cluster_type not in intfrag_cluster_info_dict:
                    intfrag_cluster_info_dict[i_cluster_type] = {}
                if j_cluster_type not in intfrag_cluster_info_dict[i_cluster_type]:
                    intfrag_cluster_info_dict[i_cluster_type][j_cluster_type] = {}

                for file in files:
                    intfrag_cluster_info_dict[i_cluster_type][j_cluster_type][k_cluster_type] = \
                        ClusterInfoFile(os.path.join(root, file))

        self.info = intfrag_cluster_info_dict
        if self.paired_frags:
            self.index_ghosts()

    def index_ghosts(self):
        """From the fragment database, precompute all required data into arrays to populate Ghost Fragments"""
        for i_type in self.paired_frags:
            stacked_bb_coords = np.array([frag_pdb.chain(frag_paired_chain).get_backbone_coords()
                                          for j_dict in self.paired_frags[i_type].values()
                                          for frag_pdb, frag_paired_chain in j_dict.values()])
            stacked_guide_coords = np.array([frag_pdb.chain('9').coords for j_dict in self.paired_frags[i_type].values()
                                             for frag_pdb, _, in j_dict.values()])
            ijk_types = \
                np.array([(i_type, j_type, k_type) for j_type, j_dict in self.paired_frags[i_type].items()
                          for k_type in j_dict])
            # rmsd_array = np.array([self.info.cluster(type_set).rmsd for type_set in ijk_types])  # Todo
            rmsd_array = np.array([dictionary_lookup(self.info, type_set).rmsd for type_set in ijk_types])
            self.indexed_ghosts[i_type] = (stacked_bb_coords, stacked_guide_coords, ijk_types, rmsd_array)


class FragmentDatabase(FragmentDB):
    def __init__(self, source='biological_interfaces', fragment_length=5, init_db=False, sql=False):
        super().__init__()  # FragmentDB
        # self.monofrag_representatives_path = monofrag_representatives_path
        # self.cluster_representatives_path
        # self.cluster_info_path = cluster_info_path
        # self.reps = None
        # self.paired_frags = None
        # self.info = None
        self.source = source
        # Todo load all statistics files into the pickle!
        # self.location = frag_directory.get(self.source, None)
        self.statistics = {}
        # {cluster_id: [[mapped, paired, {max_weight_counts}, ...], ..., frequencies: {'A': 0.11, ...}}
        #  ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749]
        self.fragment_range = None
        self.cluster_info = {}
        self.fragdb = None

        if sql:
            self.start_mysql_connection()
            self.db = True
        else:  # self.source == 'directory':
            # Todo initialize as local directory
            self.db = False
            if init_db:
                logger.info('Initializing %s FragmentDatabase from disk. This may take awhile...' % source)
                self.get_monofrag_cluster_rep_dict()
                self.get_intfrag_cluster_rep_dict()
                self.get_intfrag_cluster_info_dict()
                # self.get_cluster_info()

        self.get_db_statistics()
        self.parameterize_frag_length(fragment_length)

    @property
    def location(self):
        """Provide the location where fragments are stored"""
        return frag_directory.get(self.source, None)

    def get_db_statistics(self):
        """Retrieve summary statistics for a specific fragment database located on directory

        Returns:
            (dict): {cluster_id1: [[mapped_index_average, paired_index_average, {max_weight_counts_mapped}, {paired}],
                                   total_fragment_observations],
                     cluster_id2: ...,
                     frequencies: {'A': 0.11, ...}}
                ex: {'1_0_0': [[0.540, 0.486, {-2: 67, -1: 326, ...}, {-2: 166, ...}], 2749], ...}
        """
        if self.db:
            logger.warning('No SQL DB connected yet!')  # Todo
            raise DesignError('Can\'t connect to MySQL database yet')
        else:
            stats_file = glob(os.path.join(self.location, 'statistics.pkl'))
            if len(stats_file) == 1:
                self.statistics = unpickle(stats_file[0])
            else:
                raise DesignError('There were too many statistics.pkl files found from the fragment database source!')
            # for file in os.listdir(self.location):
            #     if 'statistics.pkl' in file:
            #         self.statistics = unpickle(os.path.join(self.location, file))
            #         return

    def get_db_aa_frequencies(self):
        """Retrieve database specific interface background AA frequencies

        Returns:
            (dict): {'A': 0.11, 'C': 0.03, 'D': 0.53, ...}
        """
        return self.statistics.get('frequencies', {})

    def retrieve_cluster_info(self, cluster=None, source=None, index=None):  # Todo rework this, below func for Database
        """Return information from the fragment information database by cluster_id, information source, and source index
         cluster_info takes the form:
            {'1_2_123': {'size': ..., 'rmsd': ..., 'rep': ...,
                         'mapped': indexed_frequency_dict, 'paired': indexed_frequency_dict}
                         indexed_frequency_dict = {-2: {'A': 0.1, 'C': 0., ..., 'info': (12, 0.41)},
                                                   -1: {}, 0: {}, 1: {}, 2: {}}

        Keyword Args:
            cluster=None (str): A cluster_id to get information about
            source=None (str): The source of information to gather from: ['size', 'rmsd', 'rep', 'mapped', 'paired']
            index=None (int): The index to gather information from. Must be from 'mapped' or 'paired'
        Returns:
            (dict)
        """
        if cluster:
            if cluster not in self.cluster_info:
                self.get_cluster_info(ids=[cluster])
            if source:
                if index is not None and source in ['mapped', 'paired']:  # must check for not None. The index can be 0
                    return self.cluster_info[cluster][source][index]
                else:
                    return self.cluster_info[cluster][source]
            else:
                return self.cluster_info[cluster]
        else:
            return self.cluster_info

    def get_cluster_info(self, ids=None):
        """Load cluster information from the fragment database source into attribute cluster_info
        # todo change ids to a tuple
        Keyword Args:
            id_list=None: [1_2_123, ...]
        Sets:
            self.cluster_info (dict): {'1_2_123': {'size': , 'rmsd': , 'rep': , 'mapped': , 'paired': }, ...}
        """
        if self.db:
            logger.warning('No SQL DB connected yet!')  # Todo
            raise DesignError('Can\'t connect to MySQL database yet')
        else:
            if not ids:
                directories = get_all_base_root_paths(self.location)
            else:
                directories = []
                for _id in ids:
                    c_id = _id.split('_')
                    _dir = os.path.join(self.location, c_id[0], '%s_%s' % (c_id[0], c_id[1]),
                                        '%s_%s_%s' % (c_id[0], c_id[1], c_id[2]))
                    directories.append(_dir)

            for cluster_directory in directories:
                cluster_id = os.path.basename(cluster_directory)
                self.cluster_info[cluster_id] = unpickle(os.path.join(cluster_directory, '%s.pkl' % cluster_id))

    @staticmethod
    def get_cluster_id(cluster_id, index=3):
        """Returns the cluster identification string according the specified index

        Args:
            cluster_id (str): The id of the fragment cluster. Ex: 1_2_123
        Keyword Args:
            index_number=3 (int): The index on which to return. Ex: index_number=2 gives 1_2
        Returns:
            (str): The cluster_id modified by the requested index_number
        """
        while len(cluster_id) < 3:
            cluster_id += '0'

        if len(cluster_id.split('_')) != 3:  # in case of 12123?
            id_l = [cluster_id[:1], cluster_id[1:2], cluster_id[2:]]
        else:
            id_l = cluster_id.split('_')

        info = [id_l[i] for i in range(index)]

        while len(info) < 3:  # ensure the returned string has at least 3 indices
            info.append('0')

        return '_'.join(info)

    def parameterize_frag_length(self, length):
        """Generate fragment length range parameters for use in fragment functions"""
        _range = math.floor(length / 2)  # get the number of residues extending to each side
        if length % 2 == 1:
            self.fragment_range = (0 - _range, 0 + _range + index_offset)
            # return 0 - _range, 0 + _range + index_offset
        else:
            self.log.critical('%d is an even integer which is not symmetric about a single residue. '
                              'Ensure this is what you want' % length)
            self.fragment_range = (0 - _range, 0 + _range)

    def start_mysql_connection(self):
        self.fragdb = Mysql(host='cassini-mysql', database='kmeader', user='kmeader', password='km3@d3r')
