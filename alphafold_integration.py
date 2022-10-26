# Setting up SymDesign for Alphafold
# All one hot encoding for Alphafold use 3 letter alphabetical order. index 20 is 'X', 21 is '-'
# Monomer features
features = {
    'aatype',
    # np.array with shape (seq_length, 21), dtype=np.int32
    # where each position contains the one-hot encoded sequence with an unknown
    # The multimer model performs the one-hot operation itself so input to the model should be
    # np.array with shape (seq_length), dtype=np.int32
    'between_segment_residues',
    # np.array with shape (seq_length), dtype=np.int32
    # ??? not sure what this is for yet
    'domain_name',
    # np.array with shape (number_of_chains), dtype=object
    # domain information from the fasta file after the >. EX from the PDB:
    # 4GRD_1|Chains A, B, C, D|Phosphoribosylaminoimidazole carboxylase catalytic subunit|Burkholderia cenocepacia (216591)
    'residue_index',
    # np.array with shape (seq_length), dtype=np.int32 where all positions increment from 0 to seq_length-1
    # HOW DOES THIS LOOK FOR MULTIMER?
    'seq_length',
    # int -> np.array with shape (seq_length), dtype=np.int32 where all positions are the seq_length
    'sequence',
    # np.array with shape (seq_length), dtype=np.object (bytes) where all positions are the sequence id
    'deletion_matrix',
    # list[list[int]] -> np.array with shape (num_alignments, seq_length), dtype=np.float32
    # The element at `deletion_matrix[i][j]` is the number of residues deleted from
    # the aligned sequence i at residue position j
    'deletion_matrix_int',
    # list[list[int]] -> np.array with shape (num_alignments, seq_length), dtype=np.int32
    # The element at `deletion_matrix[i][j]` is the number of residues deleted from
    # the aligned sequence i at residue position j
    'msa',
    # np.array with shape (num_alignments, seq_length), dtype=np.int32 where positions are the one-hot encoded sequence
    # These are possibly up to index 22 for 'X' and '-' and missing_msa_token (only for training)
    'msa_species_identifiers',
    # np.array with shape (num_alignments), dtype=np.object (bytes) where each index is the species_id for that sequence
    'num_alignments',
    # int -> np.array with shape (seq_length), dtype=np.int32 where all positions are the num_alignments
    'template_aatype',
    # one hot encoded np.array with shape (num_templates, seq_length, 22) 22 is the ?one-hot plus unknown or gaps?
    'template_all_atom_masks',
    # np.array with shape (num_templates, seq_length, 37), dtype=np.int32
    # where each axis=-1 are the atom mask for each atom residue
    'template_all_atom_positions',
    # np.array with shape (num_templates, seq_length, 37, 3), dtype=np.float32
    # where each axis=-1 are the corrdinates with padding for each atom
    'template_domain_names',
    # np.array with shape (num_templates), dtype=np.object (bytes) where each index is the PDB_code and asymID (chainID)
    'template_e_value',
    # np.array with shape (num_templates), dtype=np.float32 where each index is alignment e_value
    'template_neff',
    # np.array with shape (num_templates), dtype=np.float32 where each index is alignment neff
    'template_prob_true',
    # np.array with shape (num_templates, 1), dtype=np.float32 where each value is the probability of being true
    'template_release_date',
    # np.array with shape (num_templates), dtype=np.object (bytes) where each index is the release date
    'template_score',
    # np.array with shape (num_templates, 1), dtype=np.float32 where each value is the score
    'template_similarity',
    # np.array with shape (num_templates, 1), dtype=np.float32 where each value is the frequency of similarity (hamming distance)
    'template_sequence',
    # np.array with shape (num_templates), dtype=np.object (bytes) where each is the aligned template sequence
    'template_sum_probs',
    # np.array with shape (num_templates, 1), dtype=np.float32 where each value is the score
    'template_confidence_scores'
    # np.array with shape (num_templates, seq_length), dtype=np.int64 with each being the per residue score
    }

# ----- Multimer ------
# For each run of alphafold, we must supply the number of desired chains of that multimer as unique sequences to make
# chain_features. The chain features will be generated for each as described above and put into a larger dictionary
# with the additional information provided below
# {'chainID1': features1, ...
#  as above... now becomes
#  'chainID1': multimer_features1, ...
# msa sequences are paired (seems like a tedious process) if a heteromer is provided,
# otherwise, the single sequence is used

# All these features below stem from the requirements found at
# alphafold.alphafold.data.feature_processing.REQUIRED_FEATURES
multimer_features = {
    'msa_mask',  # also made in the monomer version
    'deletion_matrix',
    # '_all_seq' features are only used if the model is a heteromer
    'msa_all_seq',
    'msa_mask_all_seq',
    'deletion_matrix_all_seq',
    'deletion_matrix_int_all_seq',
    'num_alignments_all_seq',
    'msa_species_identifiers_all_seq',
    # END '_all_seq'
    'assembly_num_chains',
    # np.array with shape (number_of_chains) the number of chains in the model
    'all_atom_masks',
    # np.array with shape (num_templates, seq_length, 37), dtype=np.int32
    # where each axis=-1 are the atom mask for each atom residue
    'all_atom_positions',
    # np.array with shape (num_templates, seq_length, 37, 3), dtype=np.float32
    # where each axis=-1 are the corrdinates with padding for each atom
    # where each features['deletion_matrix_int'] is converted
    'deletion_matrix',
    # converted from 'deletion_matrix_int' to float32 for multimer
    'deletion_mean',
    # take the mean over each sequence in multiple sequence alignment
    'template_all_atom_mask',  # <- notice the change from *_masks to *_mask
    # np.array with shape (num_templates, seq_length, 37), dtype=np.int32
    # where each axis=-1 are the atom mask for each atom residue
    'entity_mask',
    # np.array with shape (seq_length), dtype=np.int32 where the default value is 1
    'auth_chain_id',
    # np.array with shape (1,), dtype=np.object_
    'aatype',
    # The multimer model performs the one-hot operation itself so input to the model should be
    # np.array with shape (seq_length), dtype=np.int32

    # These are added by the function alphafold.alphafold.data.pipeline_multimer.add_assembly_features()
    # and are really easy to add by myself
    'asym_id',
    # np.array with shape (seq_length), dtype=np.int32
    # one array for each chain up to number_of_chains
    # each successive chain will increase the integer value by 1
    'sym_id',
    # np.array with shape (seq_length), dtype=np.int32
    # one array for each chain up to number_of_chains
    # each successive symmetric copy will increase the integer value by 1 to equal the multimeric number
    'entity_id',
    # np.array with shape (seq_length), dtype=np.int32
    # one array for each chain up to number_of_chains
    # each successive entity will increase the integer value by 1 to equal the entity number
    }


# So this constitutes the actions needed to make the "multimer" features
"""
import alphafold

all_chain_features = {}
available_chain_ids = structure.utils.chain_id_generator()  # structure.model
for entity in pose.entities:
    entity_features = entity.process_alphafold_features()
    # The above function creates most of the work for the adaptation
    # particular importance needs to be given to the MSA used.
    # Should fragments be utilized in the MSA? If so, naming them in some way to pair is required!
    # Follow the example in alphafold.alphafold.data.pipeline.make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
    # to featurize
    for sym_id in range(1, pose.number_of_symmetry_mates):
        chain_id = next(available_chain_ids)  # The mmCIF formatted chainID with 'AB' type notation
        all_chain_features[chain_id] = copy.deepcopy(entity_features)

# Perform myself
# all_chain_features = alphafold.alphafold.data.pipeline_multimer.add_assembly_features(all_chain_features)

np_example = alphafold.alphafold.data.feature_processing.pair_and_merge(
    all_chain_features=all_chain_features)

# Pad MSA to avoid zero-sized extra_msa.
np_example = pad_msa(np_example, 512)

return np_example
"""