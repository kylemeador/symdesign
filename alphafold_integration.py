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
    # where each axis=-1 are the coordinates with padding for each atom
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
# The uniprot_90 is required when running multimer as I presume the uniprot references are used to pair.
# Would uniclust30 work for this?

# All these features below stem from the requirements found at
# alphafold.alphafold.data.feature_processing.REQUIRED_FEATURES
multimer_features = {
    'msa_mask',  # also made in the monomer version
    'deletion_matrix',
    # START '_all_seq' features are only used if the model is a heteromer
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
    # where each axis=-1 are the coordinates with padding for each atom
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
    # and are really easy to add by myself. See below...
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

def Entity.alphafold_process() -> pipeline.FeatureDict:
    # IS THIS NECESSARY. DON"T THINK SO IF I HAVE MSA
    chain_features = alphafold.alphafold.data.pipeline.DataPipeline.process(
        input_fasta_path=
        msa_output_dir=
    )
        # This runs
        sequence_features = make_sequence_features(sequence=input_sequence,
                                                   description=input_description,
                                                   num_res=num_res)
            features = {
                'aa_type': MAKE ONE HOT with X i.e. unknown are X
                'between_segment_residues': np.zeros((seq_length,), dtype=np.int32)
                'domain_name': np.array([description.encode('utf-8')], dtype=np.object_)
                'residue_index': np.arange(seq_length, dtype=np.int32)
                'seq_length': np.full(seq_length, seq_length, dtype=np.int32)
                'sequence': np.array([sequence.encode('utf-8')], dtype=np.object_)
            }
        msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa)) <- I can use a single one...
            features = {
                'deletion_matrix_int': GET THIS FROM THE MATRIX PROBABLY USING CODE IN COLLAPSE PROFILE cumcount...
                'msa': sequences_to_numeric(HHBLITS_AA_TO_ID, dtype=np.int32)
                'num_alignments': np.full(num_residues, num_alignments, dtype=np.int32)  # Fill by the number of 
                # residues how many sequences are in the MSA
                'msa_species_identifiers': np.array([id.encode('utf-8') for id in species_ids], dtype=np.object_)
            }
        return {**msa_features,
                **sequence_features,
                **template_result.features} <- for templates. Can use NULL # Todo grab from .ipynb
                    notebook_utils.empty_placeholder_template_features(num_templates=0, num_res=len(sequence))

    if heteromer:
        # Unsure about the order of this next part
        msa = ?? This is likely a MultipleSeqAlignment type object that gets parsed from .sto file
        all_seq_features = pipeline.make_msa_features([msa])
        valid_feats = msa_pairing.MSA_FEATURES + (
            'msa_species_identifiers',
        )
        feats = {f'{k}_all_seq': v for k, v in all_seq_features.items()
                 if k in valid_feats}
        chain_features.update(feats)
    
    return feats

def Pose.alphafold_process_features():
    all_chain_features = {}
    available_chain_ids = structure.utils.chain_id_generator()  # structure.model
    for entity in pose.entities:
        # Todo make process_alphafold_features like the below function
        #  entity_features = pipeline.make_sequence_features(sequence=sequence, description='query', num_res=len(sequence)))
        #  entity_features = pipeline_multimer.DataPipeline._process_single_chain(sequence=sequence, description='query', 
        #                                                                         num_res=len(sequence)))
        entity_features: dict[str, Any] = entity.process_alphafold_features()
        # The above function creates most of the work for the adaptation
        # particular importance needs to be given to the MSA used.
        # Should fragments be utilized in the MSA? If so, naming them in some way to pair is required!
        # Follow the example in:
        #    alphafold.alphafold.data.pipeline.make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
        # to featurize
        if multimer:
            # The chain_id passed to this function is used by the entity_features to (maybe) tie different chains to this chain
            entity_features = convert_monomer_features(entity_features, chain_id=chain_id)
            # The chain_id passed should be oriented so that each additional entity has the chain_id of the 
            chain_id = available_chain_ids[number_of_symmetry_mates * zero indexed entity_idx]
            # Therefor for an A4B4 C4, the chain_id for entity idx 0 would be A and for entity idx 1 would be E
            # This may not be important, but this is how multimer gets run  
        for sym_id in range(1, pose.number_of_symmetry_mates):
            chain_id = next(available_chain_ids)  # The mmCIF formatted chainID with 'AB' type notation
            all_chain_features[chain_id] = copy.deepcopy(entity_features)
            # NOT HERE chain_features = convert_monomer_features(copy.deepcopy(entity_features), chain_id=chain_id)
            # all_chain_features[chain_id] = chain_features
    
    # Perform v myself
    # all_chain_features = alphafold.alphafold.data.pipeline_multimer.add_assembly_features(all_chain_features)
    # BY
    # Make a dictionary with for `<seq_id>_<sym_id>` where seq_id is the chain name assigned to the Entity where chain 
    # names increment according to excel i.e. A,B,...AA,AB,... and sym_id increments from 1 to number_of_symmetry_mates
    # Where chain_features comes from each entry in the all_chain_features dictionary
    # Where chain_id increments for each new chain instance i.e. A_1 is 1, A_2 is 2, ...
    # Where entity_id increments for each new Entity instance i.e. A_1 is 1, A_2 is 1, ...
    # Where sym_id increments for each new Entity instance i.e. A_1 is 1, A_2 is 2, ..., B_1 is 1, B2 is 2
    # {'A_1': chain_features.update({'asym_id': chain_id * np.ones(sequence_length), 
                                     'sym_id': sym_id * np.ones(sequence_length),
                                     'entity_id': entity_id * np.ones(sequence_length)})
    
    np_example = alphafold.alphafold.data.feature_processing.pair_and_merge(
        all_chain_features=all_chain_features)
    
    # Pad MSA to avoid zero-sized extra_msa.
    np_example = alphafold.data.pipeline_multimer.pad_msa(np_example, 512)
    
    return np_example

# # Turn my input into a feature dict generation. Rely on alphafold.run_alphafold.predict_structure()
# # Is this a good idea?
# DataPipeline:
#     def process(self, input_fasta_path=fasta_path, msa_output_dir=msa_output_dir) -> FeatureDict:
#         
#         return features
"""