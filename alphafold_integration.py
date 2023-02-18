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
    # HOW DOES THIS LOOK FOR MULTIMER? It seems to be repeating for each additional chain
    'seq_length',
    # int -> np.array with shape (seq_length), dtype=np.int32 where all positions are the seq_length
    # Gets converted in multimer to a single value with shape->(), size=1
    'sequence',
    # np.array with shape (seq_length), dtype=np.object (bytes) where all positions are the aa type sequence
    # In multimer converted to np.array with shape->(), size=1 (seq_length), dtype=np.object (bytes)
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
# Would uniclust30 work for this? Only if the sequences come with species information that can then be paired

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
    # converted from 'deletion_matrix_int' to float32 for multimer/monomer
    'deletion_mean',
    # take the mean over each sequence in multiple sequence alignment
    'template_all_atom_mask',  # <- notice the change from *_masks to *_mask
    # np.array with shape (num_templates, seq_length, 37), dtype=np.int32
    # where each axis=-1 are the atom mask for each atom residue
    'entity_mask',
    # np.array with shape (seq_length), dtype=np.int32 where the default value is 1, which I believe represents a
    # position to be predicted, i.e 'entity_id' > 0
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

# So this constitutes the actions needed to make the template features
"""
    TEMPLATE_FEATURES = {
        'template_aatype': np.float32,
        'template_all_atom_masks': np.float32,
        'template_all_atom_positions': np.float32,
        'template_domain_names': object,
        'template_sequence': object,
        'template_sum_probs': np.float32,
    }
    template_features = {key: [] for key in TEMPLATE_FEATURES}
    
    # Todo get these into symdesign *****
    for hit in template_hits:
        # Where hit is parsed from hmmsearch or hhsearch
        ""
        hit = TemplateHit(
            index=i,  # The number of the hit in a sequence of multiple hits
            name=f'{pdb_id}_{chain}',
            aligned_cols=aligned_cols, (int)  # This represents the number of matches
            sum_probs=None,
            query=query_sequence,
            hit_sequence=hit_sequence.upper(),
            # This is the sequence to be predicted
            indices_query=indices_query,  # <- [0,1,2,-1,-1,...] Where -1 is a gap '-' and every aligned index is 
            #                                  incremented. Seems similar to my Entity.alignment function
            indices_hit=indices_hit,  # Same data type as above
        )
        ""
        result = featurize(feature_hit)
        # performs the work of aligning a PDB from mmcif to the query sequence. inputs a black all_atom_position/mask 
        if sequence doesn't align
        features = {
            'template_all_atom_positions': np.array(templates_all_atom_positions),
            'template_all_atom_masks': np.array(templates_all_atom_masks),  <- contains 1 where atom is present
            'template_sequence': output_templates_sequence.encode(),
            'template_aatype': np.array(templates_aatype),  <- one hot encoded
            'template_domain_names': f'{pdb_id.lower()}_{chain_id}'.encode()
        }
        # Todo get these into symdesign ******
        
        # Add each result to the concatenated dictionary block
        for key, feature in template_features.items():
            feature.append(result[key)
    # Process concatenated block to a stacked np array with the correct dtype
    for name in template_features:
      if num_hits > 0:
        template_features[name] = np.stack(
            template_features[name], axis=0).astype(TEMPLATE_FEATURES[name])
      else:
        # Make sure the feature has correct dtype even if empty.
        template_features[name] = np.array([], dtype=TEMPLATE_FEATURES[name])

    return template_features
    # return TemplateSearchResult(
    #     features=template_features,  # DONT CARE ABOUT THESE -> errors=errors, warnings=warnings)
"""
# So this constitutes the actions needed to make the "multimer" sequence/msa features
"""
Setting up the MSA for use in alphafold multimer
Multimer makes explicit use of the multiple sequence alignment species identifier present from any MSA constructed 
during data initialization
Where the species identifier from each of the multimeric sequences are paired so that the resulting msa has 
sequences only resulting from the same organisms. I.e. homologous msa sequences which are pulled from the some homology
This would allow multimer to make predictions based on relevant evolutionary coupling information for the domains 
in question. I think this will fail to materialize upon use of the msa as a prediction of a designed complex.
The use of multiple sequence alignments for the design would need to be carried out for the design in question to 
gather any relevant information regarding a new interface installed.  
 
An Msa object, as used in alphafold is a dataclass with three attributes. 
.sequences, .descriptions, and .deletion_matrix

# Sequences coming from UniProtKB database come in the
# `db|UniqueIdentifier|EntryName` format, e.g. `tr|A0A146SKV9|A0A146SKV9_FUNHE`
# or `sp|P0C2L1|A3X1_LOXLA` (for TREMBL/Swiss-Prot respectively).
where the UniqueIdentifier is the AccessionIdentifier
where the EntryName is the species_id
identifiers = msa_identifiers.get_identifiers(msa.descriptions[sequence_index])
species_ids.append(identifiers.species_id.encode('utf-8'))

# Go without, or go with a custom?
# Todo Ideally I could construct a msa that uses the interface fragment observed sequences to construct a
# msa type object that has the occurrences of the fragments. 
To run without use of msa I need to adjust the following parameters

cfg = config.model_config('model_3_ptm')  # <- replace with model_3_multimer_v3
# NONE OF THESE PARAMETERS ARE AVAILABLE WITH MULTIMER. HAVE TO CONSTRUCT A "PSEUDO MSA"
cfg.model.num_recycle = 0  # Set because the num_recycle are controlled through .ipynb
cfg.data.common.num_recycle = 0  # Set because the num_recycle are controlled through .ipynb
cfg.data.eval.max_msa_clusters = 1
cfg.data.common.max_extra_msa = 1
cfg.data.eval.masked_msa_replace_fraction = 0
cfg.model.global_config.subbatch_size = None

SOKRYPTON af_backprop USES:
make_msa_features(msas=[[seq]], deletion_matrices=[[[0]*length]])
# IMPORTANT
for running without an msa, like sokrypton, I can just provide an msa that is one sequence deep (the query)

"""