import os
from copy import copy
from json import dumps, loads

from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from PDB import PDB
from PathUtils import program_command, nano
from Query.PDB import user_input_format, input_string, format_string, numbered_format_string, confirmation_string, \
    bool_d, invalid_string
from SequenceProfile import write_fasta, read_fasta_file


def query_user_for_flags(mode='design'):
    flags = \
        {'design':
         {'symmetry': '(str) The symmetry to use for the Design. Won\'t be assigned if missing unless targets are %s'
                      % nano,
          'nanohedra_output': '(bool) Whether the design targets are a %s output' % nano,
          'fragments_exist': '(str) If fragment data has been generated for the design, where is it located?',
          'generate_fragments': '(bool) Whether fragments should be generated fresh for each Pose',
          'design_with_fragments': '(bool) Whether to design with fragment amino acid frequency info',
          'design_with_evolution': '(bool) Whether to design with evolutionary amino acid frequency info',
          'output_assembly': '(bool) If symmetric, whether the expanded assembly should be output. 2- and 3-D materials'
                             ' will be output with a single unit cell.',
          'mask_design_using_sequence': '(bool) Whether design should be masked at certain residues provided in a '
                                        '.fasta file.run \'%s mask -s path/to/your.pdb\' to set this up.'
                                        % program_command,
          'input_location': '(str) Specify a file with a list of input files or a directory where input files are '
                            'located. If the input is a %s output, specifying the master output directory is sufficient'
                            % nano
          },
         'filter':
         {  # TODO

          }}
    write_file = False
    while not write_file:
        print('For a %s run, the following flags can be used to customize the design. You should include these in a '
              'JSON formatted file (run \'%s flags --template\' for a template) or select from the following list to '
              'generate a flags file automatically.\n%s'
              % (mode, program_command, user_input_format % '\n'.join(numbered_format_string % item
                                                                      for idx, item in enumerate(flags[mode].items(),
                                                                                                 1))))
        flags_input = input('Enter the numbers corresponding to the flags your design requires.\nEx: 1 2 4 5 6\n%s'
                            % input_string)
        flag_numbers = flags_input.split()
        chosen_flags = [list(flags[mode].keys())[flag] for flag in map(int, flag_numbers)]
        value_array = [input('For %s what %s value should be used?' % (flag, flags[mode][flag].split()[0]))
                       for idx, flag in enumerate(chosen_flags)]
        output_dict = dict(zip(chosen_flags, value_array))
        while True:
            validation = input('You selected:\n%s\n\n%s' % ('\n'.join(format_string % item
                                                                      for item in output_dict.items()),
                                                            confirmation_string))
            if validation.lower() in bool_d:
                write_file = True
                break
            else:
                print('%s %s is not a valid choice!' % invalid_string, validation)

    with open('%s_flags' % mode, 'w') as f:
        f.write(dumps(output_dict))


def load_flags(file):
    with open(file, 'r') as f:
        flags = loads(f.readlines())

    return flags


def generate_sequence_mask(fasta_file):
    """From a sequence with a corresponding mask, grab the indices that should be masked in the target structural
    calculation"""
    sequence_and_mask = read_fasta_file(fasta_file)
    sequence = sequence_and_mask[0]
    mask = sequence_and_mask[1]
    if not len(sequence) == len(mask):
        exit('The sequence and mask are different lengths! please correct the alignment and lengths before proceeding.')

    return [idx for idx, aa in enumerate(mask, 1) if aa == '-']


def generate_mask_template(pdb_file):
    pdb = PDB.from_file(pdb_file)
    sequence = SeqRecord(Seq(''.join(pdb.atom_sequences.values()), IUPAC.protein), id=pdb.filepath)
    sequence_mask = copy(sequence)
    sequence_mask.id = 'mask'
    sequences = [sequence, sequence_mask]
    fasta_file = write_fasta(sequences, file_name='%s_sequence_for_mask' % os.path.splitext(pdb.filepath)[0])
    print('The mask template was written to %s. Please edit this file so that the mask can be generated for protein '
          'design. Mask should be formatted so a \'-\' replaces all sequence of interest to be overlooked during design'
          '. Example:\n>pdb_template_sequence\nMAGHALKMLV...\n>mask\nMAGH----LV\n' % fasta_file)
