import argparse
import os

from symdesign.structure.model import Model
from symdesign import flags, utils

# globals
logger = utils.start_log(name=os.path.basename(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concatenate a group of structure files. The name of the resulting '
                                                 'file will be the individual structures concatenated in the same order'
                                                 ' as provided. For a directory, this is alphabetically')
    input_mutex = parser.add_mutually_exclusive_group(required=True)
    input_mutex.add_argument(*flags.file_args, **flags.file_kwargs)
    input_mutex.add_argument(*flags.directory_args, **flags.directory_kwargs)
    parser.add_argument(*flags.output_directory_args, **flags.output_directory_kwargs, required=True)
    args, additional_args = parser.parse_known_args()

    # input files
    if args.directory:
        args.files = utils.get_directory_file_paths(args.directory, extension='.pdb')
    # initialize PDB Objects
    models = [Model.from_file(file, log=logger) for file in args.files]
    model = Model.from_chains([chain for model in models for chain in model.chains],
                              name='-'.join(model.name for model in models),
                              log=logger, entities=False)
    # Output file
    os.makedirs(args.output_directory, exist_ok=True)
    model.write(out_path=os.path.join(args.output_directory, f'{model.name}.pdb'))
