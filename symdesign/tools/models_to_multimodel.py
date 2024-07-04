from os import path
from argparse import ArgumentParser
from pathlib import Path


if __name__ == '__main__':
    parser = ArgumentParser(description='Create a list of files the belong in a directory')
    parser.add_argument('-d', '--directory', type=path.abspath, help='Directory where models are located',
                        required=True)
    parser.add_argument('-o', '--out_path', type=path.abspath,
                        help='File where multimodel should be written', required=True)
    parser.add_argument('-s', '--suffix', type=str, help='A suffix in the files to search for', default='')
    parser.add_argument('-e', '--extension', type=str, help='An extension in the files to search for',
                        default='')

    args, additional_args = parser.parse_known_args()
    # ---------------------------------------------------
    sorted_models = sorted(Path(args.directory).glob(f'*{args.suffix}.{args.extension}'), key=lambda p: p.stem)
    with open(args.out_path, 'w') as f:
        for model_number, model in enumerate(sorted_models, 1):
            f.write('{:9s}{:>4d}\n'.format('MODEL', model_number))
            with open(model, 'r') as f_model:
                f.write('%s\n' % ''.join(line for line in f_model.readlines() if line[:4] == 'ATOM')
                        .lstrip('\n').rstrip('\n'))
            f.write('ENDMDL\n')
