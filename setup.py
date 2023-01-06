from __future__ import annotations

import os
import subprocess

from symdesign import utils
from symdesign.utils import path as putils, input_string

rosetta_url = 'https://www.rosettacommons.org/software/license-and-download'
rosetta_compile_url = 'https://www.rosettacommons.org/docs/latest/build_documentation/Build-Documentation'
rosetta_extras_url = 'https://www.rosettacommons.org/docs/latest/rosetta_basics/running-rosetta-with-options#running-' \
                     'rosetta-with-multiple-threads'
rosetta_variable_dictionary = {0: 'ROSETTA', 1: 'Rosetta', 2: 'rosetta'}
# Todo use search_env_for_variable() to accomplsh this
string_ops = [str.upper, str.lower, str.title]


def search_env_for_variable(search_variable: str) -> str | None:
    """Find shell variables from a list of possible string syntax using an input string

    Args:
        search_variable: The string to search the environment for
    Returns:
        The name of the identified environmental variable
    """
    env_variable = string = None
    search_strings = []
    try:
        string_op_it = iter(string_ops)
        while env_variable is None:
            string = next(string_op_it)(search_variable)
            search_strings.append(string)
            env_variable = os.environ.get(string)
    except StopIteration:
        pass

    if env_variable is None:
        print(f'"{search_variable}" environment inaccessible as no environmental variable was found at any of '
              f'${", $".join(search_strings)}. If you believe there was a mistake, add this enviromental variable to '
              f'the {putils.config_file} file. Ex: '
              '{'
              f'"{search_variable}_env": {search_variable.upper()}, ...'
              '}'
              f' where the value {search_variable.upper()} is the environmental variable (ensure without $)')
    else:
        print(f'Found "{env_variable}" for the environmental variable ${string}')

    return string  # env_variable


if __name__ == '__main__':
    print(f'To properly set up your python environment use the {putils.conda_environment} to initialize your '
          "environment. If you are using anaconda/conda for instance, the command 'conda env create --file "
          f"{putils.conda_environment}' will handle this for you")
    print(f'First, follow this url "{rosetta_url}" to begin licensing and download of the Rosetta Software suite if you'
          " haven't installed already")
    choice1 = input('Once downloaded, type "Y" to continue with install or "S" to skip if Rosetta is already '
                    "installed. FYI this program is capable of using Rosetta's multithreading and MPI builds for "
                    f'faster execution. If you want to learn more, visit {rosetta_extras_url} for details{input_string}')
    while True:
        if choice1.strip() == 'Y':
            print('Next, you will want to move the downloaded tarball to a directory where all the Rosetta software '
                  'will be stored.\nOnce moved, you will want to unzip/extract all files in the tarball.\n'
                  'This can be done with the command tar -zxvf [put_tarball_name_here] in a new shell')
            input('This command may take some time. Once this is finished press Enter.\nIn the meantime, you may be '
                  f'interested in reading about compilation "{rosetta_compile_url}" and different features available '
                  f'with increased computation speed "{rosetta_extras_url}".{input_string}')
            input("Finally, lets compile Rosetta. If you aren't familiar with this process and haven't looked at the "
                  'above links, check them out for assistance. To take full advantage of computation time, think '
                  'carefully about how your computing environment can be set up to work with Rosetta. It is recommended'
                  ' for large design batches to simply use default options for compilation. If you want to have '
                  'individual jobs finish quicker, MPI compatibility may be of interest. Navigate to the MPI resources '
                  'in rosettacommons.org for more information on setting this up.\nPress Enter once you have completed '
                  f'compilation.{input_string}')
            break
        elif choice1.strip() == 'S':
            break
        else:
            choice1 = input(f'"{choice1} is invalid input for [Y/S]. Try again{input_string}')

    print('Great! Attempting to find environmental variable for Rosetta "main" directory')
    rosetta_env_variable = ''
    i = 0
    while True:
        # try:
        rosetta = str(os.environ.get(rosetta_variable_dictionary[i]))
        if rosetta.endswith('/main'):
            print('Automatic detection successful!')
            rosetta_env_variable = rosetta_variable_dictionary[i]
            # break
        else:
            i += 1
            if i >= 3:
                while rosetta_env_variable == '':
                    print('Failed detection of Rosetta environmental variable = %s' % ', '.join(
                        rosetta_variable_dictionary[i] for i in rosetta_variable_dictionary))
                    print('For setup to be successful, the location of your Rosetta install needs to be accessed. It is'
                          ' recommended to modify your shell to include an environmental variable "ROSETTA" (accessed'
                          ' by "$ROSETTA"), leading to the "main" directory of Rosetta.')
                    choice2 = input('If you have one, please enter it below or reply N to set one up.\nInput:')
                    choice2 = choice2.lstrip('$')
                    if choice2.strip() == 'N':
                        print('To make the ROSETTA environmental variable always present at the command line, the '
                              'variable needs to be declared in your ~/.profile file (or analogous shell specific file '
                              'like .bashrc (cshrc, zshrc, or tcshrc if you prefer) To add yourself, append the command'
                              ' below to your ~/.profile file, replacing path/to/rosetta_src_20something.version#/main '
                              'with your actual path.\nexport ROSETTA=path/to/rosetta_src_20something.version#/main\n')
                        input('Once completed, press Enter.\nInput:')
                    else:
                        rosetta = str(os.environ.get(choice2.strip()))

                    if rosetta.endswith('/main'):
                        rosetta_env_variable = choice2
                    else:
                        j = -1
                        while True:
                            j += 1
                            rosetta = str(os.environ.get(rosetta_variable_dictionary[j]))
                            if rosetta.endswith('/main'):
                                rosetta_env_variable = rosetta_variable_dictionary[j]
                                break
                            elif j >= 3:
                                break
            else:
                continue
        if os.path.exists(rosetta):
            print(f'Wonderful, Rosetta environment located and exists. You can now use all the features of '
                  f'{putils.program_name} to interface with Rosetta')
            print(f'All {putils.program_name} files are located in {putils.git_source}')
            break
        else:
            print(f"Rosetta environmental path doesn't exist. Ensure that ${rosetta_env_variable} is correct... "
                  f"Trying again")

    # print('Next, the proper python environment needs to be set up. The following modules need to be available to '
    #       'python otherwise, %s will not run. These inclue:\n-sklearn\n-numpy\n-biopython')
    input("If you don't have these, use a package manager such as pip or conda to install these in your environment.\n"
          f'Press Enter once these are located to continue{input_string}')

    print('Finally, hh-suite needs to be available. This is being created for you in the dependencies directory')
    hhblits_latest_url = 'https://wwwuser.gwdg.de/~compbiol/uniclust/uniclust-latest'
    hhsuite = subprocess.Popen()
    hhsuite.communicate()

    print('Set up is complete! You can now use %s for design of protein interfaces generated using Nanohedra.'
          % putils.program_name)
    print('To design materials, navigate to your desired Nanohedra output directory and run the command %s for details'
          % putils.program_exe)

    # TODO Set up SymDesign.py and ProcessRosettaCommands.sh depending on status of PathUtils
    # Todo ensure that FreeSASA is built. May need to investigate this option
    #  --disable-threads

    config = {'rosetta_env': search_env_for_variable(putils.rosetta_str),
              'hhblits_env': search_env_for_variable(putils.hhblits),
              # TODO
              'rosetta_make': 'mpi'  # 'default', 'python', 'mpi' 'cxx11thread', 'cxx11threadmpi'
              }

    utils.write_json(config, putils.config_file)
    # Set up git submodule
    p0 = subprocess.Popen(['git', 'submodule', 'update', '--init', '--recursive'])
    p0.communicate()
    # Set up freesasa dependency
    os.chdir(putils.freesasa_dir)
    p1 = subprocess.Popen(['autoreconf', '-i'])
    p1.communicate()
    p2 = subprocess.Popen(['./configure', '--disable-xml', '--disable-json'])
    p2.communicate()
    p3 = subprocess.Popen(['make'])
    p3.communicate()
    # Set up stride (or Todo another secondary structure program)
    os.chdir(putils.stride_dir)
    p4 = subprocess.Popen(['tar', '-zxf', 'stride.tar.gz'])
    p4.communicate()
    p5 = subprocess.Popen(['make'])
    p5.communicate()
    # Set up orient dependency
    p6 = subprocess.Popen(['gfortran', '-o', putils.orient_exe_path, f'{putils.orient_exe_path}.f'])
    p6.communicate()
    # Set up errat dependency
    p7 = subprocess.Popen(['g++', '-o', putils.errat_exe_path, putils.errat_residue_source])
    p7.communicate()
