import os
import subprocess
import PathUtils as PUtils


rosetta_url = 'https://www.rosettacommons.org/software/license-and-download'
rosetta_compile_url = 'https://www.rosettacommons.org/docs/latest/build_documentation/Build-Documentation'
rosetta_extras_url = 'https://www.rosettacommons.org/docs/latest/rosetta_basics/running-rosetta-with-options#running-' \
                     'rosetta-with-multiple-threads'
rosetta_variable_dictionary = {0: 'ROSETTA', 1: 'Rosetta', 2: 'rosetta'}


def main():
    print('First, follow this url \'%s\' to begin licensing and download of the Rosetta Software suite if you have not '
          'installed already.' % rosetta_url)
    choice1 = input('Once downloaded, type \'Y\' to continue with install or \'S\' to skip if Rosetta is already '
                    'installed. FYI this program is capable of using Rosetta\'s multithreading and MPI builds for '
                    'faster execution. If you want to learn more, visit %s for details. \nInput:' % rosetta_extras_url)
    while True:
        if choice1.strip() == 'Y':
            print('Next, you will want to move the downloaded tarball to a directory where all the Rosetta software '
                  'will be stored.\nOnce moved, you will want to unzip/extract all files in the tarball.\n'
                  'This can be done with the command tar -zxvf [put_tarball_name_here] in a new shell')
            input('This command may take some time. Once this is finished press Enter.\nIn the meantime, you may be '
                  'interested in reading about compilation \'%s\' and different features available with increased '
                  'computation time \'%s\'.\nInput:' % (rosetta_compile_url, rosetta_extras_url))
            input('Finally, Rosetta needs to be compiled. If you haven\'t taken a look, the process can be found at '
                  '\'%s\'. To take full advantage of computation time, think carefully about how your computing '
                  'environment can be set up to work with Rosetta. It is recommended for large design batches to '
                  'simply use default options for compilation. If you will want to have individual jobs finish quicker,'
                  'MPI compatability may be of interest. Navigate to the MPI resources in rosettacommons.org (links '
                  'above) for more information on setting this up.\n'
                  'Press Enter once you have completed the compilation process.\nInput:' % rosetta_compile_url)
            break
        elif choice1.strip() == 'S':
            break
        else:
            'Invalid input. Try again\nInput:'

    print('Great! Attempting to find environmental variable for Rosetta \'main\' directory')
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
                    print('Failed detection of Rosetta environmental variable = %s' % ', '.join(rosetta_variable_dictionary[i] for i in rosetta_variable_dictionary))
                    print('For setup to be successful, the location of your Rosetta install needs to be accessed. It is'
                          ' recommended to modify your shell to include an environmental variable \'ROSETTA\' (accessed'
                          ' by \'$ROSETTA\'), leading to the \'main\' directory of Rosetta.')
                    choice2 = input('If you have one, please enter it below (without $) or reply N to set one up.\n'
                                    'Input:')
                    if choice2.strip() == 'N':
                        print('To make the ROSETTA environmental variable always present at the command line, the '
                              'variable needs to be declared in your ~/.profile file (or analogous shell specific file '
                              'like .bashrc (or csh, zsh) if you prefer) To add yourself, append the command below to '
                              'your ~/.profile file, replacing the path/to/rosetta_src_20something.version#/main '
                              'with your actual path.\nexport ROSETTA=path/to/rosetta_src_20something.version#/main\n')
                        input('Once completed, press Enter.\nInput:')
                    else:
                        rosetta = str(os.environ.get(choice2.strip()))

                    if rosetta.endswith('/main'):
                        rosetta_env_variable = choice2
                    else:
                        i = -1
                        while True:
                            i += 1
                            rosetta = str(os.environ.get(rosetta_variable_dictionary[i]))
                            if rosetta.endswith('/main'):
                                rosetta_env_variable = rosetta_variable_dictionary[i]
                                break
                            elif i >= 3:
                                break
        if os.path.exists(rosetta):
            print('Wonderful, Rosetta environment located and exists. You can now use all the features of %s to interface with Rosetta'
                   % PUtils.program_name)
            print('All %s files are located in %s' % (PUtils.program_name, PUtils.source))
            break
        else:
            print('Rosetta environmental path doesn\'t exist. Ensure that $%s is correct... Trying again' % rosetta_env_variable)

    print('Next, the proper python environment needs to be set up. The following modules need to be available to '
          'python otherwise, %s will not run. These inclue:\n-sklearn\n-numpy\n-biopython')
    input('If you don\'t have these, use a package manager such as pip or conda to install these in your environment.\n'
          'Press Enter once these are located to continue.\nInput:')

    print('Finally, hh-suite needs to be available. This is being created for you in the dependencies directory.')
    hhsuite = subprocess.Popen()
    hhsuite.communicate()

    print('Set up is complete! You can now use %s for design of protein interfaces generated using Nanohedra.'
          % PUtils.program_name)
    print('To design materials, navigate to your desired Nanohedra output directory and run the command %s for details'
          % PUtils.command)

# TODO Set up SymDesign.sh and ProcessRosettaCommands.sh depending on status of PathUtils


if __name__ == '__main__':
    main()
