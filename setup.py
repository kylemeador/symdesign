from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from symdesign import utils
from symdesign.utils import path as putils, input_string

logger = utils.start_log()
# logger = logging.getLogger(__name__)
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
        logger.debug(f'"{search_variable}" environment inaccessible as no environmental variable was found at any of '
                     f'${", $".join(search_strings)}. If you believe there was a mistake, add this environmental '
                     f'variable to the {putils.config_file} file. Ex: '
                     '{'
                     f'"{search_variable}_env": {search_variable.upper()}, ...'
                     '}'
                     f' where the value {search_variable.upper()} is the environmental variable (ensure without $)')
    else:
        logger.debug(f'Found "{env_variable}" for the environmental variable ${string}')

    return string  # env_variable


restart_command = f'python {__file__}'
failure_prompt = f' If this command fails part way through, you can restart by running\n\t{restart_command}'


def download_alphafold_latest_params(version: str = None, dry_run: bool = False) -> str:
    """Return the name of the database version that was downloaded

    Args:
        version: Whether a particular version should be used. The default is the latest
        dry_run: Whether to perform a "dry install" without any substantial commands
    Returns:
        The name of the database version fetched
    """
    if version is None:
        # Default as of creation on 2/2/23
        version = 'alphafold_params_2022-12-06.tar'

    source_url = f'https://storage.googleapis.com/alphafold/{version}'
    # version = os.path.basename(source_url)

    putils.make_path(putils.alphafold_params_dir)
    os.chdir(putils.alphafold_params_dir)

    # Use the version to wget the file to the dependencies/alphafold/params directory
    putils.make_path(putils.hhsuite_db_dir)  # Make all dirs - dependencies/hhsuite/databases
    download_cmd = ['wget', '--directory-prefix', putils.alphafold_params_dir, '--continue',
                    '-U', '\'Mozilla/5.0', '(X11;', 'U;', 'Linux', 'i686', '(x86_64);', 'en-GB;',
                    'rv:1.9.0.1)', 'Gecko/2008070206', 'Firefox/3.0.1\'', '--inet4-only',
                    '--no-check-certificate', source_url
                    ]  # '--no-verbose'
    logger.debug(f'Download command:\n\t{subprocess.list2cmdline(download_cmd)}')
    if dry_run:
        pass
    else:
        logger.info(f'This command will take several minutes to finish depending on your internet speed... '
                    f'Please be patient and connected to this terminal.{failure_prompt}')
        download_p = subprocess.Popen(download_cmd)
        download_out, download_err = download_p.communicate()
        # logger.debug(f'download stdout:\n{download_out}\n\ndownload stderr:\n{download_err}')

    # Unzip the file to the dependencies/alphafold/params directory
    downloaded_file = os.path.join(putils.alphafold_params_dir, version)
    unzip_cmd = ['tar', 'xv', f'--file={downloaded_file}', f'--directory={putils.alphafold_params_dir}',
                 '--preserve-permissions']
    logger.debug(f'untar command:\n\t{subprocess.list2cmdline(unzip_cmd)}')
    if dry_run:
        pass
    else:
        unzip_p = subprocess.Popen(unzip_cmd)
        unzip_out, unzip_err = unzip_p.communicate()
        logger.debug(f'unzip stdout:\n{unzip_out}\n\nunzip stderr:\n{unzip_err}')

    # Remove the .tar file if everything worked
    if dry_run:
        pass
    else:
        os.remove(downloaded_file)

    # Download the stereo chemical props file and copy to the correct place
    # shutil.mkdir -p /alphafold/alphafold/common
    os.makedirs(putils.alphafold_common_dir, exist_ok=True)
    # shutil.copy('stereo_chemical_props.txt', putils.alphafold_common_dir)
    download_chem_cmd = ['wget', '-q', '-P', putils.alphafold_common_dir,
                         'https://git.scicore.unibas.ch/schwede/openstructure/-/raw/'
                         '7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt']
    logger.debug(f'Download stereo chemical props command:\n\t{subprocess.list2cmdline(download_chem_cmd)}')
    if dry_run:
        pass
    else:
        download_p = subprocess.Popen(download_chem_cmd)
        download_out, download_err = download_p.communicate()
        logger.debug(f'download stdout:\n{download_out}\n\ndownload stderr:\n{download_err}')

    # Todo If using the env version...
    # shutil.mkdir -p conda/lib/python3.8/site-packages/alphafold/common/
    # shutil.copy('/content/stereo_chemical_props.txt /opt/conda/lib/python3.8/site-packages/alphafold/common/

    # Make the openmm patch
    conda_env_path = os.environ['CONDA_PREFIX']
    vers = sys.version_info
    openmm_path = os.path.join(conda_env_path, 'lib', f'python{vers.major}.{vers.minor}', 'site-packages')

    os.chdir(openmm_path)
    patch_openmm_cmd = ['patch', '-p0', f'--input={putils.alphafold_openmm_patch}']  # '<',
    # putils.alphafold_openmm_patch]
    logger.debug(f'patch command:\n\t{subprocess.list2cmdline(patch_openmm_cmd)}')
    if dry_run:
        pass
    else:
        # with open(putils.alphafold_openmm_patch, 'rb') as f:
        #     path_lines = f.read()
        patch_p = subprocess.Popen(patch_openmm_cmd)  #, stdin=subprocess.PIPE)
        patch_out, patch_err = patch_p.communicate()  # input=path_lines)  # .encode('utf-8'))
        logger.debug(f'patch stdout:\n{patch_out}\n\ndownload stderr:\n{patch_err}')

    return downloaded_file


def download_hhblits_latest_database(version: str = None, dry_run: bool = False) -> str:
    """Return the name of the database version that was downloaded

    Args:
        version: Whether a particular version should be used. The default is the latest
        dry_run: Whether to perform a "dry install" without any substantial commands
    Returns:
        The name of the database version fetched
    """
    # hhblits_latest_url = 'https://wwwuser.gwdg.de/~compbiol/uniclust/uniclust-latest' NOT RIGHT
    # fetch_uniclust_latest = ['wget', 'http://wwwuser.gwdg.de/~compbiol/uniclust/current_release/',
    #                          '`wget', '-O', '- http://wwwuser.gwdg.de/~compbiol/uniclust/current_release/', '|', 'grep',
    #                          '-Po', '\'indexcolname"><a\ href="\K[^"]*_hhsuite.tar.gz\'`']

    if dry_run:
        stderr_dest = None
    else:
        stderr_dest = subprocess.DEVNULL
    # Set the version
    if version is None:
        version = 'current_release'
    uniclust_url = 'http://wwwuser.gwdg.de/~compbiol/uniclust/{}/{}'.format(version, '{}')

    # Acquire the name of the uniclust release
    # uniclust_version_request_cmd = ['wget', '-O', '-', uniclust_url.format(''), '--no-verbose']
    # wget_file = 'wget_result.temp'
    wget_file = Path('wget_result.temp')

    uniclust_version_request_cmd = ['wget', '--output-document', wget_file, uniclust_url.format(''), '--no-verbose']
    logger.debug(f'uniclust webpage fetch command:\n\t{subprocess.list2cmdline(uniclust_version_request_cmd)}')
    request_p = subprocess.Popen(uniclust_version_request_cmd)
    request_out, request_err = request_p.communicate()
    # # request_p = subprocess.Popen(uniclust_version_request_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # with open(wget_file, 'w') as request_f:
    #     request_p = subprocess.Popen(uniclust_version_request_cmd, stdout=request_f, stderr=stderr_dest)
    #     request_out, request_err = request_p.communicate()
    # # request_out = request_out.decode('utf-8')
    logger.debug(f'download stdout:\n{request_out}\n\ndownload stderr:\n{request_err}')

    # Format the output
    # This looks like raw html and assumes we want the only hhsuite version listed
    with open(wget_file, 'r') as f:
        wget_out = ''.join(f.readlines())
    wget_file.unlink(missing_ok=True)

    try:
        uniclust_latest_tar_file = re.search(f'[A-z0-9]*{putils.uniclust_hhsuite_file_identifier}', wget_out)[0]
    except IndexError:  # Found no matches
        raise RuntimeError(f"Couldn't retrieve the latest UniClust database from the url:\n{uniclust_url}")
    logger.info(f'UniClust latest file name: {uniclust_latest_tar_file}')
    # grep_latest_version = ['grep', '-Po', '\'indexcolname"><a\ href="\K[^"]*_hhsuite.tar.gz\'']
    # logger.debug(f'grep of desired version file command:\n\t{subprocess.list2cmdline(grep_latest_version)}')
    # grep_p = subprocess.Popen(grep_latest_version, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    # uniclust_latest_tar_file, grep_err = grep_p.communicate(input=request_out)
    # uniclust_latest_tar_file = uniclust_latest_tar_file.decode('utf-8')
    # logger.debug(f'unzip grep stdout (uniclust_latest_tar_file):\n\n'
    #              f'{uniclust_latest_tar_file}\ngrep stderr:\n{grep_err}')

    # Use the name of the latest file to wget the file to the dependencies/hhsuite directory
    putils.make_path(putils.hhsuite_db_dir)  # Make all dirs - dependencies/hhsuite/databases
    uniclust_download_cmd = ['wget', '--directory-prefix', putils.hhsuite_dir, '--continue',
                             '-U', '\'Mozilla/5.0', '(X11;', 'U;', 'Linux', 'i686', '(x86_64);', 'en-GB;',
                             'rv:1.9.0.1)', 'Gecko/2008070206', 'Firefox/3.0.1\'', '--inet4-only',
                             '--no-check-certificate', uniclust_url.format(uniclust_latest_tar_file)
                             ]  # '--no-verbose'
    logger.debug(f'UniClust download command:\n\t{subprocess.list2cmdline(uniclust_download_cmd)}')
    if dry_run:
        pass
    else:
        logger.info(f'This command will take several minutes to finish depending on your internet speed... '
                    f'Please be patient and connected to this terminal.{failure_prompt}')
        download_p = subprocess.Popen(uniclust_download_cmd)
        download_out, download_err = download_p.communicate()
        # logger.debug(f'download stdout:\n{download_out}\n\ndownload stderr:\n{download_err}')

    # Unzip the file to the dependencies/hhsuite/databases directory
    os.chdir(putils.hhsuite_db_dir)
    downloaded_file = os.path.join(putils.hhsuite_dir, uniclust_latest_tar_file)
    unzip_uniclust_cmd = ['tar', 'xzf', downloaded_file]  # , '-C', putils.hhsuite_dir]
    logger.debug(f'untar database command:\n\t{subprocess.list2cmdline(unzip_uniclust_cmd)}')
    if dry_run:
        pass
    else:
        unzip_p = subprocess.Popen(unzip_uniclust_cmd)
        unzip_out, unzip_err = unzip_p.communicate()
        logger.debug(f'unzip stdout:\n{unzip_out}\n\nunzip stderr:\n{unzip_err}')

    return uniclust_latest_tar_file.replace(putils.uniclust_hhsuite_file_identifier, '')


if __name__ == '__main__':
    # Todo argparse OR BETTER setuptools.py aware
    _dry_run = input(f'Is this a real install or a "dry run"? Enter REAL if so{input_string}')
    if _dry_run == 'REAL':
        dry_run = False
        logger.setLevel(logging.DEBUG)
    else:
        dry_run = True
        logger.setLevel(logging.DEBUG)
    logger.critical(f'Found the logger level: {logger.level}')

    # This needs to be done before running setup.py
    # print(f'To properly set up your python environment use the {putils.conda_environment} to initialize your '
    #       "environment. If you are using anaconda/conda for instance, the command 'conda env create --file "
    #       f"{putils.conda_environment}' will handle this for you")
    print(f'First, follow this url "{rosetta_url}" to begin licensing and download of the Rosetta Software suite if you'
          " haven't installed already")
    choice1 = utils.validate_input(
        'Once downloaded, type "Y" to continue with install or "S" to skip if Rosetta is already '
        "installed. FYI this program is capable of using Rosetta's multithreading and MPI builds for "
        f'faster execution. If you want to learn more, visit {rosetta_extras_url} for details.',
        ['Y', 's'])
    while True:
        if choice1 == 'Y':
            print('Next, you will want to move the downloaded tarball to a directory where all the Rosetta software '
                  'will be stored. Follow typical software recommendations for the directory or choose your own.\n'
                  'Once moved, you will want to unzip/extract all files in the tarball.\n'
                  'This can be done with the command tar -zxvf [tarball_name_here] in a new terminal')
            input('This command may take some time.\n'
                  f'In the meantime, you may be interested in reading about compilation "{rosetta_compile_url}" and '
                  f'different features available for increasing computation time "{rosetta_extras_url}".'
                  f'Once this is finished press "Enter" on your keyboard.{input_string}')
            input("Finally, lets compile Rosetta. If you aren't familiar with this process and haven't looked at the "
                  'above links, check them out for assistance. To take full advantage of computation time, think '
                  'carefully about how your computing environment can be set up to work with Rosetta. It is recommended'
                  ' for large design batches to simply use default options for compilation. If you want to have '
                  'individual jobs finish quicker, MPI compatibility may be of interest. Navigate to the MPI resources '
                  'in rosettacommons.org for more information on setting this up.\nPress "Enter" once you have '
                  f'completed compilation.{input_string}')
            break
        elif choice1 == 's':
            break
        else:
            raise RuntimeError(f'Must be either Y/S. Got {choice1}')

    print('Attempting to find environmental variable for Rosetta "main" directory')
    rosetta_env_variable = ''
    number_rosetta_variables = len(rosetta_variable_dictionary)
    retry_kw = 'retry'
    retry = None
    rosetta_make = None
    # Todo reconfigure with above environmental variable search
    i = 0
    while True:
        # try:
        rosetta_main = str(os.environ.get(rosetta_variable_dictionary[i]))
        if rosetta_main.endswith('main'):
            print('Automatic detection successful!')
            rosetta_env_variable = rosetta_variable_dictionary[i]
            # break
        else:
            i += 1
            if i == number_rosetta_variables:
                while rosetta_env_variable == '':
                    print('Failed detection of Rosetta environmental variable = %s' % ', '.join(
                        rosetta_variable_dictionary[i] for i in rosetta_variable_dictionary))
                    print('For setup to be fully functional, the location of your Rosetta install needs to be accessed.'
                          ' It is recommended to modify your shell to include an environmental variable "ROSETTA" '
                          '(accessed at "$ROSETTA"), leading to the "main" directory of Rosetta.')
                    choice2 = input(f'If you have one, please enter it below or reply N to set one up.{input_string}')
                    choice2 = choice2.lstrip('$')
                    if choice2.strip() == 'N':
                        print('To make the ROSETTA environmental variable always present at the command line, the '
                              'variable needs to be declared in your ~/.profile file (or analogous shell specific file '
                              'like .bashrc (cshrc, zshrc, or tcshrc if you prefer) To add yourself, append the command'
                              ' below to your ~/.profile file, replacing path/to/rosetta_src_20something.version#/main '
                              'with your actual path.\nexport ROSETTA=path/to/rosetta_src_20something.version#/main\n')
                        input(f'Once completed, press Enter.{input_string}')
                    else:
                        rosetta_main = str(os.environ.get(choice2.strip()))

                    if rosetta_main.endswith('main'):
                        rosetta_env_variable = choice2
                    else:
                        j = 0
                        while True:
                            rosetta_main = str(os.environ.get(rosetta_variable_dictionary[j]))
                            if rosetta_main.endswith('main'):
                                rosetta_env_variable = rosetta_variable_dictionary[j]
                                break

                            j += 1
                            if j == number_rosetta_variables:  # We ran out of variable attempts
                                break

                    if rosetta_env_variable == '':
                        retry = input(f'Rosetta dependency set up failed. To retry Rosetta connection, enter '
                                      f'"{retry_kw}"{input_string}')
                        break

            else:
                continue
        if os.path.exists(rosetta_main):
            print(f'Wonderful, Rosetta environment located and exists. You can now use all the features of '
                  f'{putils.program_name} to interface with Rosetta')
            make_types = ['mpi', 'default', 'python', 'mpi' 'cxx11thread', 'cxx11threadmpi']
            # if rosetta_env_variable == '':
            rosetta_make = utils.validate_input(
                'Did you make Rosetta with any particular build? This is usually a string of '
                'characters that are a suffix to each of the executables in the '
                f'{putils.rosetta_default_bin} directory', make_types)
            if rosetta_env_variable == '':
                rosetta_env_variable = search_env_for_variable(putils.rosetta_str)
            break
        elif retry == retry_kw:
            # print(f"Rosetta environmental path doesn't exist. Ensure that ${rosetta_env_variable} is correct... "
            print("Trying again...")
            break
        else:
            # Todo issue command to set this feature up at a later date... Rerun with --rosetta-only for help
            break

    # Set up git submodule
    git_submodule_cmd = ['git', 'submodule', 'update', '--init', '--recursive']
    # Set up freesasa dependency
    # Todo May need to investigate this option
    #  --disable-threads
    # Todo required PREINSTALL CHECK
    # sudo apt-get install build-essential autoconf libc++-dev libc++abi-dev
    freesasa_autoreconf_cmd = ['autoreconf', '-i']
    freesasa_configure_cmd = ['./configure', '--disable-xml', '--disable-json']
    make_cmd = ['make']
    # Set up stride (or Todo another secondary structure program if not linux...)
    ss_cmd = ['tar', '-zxf', 'stride.tar.gz']
    # Set up orient dependency
    # Todo required PREINSTALL CHECK
    # sudo apt install gfortran
    # Todo silence gfortran warnings
    orient_comple_cmd = ['gfortran', '-o', putils.orient_exe_path, f'{putils.orient_exe_path}.f']
    # Set up errat dependency
    errat_compile_cmd = ['g++', '-o', putils.errat_exe_path, putils.errat_residue_source]
    # Add a directory change to the command type
    change_dir_paths = defaultdict(lambda: os.getcwd(), {
        'git submodule set up': putils.git_source,
        'autoreconf freesasa': putils.freesasa_dir,
        'secondary structure analysis': putils.stride_dir,
    })
    # Set up the commands in addressable order
    commands = {
        'git submodule set up': git_submodule_cmd,  # 0
        'autoreconf freesasa': freesasa_autoreconf_cmd,  # 1
        'configure freesasa': freesasa_configure_cmd,  # 2
        'make': make_cmd,  # 3
        'secondary structure analysis': ss_cmd,  # 3
        'make2': make_cmd,  # 5
        'compile orient': orient_comple_cmd,  # 6
        'compile errat': errat_compile_cmd,  # 7
    }
    for idx, (command_type, command) in enumerate(commands.items(), 1):
        logger.debug(f'Command{idx} {command_type}:\n\t{subprocess.list2cmdline(command)}')
        if dry_run:
            pass
        else:
            # Possibly change directory
            os.chdir(change_dir_paths[command_type])
            # Execute the command
            p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            out, err = p.communicate()

    # Set up the program config file
    config = {'rosetta_env': rosetta_env_variable,
              'rosetta_main': rosetta_main,
              'rosetta_make': rosetta_make,
              'hhblits_env': search_env_for_variable(putils.hhblits),
              }

    # Get hhblits database
    _input = utils.validate_input('Finally, a UniClust database needs to be available for hhblits. The file will take '
                                  '>50 GB of hard drive space. Ensure that you have the capacity for this operation. '
                                  'This will automatically be downloaded for you in the directory '
                                  f'"{putils.hhsuite_db_dir}" if you consent.', ['Y', 'n'])
    if _input == 'n':
        # Todo issue command to set this feature up at a later date... Rerun with --hhsuite-databases for help
        pass
    else:  # _input = 'y'
        config['uniclust_db'] = download_hhblits_latest_database(dry_run=dry_run)

    # Get alphafold database. 5.3 is for params only
    _input = utils.validate_input('Finally, Alphafold databases need to be available for alphafold structure '
                                  'prediction. The download will take 5.3 GB of hard drive space. Ensure that you '
                                  'have the capacity for this operation. '
                                  'This will automatically be downloaded for you in the directory '
                                  f'"{putils.alphafold_db_dir}" if you consent.', ['Y', 'n'])
    if _input == 'n':
        # Todo issue command to set this feature up at a later date... Rerun with --alphafold-databases for help
        pass
    else:  # _input = 'y'
        config['af_params'] = download_alphafold_latest_params(dry_run=dry_run)

    # Write the config file
    utils.write_json(config, putils.config_file)

    # Todo Set up the module symdesign to be found in the PYTHONPATH variable
    print(f'Set up is now complete! {putils.program_name} is now operational. Run the command {putils.program_exe} for '
          f'usage instructions or visit the github for more info "{putils.git_url}')
    print(f'All {putils.program_name} files are located in {putils.git_source}')
