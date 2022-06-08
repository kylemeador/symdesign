#!/home/kmeador/miniconda3/bin/python
import os
import subprocess
import sys
from collections import defaultdict
from itertools import repeat, chain as iter_chain

from lxml import etree, html
from requests import get, post

from PDB import PDB
from PathUtils import pisa_db
from SymDesignUtils import pickle_object, to_iterable, remove_duplicates, io_save, start_log  # logger,

# from interface_analysis.InterfaceSorting import logger

logger = start_log(name=__name__)

# Globals
# pisa_type_extensions = {'multimers': '.xml', 'interfaces': '.xml', 'multimer': '.pdb', 'pisa': '.pkl'}
pisa_ref_d = {'multimers': {'ext': 'multimers.xml', 'source': 'pisa', 'mod': ''},
              'interfaces': {'ext': 'interfaces.xml', 'source': 'pisa', 'mod': ''},
              'multimer': {'ext': 'bioassembly.pdb', 'source': 'pdb', 'mod': ':1,1'}, 'pisa': '.pkl'}


def get_complex_interfaces(pdb_code):
    """Retrieve the coordinate only assembly analysis from PISA if a biological assembly on the PDB was not found in
    the multimers analysis.

    Caution this function has errors around ligands and dissociation patterns. Due to the incomplete data access of
    PISA this parsing is the best available for the ligand system. Multiple ligands are denoted with numbers and the
    dissociation pattern lacks these numbers, so the dissociation of interfaces may be inaccurate with regards to the
    exact PISA interface id that is dissociating when a ligand with more than one copy is involved
    """
    pdb_biomol = 1  # I believe this is always true
    header = {'User-Agent': 'Mozilla/5.0 (X11; CrOS x86_64 13505.100.0) AppleWebKit/537.36 (KHTML, like Gecko) '
                            'Chrome/87.0.4280.142 Safari/537.36'}
    pisa_query = get('https://www.ebi.ac.uk/pdbe/pisa/cgi-bin/piserver?qa=%s' % pdb_code, headers=header)
    # print(pisa_query.content)
    # print('\n' * 10)
    pisa_content_tree = html.fromstring(pisa_query.content)
    input_dir_key = pisa_content_tree.xpath('//input[@name="dir_key"]')
    session_id = input_dir_key[0].value
    assert len(session_id.split('-')) == 3, 'Error grabbing Session ID (%s) for %s. GET request:%s' \
                                            % (session_id, pdb_code, pisa_query)

    url = 'https://www.ebi.ac.uk/msd-srv/prot_int/cgi-bin/piserver'
    data = {'page_key': 'assembly_list_page', 'action_key': 'act_go_to_asmpage_p', 'dir_key': session_id,
            'session_point': '8', 'action_no': '-2_0'}  # This is for the complex page!
    assembly_query = post(url, data=data)
    pisa_content_tree = html.fromstring(assembly_query.content)
    complex_found = pisa_content_tree.xpath('//form/small')
    # print(pisa_query.content)
    # print('\n' * 10)
    if not complex_found:  # ensure that the complex has an assembly, it may be a monomer if we have reached this point
        return {'id': -2, 'composition': None, 'chains': {}, 'ligands': {}, 'stable': None, 'dg_diss': 0., 'dg_int': 0.,
                'pdb_BA': pdb_biomol, 'interfaces': {}}

    int_xml = get('https://www.ebi.ac.uk/msd-srv/prot_int/data/pisrv_%s/engagedinterfaces--2.xml' % session_id)
    # xml = get('https://www.ebi.ac.uk/msd-srv/prot_int/data/pisrv_674-OC-J26/engagedinterfaces--2.xml' % session_id)
    # print(int_xml.content)
    try:
        interface_tree = etree.fromstring(int_xml.content)
    except etree.XMLSyntaxError:  # there is no engagedinterfaces--2.xml document, return null result. See 1Z0H for ex
        return {'id': -2, 'composition': None, 'chains': {}, 'ligands': {}, 'stable': None, 'dg_diss': 0., 'dg_int': 0.,
                'pdb_BA': pdb_biomol, 'interfaces': {}}
    # interface_root = interface_tree.getroot()
    interface_d, interface_chains = {}, {}
    for interface in interface_tree.findall('ENGAGEDINTERFACE'):
        if len(interface) > 1:
            # type = int(interface.find('INTERFACETYPE').text)
            chains = interface.find('INTERFACESTRUCTURES').text
            chains = chains.split('+')  # Todo handle <INTERFACESTRUCTURES><sup>7</sup>B+A</INTERFACESTRUCTURES>
            number = int(interface.find('INTERFACENO').text) + 1  # offset to make same as interfaceID in interfaces.xml
            interface_chains[int(number)] = chains
            if int(number) in interface_d:
                interface_d[int(number)]['nocc'] += 1
            else:
                interface_d[int(number)] = {'nocc': 1, 'diss': 0}

    sum_xml = get('https://www.ebi.ac.uk/msd-srv/prot_int/data/pisrv_%s/assemblysummary--2.xml' % session_id)
    bad_attr = 'ASSEMBLYTITLE>'
    # this is required because the summary is malformed. Might change in the future, in which case its still compatible
    summary_tree = etree.fromstring(bytes(sum_xml.text[:sum_xml.text.find('\n<ASSEMBLYTYPE')]
                                          + sum_xml.text[sum_xml.text.rfind(bad_attr) + len(bad_attr):], 'utf-8'))
    # summary_root = summary_tree.getroot()
    # surface_area = int(summary_root.find('SURFACEAREA').text)
    # symmetry_number = int(summary_root.find('SYMMETRYNUMBER').text)
    # entropy_diss = float(summary_root.find('ENTROPYDISS').text)
    # complex_id = summary_root.find('MULTIMERICSTATE').text
    complex_composition = summary_tree.find('COMPOSITION').text
    chains, ligands = defaultdict(int), defaultdict(int)
    new_part, multi_id_part, last_part = '', False, None
    for idx, part_id in enumerate(complex_composition):
        if multi_id_part:
            if part_id == ']':  # the multipart is ending (please watch your step, thank you)
                multi_id_part = False  # stop collecting the parts
                if new_part.isdigit():  # check if it is a coefficient
                    if last_part in chains:  # check if coefficient applies to chains # if last_part: <-- not required?
                        chains[last_part] += int(new_part) - 1  # offset as we already added it once
                    else:  # it must apply to a ligand
                        ligands[last_part] += int(new_part) - 1  # offset as we already added it once
                else:  # it must be a ligand
                    ligands[new_part] += 1
                    last_part = new_part
                new_part = ''
            else:
                new_part += part_id  # add the multipart to a new part
        elif part_id == '[':  # we need the last part found
            multi_id_part = True
        else:
            chains[part_id] += 1
            last_part = part_id

    d_g_diss = float(summary_tree.find('DELTAGDISSOCIATION').text)
    d_g_int = float(summary_tree.find('DELTAGFORMATION').text)
    if d_g_diss >= 0:
        stable = True
    else:
        stable = False

    # REMARK 350 from PDB indicating biological assembly. If R350 = 0, no assigned BA for set complex
    # pdb_biomol = int(summary_root.find('BIOMOLECULER350').text)
    diss_pattern = summary_tree.find('DISSOCIATIONPATTERN').text
    diss_list = diss_pattern.split('+')
    for group in diss_list:
        if len(group) <= 1:  # any interface with a single chain is dissociated
            for interface_number, chain_pair in interface_chains.items():
                if group in chain_pair:
                    print('Group: %s\tChain Pair: %s' % (group, chain_pair))
                    interface_d[interface_number]['diss'] = 1
        else:  # only interfaces with both chains in group are not dissociated
            group = group.strip()
            if group.find('[') != -1:
                # group_item = group[group.find('['): group.find(']')]
                group = group.split('[')  # : group.find(']')]
                group = list(map(str.rstrip, group, repeat(']')))
                remove = []
                for idx, element in enumerate(group):
                    if element.isdigit():
                        group.append(group[idx - 1])
                        remove.append(element)
                for element in remove:
                    group.remove(element)
            for group_part in group:
                for interface_number, chain_pair in interface_chains.items():
                    if group_part in chain_pair:
                        if chain_pair[0] not in group or chain_pair[1] not in group:
                            # print('Group Pair: %s\tChain Pair: %s' % (group, chain_pair))
                            interface_d[interface_number]['diss'] = 1

    return {'id': -2, 'composition': complex_composition.strip(), 'chains': chains, 'ligands': ligands,
            'stable': stable, 'dg_diss': d_g_diss, 'dg_int': d_g_int, 'pdb_BA': pdb_biomol, 'interfaces': interface_d}


def parse_pisa_multimers_xml(xml_file_path):  # , download_structures=False, outdir=None, force_rerun=False):
    # function retrieved from ssbio on August 27, 2019 from pisa.py, modified for single xml use and interfaces
    """Retrieve PISA information from an XML results file
    See: http://www.ebi.ac.uk/pdbe/pisa/pi_download.html for more info
    XML description of macromolecular assemblies:
        http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/multimers.pisa?pdbcodelist
        where "pdbcodelist" is a comma-separated (strictly no spaces) list of PDB codes. The resulting file contain XML
        output of assembly data, equivalent to that displayed in PISA assembly pages, for each of the specified PDB
        entries.   NOTE: If a mass-download is intended, please minimize the number of retrievals by specifying as many
        PDB codes in the URL as feasible (20-50 is a good range), and never send another URL request until the previous
        one has been completed (meaning that the multimers.pisa file has been downloaded). Excessive requests will
        silently die in the server queue.
    Args:
        (str): Path to xml of pdb of interest
    Returns:
        (dict): Parsed PISA multimers
        {(1, 1): {'id': '1', 'composition': 'B[8]', 'chains': defaultdict(<class 'int'>, {'A': 4, 'C': 4}),
                  'ligands': {}, 'stable': True, 'd_g_diss': 30.2, 'd_g_int': -85.34, 'pdb_BA': 1,
                  'interfaces': {1: {'nocc': 4, 'diss': 0}, ...},
         (1, 2): ...}}

        # KM Added below.
        There is a potential logic error when using the [( set_ID, complex_ID), ...] notation below in that there
        may be multiple sets with the same set_ID, complex_ID. I have since modified the code to use a counter instead
        of complex id and created a new field for complex id.
        Format:
        {(set_ID, counter): {asssembly_dict}, ...}
        assembly_dict = {complex_ID, assembly_composition: , assembly_num_each_chain: , ligands: , complex_stable?: ,
        DeltaG_of_dissociation: , DeltaG_of_interface: , PDB_Biological_Assembly#: , Interface_data: }
    """
    parser = etree.XMLParser(ns_clean=True)
    tree = etree.parse(xml_file_path, parser)
    root = tree.getroot()
    pisa = {}
    for pdb in root.findall('pdb_entry'):
        # Check the assembly status
        # status = pdb.find('status').text
        # errors = ['Entry not found', 'Overlapping structures', 'No symmetry operations']
        # if status in errors:
        #     pisa['status'] = status

        pdb_code = pdb.find('pdb_code')
        # Check monomer status
        # try:
        total_asm = pdb.find('total_asm')
        if total_asm is None:
            num_complexes = 0
        else:
            num_complexes = int(total_asm.text)
        # except AttributeError:

        if num_complexes == 0:
            pisa[(1, 1)] = 'MONOMER'

        elif num_complexes > 0:
            # sets - All sets which could make up the crystal
            sets = pdb.findall('asm_set')
            # set_id - One PQS set that denotes how the crystal could be built by combinations of underlying complexes
            # complex_id - multimer IDs for individual complexes
            for s in sets:
                set_id = int(s.find('ser_no').text)
                # All assemblies
                all_assemblies = s.findall('assembly')
                for counter, assembly in enumerate(all_assemblies, 1):
                    # This part tells you the actual composition of the predicted complex (chains and ligands)
                    parts = assembly.findall('molecule')
                    chains, ligands = defaultdict(int), defaultdict(int)
                    for part in parts:
                        part_id = part.find('chain_id').text
                        if part_id.startswith('['):
                            # part_id = 'LIG_' + part_id.split(']')[0].strip('[')
                            ligands[part_id.split(']')[0].strip('[')] += 1
                        else:
                            chains[part_id] += 1

                    # ligands = {str(key.split('_')[1]): chains.pop(key) for key in list(chains.keys())
                    #            if key.startswith('LIG_')}
                    ############################################################################################
                    complex_id = int(assembly.find('id').text)
                    complex_composition = str(assembly.find('composition').text)
                    d_g_diss = float(assembly.find('diss_energy').text)
                    d_g_int = float(assembly.find('int_energy').text)
                    # REMARK 350 from PDB indicating biological assembly. If R350 = 0, no assigned BA for set complex
                    pdb_biomol = int(assembly.find('R350').text)
                    if d_g_diss >= 0:
                        stable = True
                    else:
                        stable = False

                    ############################################################################################
                    # KM added 8/27/19
                    interface_d = {}
                    interfaces = assembly.find('interfaces')
                    # for i in interfaces:
                    #     interfaces = i.findall('interface')
                    for interface in interfaces:
                        occurrences = int(interface.find('nocc').text)
                        if occurrences > 0:
                            int_id = int(interface.find('id').text)
                            # nocc = int(interface.find('nocc').text)
                            diss = int(interface.find('diss').text)
                            interface_d[int_id] = {'nocc': occurrences, 'diss': diss}

                    ############################################################################################
                    # if complex_id in pisa:
                    # if (set_id, complex_id) in pisa:
                    #     continue
                    # else:
                    #     pisa[complex_id] = adder
                    pisa[(set_id, counter)] = \
                        {'id': complex_id, 'composition': complex_composition.strip(), 'chains': chains,
                         'ligands': ligands, 'stable': stable, 'dg_diss': d_g_diss, 'dg_int': d_g_int,
                         'pdb_BA': pdb_biomol, 'interfaces': interface_d}

    # check to see if the biologcial assembly exists with in the multimers. Otherwise, get interface info about it
    for set_complex_dict in pisa.values():
        if set_complex_dict['pdb_BA'] != 0:
            return pisa

    pisa[(0, 0)] = get_complex_interfaces(pdb_code)
    return pisa


def parse_pisa_interfaces_xml(file_path):
    """Retrieve PISA information from an XML results file
    Args:
        file_path (str): path to xml of pdb of interest

    Returns:
        interface_d (dict): {InterfaceID: {Stats, Chain Data}}
            Chain Data - {ChainID: {Name: , Rot: , Trans: , AtomCount: , Residues: {Number: BSA, ... }}
            Ex: {1: {'occ': 2, 'area': 998.23727478, 'solv_en': -11.928783903, 'stab_en': -15.481081211,
                 'chain_data': {1: {'chain': 'C', 'r_mat': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                    't_vec': [0.0, 0.0, 0.0], 'num_atoms': 104,
                                    'int_res': {'87': 23.89, '89': 45.01, ...},
                                2: ...}},
                 2: {'occ': ..., },
                 'all_ids': {interface_id1: [interface_id1, matching_id2], ...}
                } interface_type and id connect the interfaces that are the same, but present in multiple PISA complexes

        chain_residue_d (dict): {chain: {residue data}}
            Ex: {'A': {1: {'aa': 'ALA', 'asa': 50.172800005}, 2: {'aa': ....}, ...}, 'B': {}, ...}

    This finds all the unique interfaces by the interface 'type' it then only returns those interfaces that are unique
    """
    parser = etree.XMLParser(ns_clean=True)
    tree = etree.parse(file_path, parser)
    root = tree.getroot()
    # coord_list = ['i', 'j', 'k']

    pdb = root.find('pdb_entry')
    interface_d, chain_residue_d, interface_types = {}, {}, {}
    observed_chains = set()
    interfaces = pdb.findall('interface')
    for interface in interfaces:
        # Remove redundant interfaces
        int_id = int(interface.find('id').text)
        int_type = int(interface.find('type').text)
        if int_type in interface_types:
            interface_types[int_type].append(int_id)
            continue
        else:
            interface_types[int_type] = [int_id]
        n_occ = int(interface.find('n_occ').text)
        area = float(interface.find('int_area').text)
        solv_en = float(interface.find('int_solv_en').text)
        stabilization_en = float(interface.find('stab_en').text)  # solv_en - h_bonds(~0.5) - salt_bridges(~0.3)

        interface_d[int_id] = {'occ': n_occ, 'area': area, 'solv_en': solv_en, 'stab_en': stabilization_en,
                               'chain_data': {}}

        # residue_default_dict = {'seq_num': 0, 'aa': 'x', 'asa': -1, 'bsa': -1}
        molecules = interface.findall('molecule')
        for z, molecule in enumerate(molecules):
            chain_id = molecule.find('chain_id').text
            # remove ligands from interface analysis
            if chain_id[0] == '[':  # we have a ligand
                interface_d[int_id]['chain_data'][z] = {'chain': False, 'ligand': chain_id}
            else:  # we have a chain
                position_matrix = [
                    [float(molecule.find('rxx').text), float(molecule.find('rxy').text), float(molecule.find('rxz').text)],
                    [float(molecule.find('ryx').text), float(molecule.find('ryy').text), float(molecule.find('ryz').text)],
                    [float(molecule.find('rzx').text), float(molecule.find('rzy').text), float(molecule.find('rzz').text)]]
                translation_vector = [float(molecule.find('tx').text), float(molecule.find('ty').text),
                                      float(molecule.find('tz').text)]
                num_atoms = int(molecule.find('int_natoms').text)
                # sym_op = molecule.find('symop_no').text
                # unit_cell = []
                # for i in coord_list:
                #     unit_cell.append(int(molecule.find('cell_' + i).text))
                # num_residues = molecule.find('int_nres').text

                int_residues = {}
                resi = molecule.findall('residues')
                for i in resi:
                    residues = i.findall('residue')
                    if chain_id not in observed_chains:
                        observed_chains.add(chain_id)
                        residue_dict = {}
                        for residue in residues:
                            seq_num = int(residue.find('seq_num').text)
                            aa = residue.find('name').text
                            asa = round(float(residue.find('asa').text), 2)
                            residue_dict[seq_num] = {'aa': aa, 'asa': asa}
                        chain_residue_d[chain_id] = residue_dict

                    # Find the residues which are buried in an interface
                    for residue in residues:
                        bsa = round(float(residue.find('bsa').text), 2)
                        if bsa != 0:
                            int_residues[int(residue.find('seq_num').text)] = bsa
                            # int_residues.append((int(residue.find('seq_num').text), bsa))
                interface_d[int_id]['chain_data'][z] = {'chain': chain_id, 'r_mat': position_matrix,
                                                        't_vec': translation_vector, 'num_atoms': num_atoms,
                                                        'int_res': int_residues}  # 'num_resi': num_residues,
                #                                       'symop_no': sym_op, 'cell': unit_cell}

    interface_d['all_ids'] = {int_types[0]: int_types for int_types in interface_types.values()}

    return interface_d, chain_residue_d


def retrieve_pisa_file_path(pdb_code, directory=pisa_db, pisa_data_type='pisa'):
    """Returns the PISA path that corresponds to the PDB code from the local PISA Database.
    Attempts a download if the files are not found
    """
    if pisa_data_type in pisa_ref_d:  # type_extensions:
        pdb_code = pdb_code.upper()
        sub_dir = pdb_code[1:3].lower()  # if subdir is used in directory structure
        root_path = os.path.join(directory, sub_dir)
        specific_file = ' %s_%s' % (pdb_code, pisa_ref_d[pisa_data_type]['ext'])  # file_type, type_extensions[file_type])
        if os.path.exists(os.path.join(root_path, specific_file)):
            return os.path.join(root_path, specific_file)
        else:  # attempt to make the file if not pickled or download if requisite files don't exist
            if pisa_data_type == 'pisa':
                if extract_pisa_files_and_pickle(root_path, pdb_code):  # return True if requisite files found
                    return os.path.join(root_path, specific_file)
                else:  # try to download required files, they were not found
                    logger.info('Attempting to download PISA files')
                    for pisa_data_type in pisa_ref_d:
                        download_pisa(pdb_code, pisa_data_type, out_path=directory)
                    if extract_pisa_files_and_pickle(root_path, pdb_code):  # return True if requisite files found
                        return os.path.join(root_path, specific_file)
            else:
                download_pisa(pdb_code, pisa_data_type, out_path=directory)

    return None  # if Download failed, or the files are not found


def extract_pisa_files_and_pickle(root, pdb_code):
    """Take a set of pisa files acquired from pisa server and parse for saving as a complete pisa dictionary"""
    individual_d = {}
    try:
        interface, chain = parse_pisa_interfaces_xml(os.path.join(root, '%s_interfaces.xml' % pdb_code))
        individual_d['interfaces'] = interface
        individual_d['chains'] = chain
        individual_d['multimers'] = parse_pisa_multimers_xml(os.path.join(root, '%s_multimers.xml' % pdb_code))
    except OSError:
        return False

    pickle_object(individual_d, '%s_pisa' % pdb_code, out_path=root)

    return True


def download_pisa(pdb, pisa_type, out_path=os.getcwd(), force_singles=False):
    """Downloads PISA .xml files from http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/
    Args:
        pdb (str,list): Either a single pdb code, a list of pdb codes, or a file with pdb codes, comma or newline
            delimited
        pisa_type (str): Either 'multimers', 'interfaces', or 'multimer' to designate the PISA File Source
    Keyword Args:
        out_path=os.getcwd() (str): Path to download PISA files
        force_singles=False (bool): Whether to force downloading of one file at a time
    Returns:
        None
    """
    import xml.etree.ElementTree as ETree

    def retrieve_pisa(pdb_code, _type, filename):
        p = subprocess.Popen(['wget', '-q', '-O', filename, 'https://www.ebi.ac.uk/pdbe/pisa/cgi-bin/%s.%s?%s' %
                              (_type, pisa_ref_d[_type]['source'], pdb_code)])
        if p.returncode != 0:  # Todo if p.returncode
            return False
        else:
            return True

    def separate_entries(tree, ext, out_path=os.getcwd()):
        for pdb_entry in tree.findall('pdb_entry'):
            if pdb_entry.find('status').text.lower() != 'ok':
                failures.extend(modified_pdb_code.split(','))
            else:
                # PDB code is uppercase when returned from PISA interfaces, but lowercase when returned from PISA Multimers
                filename = os.path.join(out_path, '%s_%s' % (pdb_entry.find('pdb_code').text.upper(), ext))
                add_root = ETree.Element('pisa_%s' % pisa_type)
                add_root.append(pdb_entry)
                new_xml = ETree.ElementTree(add_root)
                new_xml.write(open(filename, 'w'), encoding='unicode')  # , pretty_print=True)
                successful_downloads.append(pdb_entry.find('pdb_code').text.upper())

    def process_download(pdb_code, file):
        # nonlocal fail
        nonlocal failures
        if retrieve_pisa(pdb_code, pisa_type, file):  # download was successful
            # Check to see if <status>Ok</status> for the download
            etree = ETree.parse(file)
            if force_singles:
                if etree.find('status').text.lower() == 'ok':
                    successful_downloads.append(pdb_code)
                    # successful_downloads.extend(modified_pdb_code.split(','))
                else:
                    failures.extend(modified_pdb_code.split(','))
            else:
                separate_entries(etree, pisa_ref_d[pisa_type]['ext'])
        else:  # download failed
            failures.extend(modified_pdb_code.split(','))

    if pisa_type not in pisa_ref_d:
        logger.error('%s is not a valid PISA file type' % pisa_type)
        sys.exit()
    if pisa_type == 'multimer':
        force_singles = True

    file = None
    clean_list = to_iterable(pdb)
    count, total_count = 0, 0
    multiple_mod_code, successful_downloads, failures = [], [], []
    for pdb in clean_list:
        pdb_code = pdb[0:4].lower()
        file = os.path.join(out_path, '%s_%s' % (pdb_code.upper(), pisa_ref_d[pisa_type]['ext']))
        if file not in os.listdir(out_path):
            if not force_singles:  # concatenate retrieval
                count += 1
                multiple_mod_code.append(pdb_code)
                if count == 50:
                    count = 0
                    total_count += count
                    logger.info('Iterations: %d' % total_count)
                    modified_pdb_code = ','.join(multiple_mod_code)
                else:
                    continue
            else:
                modified_pdb_code = '%s%s' % (pdb_code, pisa_ref_d[pisa_type]['mod'])
                logger.info('Fetching: %s' % pdb_code)

            process_download(modified_pdb_code, file)
            multiple_mod_code = []

    # Handle remaining codes in concatenation instances where the number remaining is < 50
    if count > 0 and multiple_mod_code != list():
        modified_pdb_code = ','.join(multiple_mod_code)
        process_download(modified_pdb_code, file)

    # Remove successfully downloaded files from the input
    # duplicates = []
    for pdb_code in successful_downloads:
        if pdb_code in clean_list:
            # try:
            clean_list.remove(pdb_code)
        # except ValueError:
        #     duplicates.append(pdb_code)
    # if duplicates:
    #     logger.info('These files may be duplicates:', ', '.join(duplicates))

    if not clean_list:
        return True
    else:
        failures.extend(clean_list)  # should just match clean list ?!
        failures = remove_duplicates(failures)
        logger.warning('Download PISA Failures:\n[%s]' % failures)
        io_save(failures)

        return False


def extract_xtal_interfaces(pdb_path):  # unused
    source_pdb = PDB.from_file(pdb_path)
    interface_data = parse_pisa_interfaces_xml(pdb_path)
    for interface in interface_data:
        chains = []
        for chain in interface['chain_data']:
            rot, trans = chain['r_mat'], chain['t_vec']
            all_resi_atoms = []
            for res_num in chain['int_res']:
                all_resi_atoms.extend(source_pdb.chain(chain).residue(res_num))
            chain_pdb = PDB.from_atoms(all_resi_atoms)
            chain_pdb.apply(rot, trans)
            chains.append(chain_pdb)
        interface_pdb = PDB.from_atoms(list(iter_chain.from_iterable(chain.atoms for chain in chains)))
        interface_pdb.write(pdb_path + interface + '.pdb')


def parse_pisas(pdb_code, out_path=pisa_db):  #  download=False,
    """Parse PISA files for a given PDB code"""
    # files = ['multimers', 'interfaces']
    # if download:
    #     for i in files:
    #         download_pisa(pdb_code, i)

    path = os.path.join(out_path, pdb_code.upper())
    multimers = parse_pisa_multimers_xml('%s_multimers.xml' % path)
    interfaces, chain_data = parse_pisa_interfaces_xml('%s_interfaces.xml' % path)

    return multimers, interfaces, chain_data


if __name__ == '__main__':
    assemblies, interface_data, chain_residue_data = parse_pisas('1BVS')
    print(assemblies)
    print(interface_data)
    print(chain_residue_data)
