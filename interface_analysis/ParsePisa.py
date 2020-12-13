#!/home/kmeador/miniconda3/bin/python
import itertools
import os
from collections import defaultdict
from copy import deepcopy

import align
from lxml import etree
from symdesign.SymDesignUtils import download_pisa


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
        xml_file_path (str): path to xml of pdb of interest
    Returns:
        pisa (dict): of parsed PISA information.
        # KM Added below.
        There is a potential logic error when using the [( set_ID, complex_ID), ...] notation below in that there
        may be multiple sets with the same set_ID, complex_ID. I have since modified the code to use a counter instead
        of complex id and created a new field for complex id.
        Format:
        {(set_ID, counter): {asssembly_dict}, ...}
        assembly_dict = {complex_ID, assembly_composition: , assembly_num_each_chain: , ligands: , complex_stable?: ,
        DeltaG_of_dissociation: , DeltaG_of_interface: , PDB_Biological_Assembly#: , Interface_data: }
        Ex:
        {(1, 1): {'id': '1', 'composition': 'B[8]', 'chains': defaultdict(<class 'int'>, {'A': 4, 'C': 4}), 'ligands': {},
        'stable': True, 'd_g_diss': 30.2, 'd_g_int': -85.34, 'pdb_BA': 1, 'interfaces': {1: {'nocc': 4, 'diss': 0}, ...}
        , (1, 2): ...}}
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
                counter = 1
                for assembly in all_assemblies:
                    # This part tells you the actual composition of the predicted complex (chains and ligands)
                    parts = assembly.findall('molecule')
                    chains = defaultdict(int)
                    for part in parts:
                        part_id = part.find('chain_id').text
                        if part_id.startswith('['):
                            part_id = 'LIG_' + part_id.split(']')[0].strip('[')
                        chains[str(part_id)] += 1

                    ligands = {}
                    for key in deepcopy(chains).keys():
                        if key.startswith('LIG_'):
                            ligands[str(key.split('_')[1])] = chains.pop(key)
                    ############################################################################################
                    adder = {}

                    complex_id = int(assembly.find('id').text)
                    complex_composition = str(assembly.find('composition').text)
                    d_g_diss = float(assembly.find('diss_energy').text)
                    d_g_int = float(assembly.find('int_energy').text)
                    pdb_biomol = int(assembly.find('R350').text)  # If R350 = 0, no assigned PDB bioassembly

                    if d_g_diss >= 0:
                        stable = True
                    else:
                        stable = False

                    adder['id'] = complex_id
                    adder['composition'] = complex_composition.strip()
                    adder['chains'] = chains
                    adder['ligands'] = ligands
                    adder['stable'] = stable
                    adder['dg_diss'] = d_g_diss
                    adder['dg_int'] = d_g_int
                    adder['pdb_BA'] = pdb_biomol
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
                            nocc = int(interface.find('nocc').text)
                            diss = int(interface.find('diss').text)
                            interface_d[int_id] = {'nocc': nocc, 'diss': diss}

                    adder['interfaces'] = interface_d
                    ############################################################################################
                    # if complex_id in pisa:
                    # if (set_id, complex_id) in pisa:
                    #     continue
                    # else:
                    #     pisa[complex_id] = adder
                    pisa[(set_id, counter)] = adder
                    counter += 1

    return pisa


def parse_pisa_interfaces_xml(file_path):
    """Retrieve PISA information from an XML results file
    Args:
        file_path (str): path to xml of pdb of interest

    Returns:
        interface_d (dict): {InterfaceID: {Stats, Chain Data}}
            Chain Data - {ChainID: {Name: , Rot: , Trans: , AtomCount: , Residues: {Num: , BSA: }}
            Ex: {1: {'occ': 2, 'area': 998.23727478, 'solv_en': -11.928783903, 'stab_en': -15.481081211,
                 'chain_data': {1: {'chain': 'C', 'r_mat': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                    't_vec': [0.0, 0.0, 0.0], 'num_atoms': 104,
                                    'int_res': {'87': 23.89, '89': 45.01, ...},
                                2: ...}},
                 2: {'occ': ..., },
                 'all_ids': {interface_type: [interface_id1, matching_id2], ...}
                } interface_type and id connect the interfaces that are the same, but present in multiple PISA complexes

        chain_residue_d (dict): {chain: {residue data}}
            Ex: {'C': {4: {'aa': 'ALA', 'asa': 50.172800005}, 5: {'aa': ....}}}

    This finds all the unique interfaces by the interface 'type' it then only returns those interfaces that are unique
    """
    parser = etree.XMLParser(ns_clean=True)
    tree = etree.parse(file_path, parser)
    root = tree.getroot()
    # coord_list = ['i', 'j', 'k']

    pdb = root.find('pdb_entry')
    interface_d, chain_residue_d, interface_types = {}, {}, {}
    observed_chains = {}
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

        # bool_chain_dict = {}
        interface_d[int_id] = {'occ': n_occ, 'area': area, 'solv_en': solv_en, 'stab_en': stabilization_en,
                               'chain_data': {}}

        # residue_default_dict = {'seq_num': 0, 'aa': 'x', 'asa': -1, 'bsa': -1}
        molecules = interface.findall('molecule')
        for z, molecule in enumerate(molecules):
            chain_id = molecule.find('chain_id').text
            # remove ligands from interface analysis
            if chain_id[0] != '[':
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
                    # if populate:
                    if chain_id not in observed_chains:
                        observed_chains.add(chain_id)
                        residue_dict = {}
                        for residue in residues:
                            seq_num = int(residue.find('seq_num').text)
                            aa = residue.find('name').text
                            asa = round(float(residue.find('asa').text), 2)
                            # bsa = float(residue.find('bsa').text)
                            residue_dict[seq_num] = {'aa': aa, 'asa': asa}  # , 'bsa': bsa}
                        chain_residue_d[chain_id] = residue_dict

                    for residue in residues:
                        bsa = round(float(residue.find('bsa').text), 2)
                        if bsa != 0:
                            int_residues[int(residue.find('seq_num').text)] = bsa
                            # int_residues.append((int(residue.find('seq_num').text), bsa))
                interface_d[int_id]['chain_data'][z] = {'chain': chain_id, 'r_mat': position_matrix,
                                                           't_vec': translation_vector, 'num_atoms': num_atoms,
                                                           'int_res': int_residues}  # 'num_resi': num_residues,
                #                                          'symop_no': sym_op, 'cell': unit_cell}
                # bool_chain_dict[z] = {'chain': chain_id, 'r_mat': position_matrix, 't_vec': translation_vector,
                #                       'num_atoms': num_atoms, 'int_res': int_residues}
                #                       'symop_no': sym_op, 'cell': unit_cell, 'num_resi': num_residues}
            else:
                interface_d[int_id]['chain_data'][z] = {'chain': False, 'ligand': chain_id}
                # bool_chain_dict[z] = {'chain': False, 'ligand': chain_id}

    interface_d['all_ids'] = {interface_types[int_type][0]: interface_types[int_type] for int_type in interface_types}

    return interface_d, chain_residue_d


def parse_pisas(pdb_code, download=False, out_path=os.getcwd()):
    """Parse PISA files for a given PDB code"""
    files = ['multimers', 'interfaces']
    path = os.path.join(out_path, pdb_code.upper())
    if download:
        for i in files:
            download_pisa(pdb_code, i)

    multimers = parse_pisa_multimers_xml('%s_multimers.xml' % path)
    interfaces, chain_data = parse_pisa_interfaces_xml('%s_interfaces.xml' % path)

    return multimers, interfaces, chain_data


def extract_xtal_interfaces(pdb):
    source_pdb = align.PDB()
    source_pdb.readfile(pdb)
    interface_data = parse_pisa_interfaces_xml(pdb)
    for interface in interface_data:
        interface_pdb = align.PDB()
        for chain in interface['chain_data']:
            # chain = key
            chain_pdb = align.PDB()
            rot = chain['r_mat']
            trans = chain['t_vec']
            resi = chain['int_res']
            all_resi_atoms = []
            for n in resi:
                resi_atoms = source_pdb.getResidueAtoms(chain, n)
                all_resi_atoms.append(resi_atoms)
            interface_atoms = list(itertools.chain.from_iterable(all_resi_atoms))
            chain_pdb.read_atom_list(interface_atoms)
            chain_pdb.apply(rot, trans)
            interface_pdb.read_atom_list(chain_pdb.all_atoms)
        interface_pdb.write(pdb + interface + '.pdb')


if __name__ == '__main__':
    assemblies, interface_data, chain_residue_data = parse_pisas('1BVS')
    print(assemblies)
    print(interface_data)
    print(chain_residue_data)
