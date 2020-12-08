#lookup_uniprotdict.py
import pickle

# single call reduces the sources of error if we modify anything in the future
with open('200121_UniProtPDBMasterDict.pkl', 'rb') as unidictf:
    uniprot_pdb_d = pickle.load(unidictf)


# pull uniprotID of a pdb code out of the dictionary
def pull_uniprot_ib_by_pdb(pdb_code, chain=False):
    # uniprot_pdb_d = pickle.load(unidictf)
    source = 'unique_pdb'
    if chain:
        pdb_code = '%s.%s' % (pdb_code, chain)
        source = 'all'

    # pdb_chain = pdb_code + '.' + chain
    for uniprot_id in uniprot_pdb_d:
        if pdb_code in uniprot_pdb_d[uniprot_id][source]:
            return uniprot_id, uniprot_pdb_d[uniprot_id]

    # return uniprot_id, uniprot_pdb_d[uniprot_id]

    # else:
    #     uniprotid_dict = {}
    #     for uniprot_id in uniprot_pdb_d:
    #         if pdb_code in uniprot_pdb_d[uniprot_id]['unique_pdb']:
    #             uniprotid_dict[uniprot_id] = uniprot_pdb_d[uniprot_id]['all']
    #
    #     return uniprotid_dict, None


# pull attributedict using UniProtID
def pullattribute(uniprotid):
    # uniprot_pdb_d = pickle.load(unidictf)
    for item in uniprot_pdb_d:
        if item == uniprotid:
            attributedict = uniprot_pdb_d[uniprotid]
            return attributedict
        # else:
        #     continue
    else:
        print('Did not find information from that UniProtID.')
        main()


def lookupattribute(uniprotid, attributedict):
    attributesearchmethod = input(
        '\nWhat do you want to look up for UniProtID ' + uniprotid + '?\n1. All PDBs (including chains) associated with this UniProt\n2. All unique PDBs (no chain info) associated with this UniProt\n3. PDB with the highest resolution of this UniProt\n4. Collection of biological interfaces for this UniProt (unique PDBs and partner unique PDBs)\n5. Partner protein associated with this UniProt with a different ID\n6. Space groups\n(1/2/3/4/5/6): ').strip()
    if attributesearchmethod == '1':
        print('1. All PDBs (including chains) associated with this UniProt:\n' + str(attributedict['all']))
    elif attributesearchmethod == '2':
        print('2. All unique PDBs (no chain info) associated with this UniProt:\n' + str(attributedict['unique_pdb']))
    elif attributesearchmethod == '3':
        print('3. PDB with the highest resolution of this UniProt:\n' + str(attributedict['top']))
    elif attributesearchmethod == '4':
        print('4. Collection of biological interfaces for this UniProt (unique PDBs and partner unique PDBs):\n' + str(
            attributedict['bio']))
    elif attributesearchmethod == '5':
        print('5. Partner protein associated with this UniProt with a different ID:\n' + str(attributedict['partners']))
    elif attributesearchmethod == '6':
        print('6. Space groups:\n' + str(attributedict['space_groups']))
    else:
        attributesearchmethod = input('Incorrect choice, please enter number 1 through 6: ')
        lookupattribute(uniprotid, attributedict)

    restart = input('Look up another attribute for this UniProt? (y/n): ').lower()
    if restart == 'y':
        lookupattribute(uniprotid, attributedict)


def main():
    lookupmethod = input('\nChoose search method: 1. Look up by PDB; 2. Look up by UniProtID\n(1/2): ')

    # I have replaced the recursion with a while loop (and break statements) that will forever cycle until the user inputs the correct parameter
    while True:
        if lookupmethod == '1':
            pdbcode = input('Enter PDB code: ').upper().strip()
            chain = input('Enter chain ID (hit ENTER to leave blank): ').upper().strip()
            if chain == '':
                chain = False
            # with open('200121_UniProtPDBMasterDict.pkl', 'rb') as unidictf:  # place at the global scale

            # result = pull_uniprot_ib_by_pdb(pdb_code, chain)
            # uniprotid = result[0]
            # attributedict = result[1]
            # ^^ I replaced the above unpacking with a multiparameter return. Same thing, but thought I would show you this is possible.
            uniprotid, attributedict = pull_uniprot_ib_by_pdb(pdbcode, chain=chain)
            # If you ever want to see a rabbit hole of dynamically sized returns look into *args and **kwargs,
            # Not super important. I don't use these options much but they exist and many people find them useful

            print('\nUniProt ID: ' + str(uniprotid))
            if uniprotid == {}:
                print('Did not find that PDB.')
                # main()  # removed the need with the while look and break statements
            elif type(uniprotid) != str:
                print('Use UniProtID or Pick only 1 chain to continue working.')
                # main()
            else:
                break
        elif lookupmethod == '2':
            uniprotid = input('Enter UniProtID: ').upper().strip()
            attributedict = uniprot_pdb_d[uniprotid]
            # attributedict = pullattribute(uniprotid)
            break
        else:
            print('Invalid choice.')
            # main()

    lookupattribute(uniprotid, attributedict)
    restartall = input('Start a new search? (y/n): ').lower().strip()
    if restartall == 'y':
        main()
    else:
        print('--END--')
        exit()


if __name__ == '__main__':
    main()
