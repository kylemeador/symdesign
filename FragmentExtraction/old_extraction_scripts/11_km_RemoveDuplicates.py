import math
import sys
import os
import numpy as np
from PDB import PDB
from Bio.SeqUtils import IUPACData


def main():
    directory = '/home/kmeador/yeates/fragment_database/IJK_fragment_pairsANDreps/ijk_clustered_xtal_fragDB_1A'

    list_to_remove = []
    # list_to_remove2 = []
    # matching_pdbs = []
    pdbs = []
    for dirpath1, dirnames1, filenames1 in os.walk(directory):
        if not dirnames1:
            for fname1 in filenames1:
                if fname1.endswith(".pdb"):
                    pdbs.append((fname1, dirpath1))
                    # pdbs.append((fname1.split('_')[0], fname1.split('_')[3], fname1.split('_')[4], dirpath1))
    print('Number of Fragment files:', len(pdbs))
    sys.exit()
            # for pdb_1 in range(len(pdbs) - 1):
            #     for pdb_2 in range(pdb_1 + 1, len(pdbs)):
            #         if pdbs[pdb_1][0].split('_')[0] == pdbs[pdb_2][0].split('_')[0]:
            #             # Added below 2 steps (chain check) for Biological library
            #             if pdbs[pdb_1][0].split('_')[6] == pdbs[pdb_2][0].split('_')[10]:
            #                 if pdbs[pdb_1][0].split('_')[10] == pdbs[pdb_2][0].split('_')[6]:
            #                     # matching_pdbs.append(pdbs[pdb_1])
            #                     if pdbs[pdb_1][0].split('_')[3] == pdbs[pdb_2][0].split('_')[4]:
            #                         if pdbs[pdb_1][0].split('_')[4] == pdbs[pdb_2][0].split('_')[3]:
            #                             list_to_remove.append(pdbs[pdb_1])
            #                             # list_to_remove.append(pdbs[pdb_2])


    print('Length of list to remove:', len(list_to_remove))
    import collections
    dup = [item for item, count in collections.Counter(list_to_remove).items() if count > 1]
    print('Number of Duplicates: ', len(dup))
    # print(len(matching_pdbs))
    # print(len(list_to_remove2))
    list_to_remove_set = set(list_to_remove)
    print('Length of Set to remove:', len(list_to_remove_set))
    # list_to_remove2 = list(set(list_to_remove2))
    # for i in range(0, 10):
    #     print(list_to_remove[i])
    # print(len(list_to_remove))
    # print(list_to_remove)
    # print(len(list_to_remove2))
    sys.exit()

    list_to_remove = list(set(list_to_remove))
    with open('removed_duplicate_fragments.txt', 'w') as write:
        for i in list_to_remove:
            write.write(os.path.join(i[1], i[0]))
            os.system('rm %s' % os.path.join(i[1], i[0]))

    # for i in list_to_remove:
    #     pass


if __name__ == '__main__':
    main()
