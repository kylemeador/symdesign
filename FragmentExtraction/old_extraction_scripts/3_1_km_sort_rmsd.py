import sys

all_to_all_rmsd_filepath = frag_db_dir = os.path.join(os.getcwd(), 'all_individual_frags') + '/20000_all_to_all_rmsd.txt'
f = open(all_to_all_rmsd_filepath, "r")
flines = f.readlines()
f.close()

all_to_all_rmsd_list = []
for line in flines:
    line = line.split()
    all_to_all_rmsd_list.append((line[0], line[1], float(line[2])))

for comp in sorted(all_to_all_rmsd_list, key=lambda tup: tup[2]):
    print comp
