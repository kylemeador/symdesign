import sys

from Align import align
from tools.ExpandASU import expand_asu

# sys.argv = list
# python rogers_magic.py file darpin start2 end2 chain2
sys.argv = ['../AlphaHelix-newcenter.pdb', '../DARP_GFP.pdb', 164, 168, 'Z']

file = sys.argv[0]  # AlphaHelix
aligned_file = sys.argv[1]  # TargetProtein
start2 = sys.argv[2]
end2 = sys.argv[3]
chain2 = sys.argv[4]


for i in range(15):
    start = 1 + i
    end = 5 + i
    chain = "N"
    out = align(file, start, end, chain, aligned_file, start2, end2, chain2)
#    os.mkdir("../Expandedfiles/config"+str(i))
    newfile = "../Expandedfiles/aligned"+str(i)+".pdb"
    out.write(newfile)
    expand_asu(newfile, "T", out_path="../Expandedfiles/")
