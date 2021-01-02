import os
from Bio.PDB import PDBParser, PDBIO, Superimposer
import FragUtils as Frag


# Globals
module = 'Cluster All to All RMSD of Individual Fragments:'


def main(frag_db_dir, aligned_clusters_outdir, rmsd_file_path):
    print('%s Beginning' % module)

    return_clusters = Frag.cluster_fragment_rmsds(rmsd_file_path)
    print('%s Clustering Finished, Creating Representatives' % module)

    # Align all Cluster Members to Cluster Representative
    cluster_count = 1
    for cluster in return_clusters:
        if len(cluster[1]) >= 10:
            cluster_rep = cluster[0]
            cluster_rep_pdb_path = os.path.join(frag_db_dir, cluster[0] + '.pdb')

            parser = PDBParser()
            cluster_rep_biopdb = parser.get_structure(cluster_rep, cluster_rep_pdb_path)

            cluster_outdir = os.path.join(aligned_clusters_outdir, str(cluster_count))
            if not os.path.exists(cluster_outdir):
                os.makedirs(cluster_outdir)
            Frag.center(cluster_rep_biopdb)

            io1 = PDBIO()
            io1.set_structure(cluster_rep_biopdb)
            io1.save(os.path.join(cluster_outdir, cluster_rep + '_representative.pdb'))

            for structure_idx in cluster[1]:
                structure_path = os.path.join(frag_db_dir, structure_idx + '.pdb')

                parser = PDBParser()
                idx_biopdb = parser.get_structure(structure_idx, structure_path)

                sup = Superimposer()
                sup.set_atoms(Frag.get_biopdb_ca(cluster_rep_biopdb), Frag.get_biopdb_ca(idx_biopdb))
                sup.apply(idx_biopdb)

                io2 = PDBIO()
                io2.set_structure(idx_biopdb)
                io2.save(os.path.join(cluster_outdir, structure_idx + '_aligned.pdb'))
            cluster_count += 1

    print('%s Finished' % module)


if __name__ == '__main__':
    # Fragment DB Directory
    indv_frag_outdir = os.path.join(os.getcwd(), 'all_individual_frags')

    # Outdir
    i_clusters_outdir = os.path.join(os.getcwd(), 'i_clusters')
    if not os.path.exists(i_clusters_outdir):
        os.makedirs(i_clusters_outdir)

    # Get All to All RMSD File
    i_rmsd_outfile_path = os.path.join(indv_frag_outdir, 'all_to_all_rmsd.txt')
    main(indv_frag_outdir, i_clusters_outdir, i_rmsd_outfile_path)
