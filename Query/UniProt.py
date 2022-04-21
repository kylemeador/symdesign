from SymDesignUtils import start_log
from Query.utils import connection_exception_handler


# Globals
logger = start_log(name=__name__)
example = 'https://rest.uniprot.org/uniprotkb/Q9HIA7.json'


def query_uniprot(uniprot_id):
    """
    {'entryType': 'UniProtKB unreviewed (TrEMBL)', 'primaryAccession': 'Q9HIA7',
    'uniProtkbId': 'Q9HIA7_THEAC', '
    entryAudit': {'firstPublicDate': '2001-03-01', 'lastAnnotationUpdateDate': '2021-06-02', 'lastSequenceUpdateDate': '2001-03-01', 'entryVersion': 97, 'sequenceVersion': 1},
    'annotationScore': 15.100000000000001,
    'organism': {'scientificName':
                 'Thermoplasma acidophilum (strain ATCC 25905 / DSM 1728 / JCM 9062 / NBRC 15155 / AMRC-C165)',
    'taxonId': 273075,
    'evidences': [{'evidenceCode': 'ECO:0000313', 'source': 'Proteomes', 'id': 'UP000001024'}],
    'lineage': ['Archaea', 'Candidatus Thermoplasmatota', 'Thermoplasmata', 'Thermoplasmatales', 'Thermoplasmataceae', 'Thermoplasma']},
    'proteinExistence': '1: Evidence at protein level',
    'proteinDescription': {'recommendedName': {'fullName': {'evidences': [{'evidenceCode': 'ECO:0000259', 'source': 'Pfam', 'id': 'PF01923'}], 'value': 'Cob_adeno_trans domain-containing protein'}}},
    'genes': [{'orderedLocusNames': [{'evidences': [{'evidenceCode': 'ECO:0000313', 'source': 'EMBL', 'id': 'CAC12554.1'}], 'value': 'Ta1434'}]}], 'comments': [{'texts': [{'evidences': [{'evidenceCode': 'ECO:0000256', 'source': 'ARBA', 'id': 'ARBA00007487'}], 'value': 'Belongs to the Cob(I)alamin adenosyltransferase family'}], 'commentType': 'SIMILARITY'}],
    'features': [{'type': 'Domain', 'location': {'start': {'value': 2, 'modifier': 'EXACT'}, 'end': {'value': 158, 'modifier': 'EXACT'}}, 'description': 'Cob_adeno_trans', 'evidences': [{'evidenceCode': 'ECO:0000259', 'source': 'Pfam', 'id': 'PF01923'}]}], 'keywords': [{'evidences': [{'evidenceCode': 'ECO:0007829', 'source': 'PDB', 'id': '1NOG'}], 'id': 'KW-0002', 'category': 'Technical term', 'name': '3D-structure'}, {'evidences': [{'evidenceCode': 'ECO:0000256', 'source': 'ARBA', 'id': 'ARBA00022840'}], 'id': 'KW-0067', 'category': 'Ligand', 'name': 'ATP-binding'}, {'evidences': [{'evidenceCode': 'ECO:0000256', 'source': 'ARBA', 'id': 'ARBA00022741'}], 'id': 'KW-0547', 'category': 'Ligand', 'name': 'Nucleotide-binding'}, {'evidences': [{'evidenceCode': 'ECO:0000313', 'source': 'Proteomes', 'id': 'UP000001024'}], 'id': 'KW-1185', 'category': 'Technical term', 'name': 'Reference proteome'}, {'evidences': [{'evidenceCode': 'ECO:0000256', 'source': 'ARBA', 'id': 'ARBA00022679'}], 'id': 'KW-0808', 'category': 'Molecular function', 'name': 'Transferase'}], 'references': [{'citation': {'id': '11029001', 'citationType': 'journal article', 'authors': ['Ruepp A.', 'Graml W.', 'Santos-Martinez M.L.', 'Koretke K.K.', 'Volker C.', 'Mewes H.W.', 'Frishman D.', 'Stocker S.', 'Lupas A.N.', 'Baumeister W.'], 'citationCrossReferences': [{'database': 'PubMed', 'id': '11029001'}, {'database': 'DOI', 'id': '10.1038/35035069'}], 'title': 'The genome sequence of the thermoacidophilic scavenger Thermoplasma acidophilum.', 'publicationDate': '2000', 'journal': 'Nature', 'firstPage': '508', 'lastPage': '513', 'volume': '407'}, 'referencePositions': ['NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA]'], 'referenceComments': [{'evidences': [{'evidenceCode': 'ECO:0000313', 'source': 'Proteomes', 'id': 'UP000001024'}], 'value': 'ATCC 25905 / DSM 1728 / JCM 9062 / NBRC 15155 / AMRC-C165', 'type': 'STRAIN'}], 'evidences': [{'evidenceCode': 'ECO:0000313', 'source': 'EMBL', 'id': 'CAC12554.1'}, {'evidenceCode': 'ECO:0000313', 'source': 'Proteomes', 'id': 'UP000001024'}]}, {'citation': {'id': '15044458', 'citationType': 'journal article', 'authors': ['Saridakis V.', 'Yakunin A.', 'Xu X.', 'Anandakumar P.', 'Pennycooke M.', 'Gu J.', 'Cheung F.', 'Lew J.M.', 'Sanishvili R.', 'Joachimiak A.', 'Arrowsmith C.H.', 'Christendat D.', 'Edwards A.M.'], 'citationCrossReferences': [{'database': 'PubMed', 'id': '15044458'}, {'database': 'DOI', 'id': '10.1074/jbc.M401395200'}], 'title': 'The structural basis for methylmalonic aciduria. The crystal structure of archaeal ATP:cobalamin adenosyltransferase.', 'publicationDate': '2004', 'journal': 'J. Biol. Chem.', 'firstPage': '23646', 'lastPage': '23653', 'volume': '279'}, 'referencePositions': ['X-RAY CRYSTALLOGRAPHY (1.55 ANGSTROMS)'], 'evidences': [{'evidenceCode': 'ECO:0007829', 'source': 'PDB', 'id': '1NOG'}]}], 'uniProtKBCrossReferences': [{'database': 'EMBL', 'id': 'AL445067', 'properties': [{'key': 'ProteinId', 'value': 'CAC12554.1'}, {'key': 'Status', 'value': '-'}, {'key': 'MoleculeType', 'value': 'Genomic_DNA'}]}, {'database': 'RefSeq', 'id': 'WP_010901837.1', 'properties': [{'key': 'NucleotideSequenceId', 'value': 'NC_002578.1'}]}, {'database': 'PDB', 'id': '1NOG', 'properties': [{'key': 'Method', 'value': 'X-ray'}, {'key': 'Resolution', 'value': '1.55 A'}, {'key': 'Chains', 'value': 'A=1-177'}]}, {'database': 'PDBsum', 'id': '1NOG', 'properties': [{'key': 'Description', 'value': '-'}]}, {'database': 'SMR', 'id': 'Q9HIA7', 'properties': [{'key': 'Description', 'value': '-'}]}, {'database': 'STRING', 'id': '273075.Ta1434', 'properties': [{'key': 'Description', 'value': '-'}]}, {'database': 'DNASU', 'id': '1456890', 'properties': [{'key': 'Description', 'value': '-'}]}, {'database': 'EnsemblBacteria', 'id': 'CAC12554', 'properties': [{'key': 'ProteinId', 'value': 'CAC12554'}, {'key': 'GeneId', 'value': 'CAC12554'}]}, {'database': 'GeneID', 'id': '1456890', 'properties': [{'key': 'Description', 'value': '-'}]}, {'database': 'KEGG', 'id': 'tac:Ta1434', 'properties': [{'key': 'Description', 'value': '-'}]}, {'database': 'eggNOG', 'id': 'arCOG00489', 'properties': [{'key': 'ToxonomicScope', 'value': 'Archaea'}]}, {'database': 'HOGENOM', 'id': 'CLU_083486_0_1_2', 'properties': [{'key': 'Description', 'value': '-'}]}, {'database': 'OMA', 'id': 'HQACTVV', 'properties': [{'key': 'Fingerprint', 'value': '-'}]}, {'database': 'OrthoDB', 'id': '98914at2157', 'properties': [{'key': 'Description', 'value': '-'}]}, {'database': 'BRENDA', 'id': '2.5.1.17', 'properties': [{'key': 'OrganismId', 'value': '6324'}]}, {'database': 'EvolutionaryTrace', 'id': 'Q9HIA7', 'properties': [{'key': 'Description', 'value': '-'}]}, {'database': 'Proteomes', 'id': 'UP000001024', 'properties': [{'key': 'Component', 'value': 'Chromosome'}]}, {'database': 'GO', 'id': 'GO:0005524', 'properties': [{'key': 'GoTerm', 'value': 'F:ATP binding'}, {'key': 'GoEvidenceType', 'value': 'IEA:UniProtKB-KW'}]}, {'database': 'GO', 'id': 'GO:0016740', 'properties': [{'key': 'GoTerm', 'value': 'F:transferase activity'}, {'key': 'GoEvidenceType', 'value': 'IEA:UniProtKB-KW'}]}, {'database': 'Gene3D', 'id': '1.20.1200.10', 'properties': [{'key': 'EntryName', 'value': '-'}, {'key': 'MatchStatus', 'value': '1'}]}, {'database': 'InterPro', 'id': 'IPR016030', 'properties': [{'key': 'EntryName', 'value': 'CblAdoTrfase-like'}]}, {'database': 'InterPro', 'id': 'IPR036451', 'properties': [{'key': 'EntryName', 'value': 'CblAdoTrfase-like_sf'}]}, {'database': 'InterPro', 'id': 'IPR029499', 'properties': [{'key': 'EntryName', 'value': 'PduO-typ'}]}, {'database': 'PANTHER', 'id': 'PTHR12213', 'properties': [{'key': 'EntryName', 'value': 'PTHR12213'}, {'key': 'MatchStatus', 'value': '1'}]}, {'database': 'Pfam', 'id': 'PF01923', 'properties': [{'key': 'EntryName', 'value': 'Cob_adeno_trans'}, {'key': 'MatchStatus', 'value': '1'}]}, {'database': 'SUPFAM', 'id': 'SSF89028', 'properties': [{'key': 'EntryName', 'value': 'SSF89028'}, {'key': 'MatchStatus', 'value': '1'}]}, {'database': 'TIGRFAMs', 'id': 'TIGR00636', 'properties': [{'key': 'EntryName', 'value': 'PduO_Nterm'}, {'key': 'MatchStatus', 'value': '1'}]}], 'sequence': {'value': 'MFTRRGDQGETDLANRARVGKDSPVVEVQGTIDELNSFIGYALVLSRWDDIRNDLFRIQNDLFVLGEDVSTGGKGRTVTREMIDYLEARVKEMKAEIGKIELFVVPGGSIESASLHMARAVSRRLERRIVAASKLTEINKNVLIYANRLSSILFMHALLSNKRLNIPEKIWSIHRVS', 'length': 177, 'molWeight': 20013, 'crc64': '13D3B46CB3ED92F6', 'md5': '5D8AFDAE2BFCB93431C348E2A0172D96'}, 'extraAttributes': {'countByCommentType': {'SIMILARITY': 1}, 'countByFeatureType': {'Domain': 1}, 'uniParcId': 'UPI000006403F'}}
    Args:
        uniprot_id:

    Returns:

    """
    if uniprot_id and len(uniprot_id) in [6, 10]:
        query_url = 'https://rest.uniprot.org/uniprotkb/%s.json' % uniprot_id
        return connection_exception_handler(query_url)
    else:
        logger.warning('UniProt ID "%s" is not of the required format and will not be found with the UniProt API'
                       % uniprot_id)
        return


def is_uniprot_thermophilic(uniprot_id):
    uniprot_json = query_uniprot(uniprot_id)
    lineage = uniprot_json.get('lineage', [])
    for element in lineage:
        if 'thermo' in element.lower():
            return True

    return False
