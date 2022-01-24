from src.data_preprocessing.Converter import Hd5Converter
from src.data_preprocessing.file_parsers import DeepDTADataParser, BindingDBParser
from src.data_preprocessing.Encoder import SmilesEncoder

import configparser
import os
import csv

excluded_compounds = []

# move to compound parser?
def encode_all_smiles(compound_parser, compound_list, tmp_filepath='tmp.csv'):
    smiles_encoder = SmilesEncoder(compound_parser.dictionary)
    with open(tmp_filepath, 'w', newline='') as tmp_file:

        for compound in compound_list:
            try:
                encoded_smiles = smiles_encoder.get_encoded_smiles(compound)[0]
                encoded_smiles.insert(0, compound)
                wr = csv.writer(tmp_file)
                wr.writerow(encoded_smiles)
            except KeyError:
                print('%s could not be embedded' % compound)
                excluded_compounds.append(compound)
                pass


def cleanup(file_path_tmp_file='tmp.csv'):
    os.remove(file_path_tmp_file)


def preprocess(data_type, affinity_matrix_file_path=None, compound_list_file_path=None, protein_list_file_path=None,
               compound_dictionary_file_path=None, protein_dictionary_file_path=None, binding_db_tsv_file_path=None,
               binding_db_fasta_file_path=None, output_file_affinity_matrix=None, output_file_smiles_embeddings=None):
    """
    :param data_type: one of "bdb", "kiba" or "davis"
    :type data_type: String
    :param affinity_matrix_file_path: path to the file containing the affinity matrix. The file mustn't contain any row
    or column labels. Each row must contain affinity values for all proteins in the dataset. If the value is unknown,
    the corresponding entry should be nan. The file is expected to contain a pickled numpy array.
    :type affinity_matrix_file_path: File path
    :param compound_list_file_path: path to the file containing the compound names corresponding to the affinity matrix
    in the order they appear in the matrix
    :type compound_list_file_path: File path
    :param protein_list_file_path: path to file containing the protein names/ identifiers corresponding to the affinity
    matrix in the order they appear in the matrix
    :type protein_list_file_path: File path
    :param compound_dictionary_file_path: path to file that contains a dictionary with compound names and SMILES
    representation. Compound names must be identical to the ones used in the compound_list_file_path.
    :type compound_dictionary_file_path: File path
    :param protein_dictionary_file_path: path to file that contains a dictionary with protein names/ identifiers and
    protein sequence. Protein names must be identical to the ones used in the protein_list_file_path.
    :type binding_db_tsv_file_path: File path
    :param binding_db_tsv_file_path: path to tab separated values (.tsv) file from BindingDB. It must at least contain
    the following columns: 'Ligand SMILES', 'Target Name Assigned by Curator or DataSource' and one binding strength
    measure ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)', 'kon (M-1-s-1)' or 'koff (s-1)']
    :type binding_db_fasta_file_path: File path
    :param binding_db_fasta_file_path: path to the fasta file that contains the amino acid sequence for the BindingDB
    proteins. The headers are expected to have the following format:
    >[protein identifier] mol:[protein|na] length:[sequence length] [description]
    example:
    >p9671 mol:protein length:487 Serine/threonine-protein kinase 4 (STK4)
    :type output_file_affinity_matrix: File path
    :param output_file_affinity_matrix: path and filename of the affinity matrix output file
    :type output_file_smiles_embeddings: File path
    :param output_file_smiles_embeddings: path and filename of the SMILES embedding output file

    Take input files and preprocess them.
    Preprocessing steps include removal of irrelevant proteins and compounds, collecting data for binding affinity
    matrix, encoding of compounds into SMILES embeddings, writing SMILES embeddings as hd5 files, convert affinity
    matrix values to kd values if necessary (BindingDB and Davis dataset) and writing the affinity matrix to a new file.

    """
    kd_flag = False
    data_type = data_type.lower()

    if data_type not in ["bdb", "kiba", "davis"]:
        print("Invalid data type %s" % data_type)
        return
    if data_type in ["bdb", "davis"]:
        kd_flag = True

    if data_type in ['davis', 'kiba']:
        if not affinity_matrix_file_path or not compound_list_file_path or not protein_list_file_path or\
                not compound_dictionary_file_path or not protein_dictionary_file_path:
            print('Affinity matrix, compound list, protein list, compound dictionary or protein dictionary file path '
                  'missing')
            return
        affinity_matrix_parser = DeepDTADataParser.AffinityMatrixFileParser(affinity_matrix_file_path,
                                                                            protein_list_file_path,
                                                                            compound_list_file_path)
        protein_parser = DeepDTADataParser.ProteinParser(protein_dictionary_file_path) #do I even need the protein parser already?
        compound_parser = DeepDTADataParser.CompoundParser(compound_dictionary_file_path)

    else:
        if not binding_db_fasta_file_path or not binding_db_tsv_file_path:
            print('Binding DB tsv or faster file path missing')
            return
        tsv_parser = BindingDBParser.BindingDBTsvParser(binding_db_tsv_file_path, binding_db_fasta_file_path)
        affinity_matrix_parser = tsv_parser.interactions_parser
        protein_parser = tsv_parser.protein_parser
        compound_parser = tsv_parser.compound_parser

    print('Creating embeddings for SMILES representations of compounds')
    encode_all_smiles(compound_parser, affinity_matrix_parser.get_compounds_list())

    hd5_converter = Hd5Converter(output_path=output_file_smiles_embeddings)
    hd5_converter.convert()

    if kd_flag:
        affinity_matrix_parser.convert_kd_matrix_to_pkd()

    affinity_matrix_parser.save_affinity_matrix_to_file(output_file_affinity_matrix,
                                                        excluded_compounds=excluded_compounds)
    cleanup()


config = configparser.ConfigParser()
config.read("config.ini")
data_type = config["GENERAL"]["datatype"].lower()
if data_type == "bdb":
    preprocess(data_type,
               binding_db_tsv_file_path=config["INPUT FILES BDB"]["binding_db_tsv"],
               binding_db_fasta_file_path=config["INPUT FILES BDB"]["fasta"],
               output_file_affinity_matrix=config["OUTPUT FILES"]["affinity_matrix"],
               output_file_smiles_embeddings=config["OUTPUT FILES"]["smiles_embeddings"]
               )

if data_type == "davis" or data_type == "kiba":
    preprocess(data_type,
               affinity_matrix_file_path=config["INPUT FILES DEEPDTA DATASETS"]["affinity_matrix"],
               compound_list_file_path=config["INPUT FILES DEEPDTA DATASETS"]["compound_ids"],
               protein_list_file_path=config["INPUT FILES DEEPDTA DATASETS"]["protein_ids"],
               compound_dictionary_file_path=config["INPUT FILES DEEPDTA DATASETS"]["compound_dictionary"],
               protein_dictionary_file_path=config["INPUT FILES DEEPDTA DATASETS"]["protein_dictionary"],
               output_file_affinity_matrix=config["OUTPUT FILES"]["affinity_matrix"],
               output_file_smiles_embeddings=config["OUTPUT FILES"]["smiles_embeddings"]
               )
