import pickle
import numpy as np
import pandas as pd
from Bio import SeqIO
import re


def _insert_index_file_content_into_dictionary(dictionary, content_type, file_path):
    with open(file_path, 'r') as fh:
        lines = fh.readlines()
        counter = 0
        for line in lines:
            line = line.strip()
            if line != '':
                dictionary[content_type][line] = counter
                counter += 1
    return dictionary


def _select_interaction_type(interaction_type):
    interaction_type = interaction_type.lower()
    possible_interaction_types = ['ki', 'ic50', 'kd', 'ec50', 'kon', 'koff']
    if interaction_type not in possible_interaction_types:
        print('Invalid interaction type %s was selected. Interaction type was set to the default "Ki" instead' %
              interaction_type)
        interaction_type = 'ki'
    interaction_types = {
        'ki': 'Ki (nM)',
        'ic50': 'IC50 (nM)',
        'kd': 'Kd (nM)',
        'ec50': 'EC50 (nM)',
        'kon': 'kon (M-1-s-1)',
        'koff': 'koff (s-1)'

    }
    return interaction_types[interaction_type]


class GenericProteinParser:
    def __init__(self):
        self.dictionary = {}

    def print_fasta(self, outfile):
        """
        :param outfile: file path and file name for the .fasta output file
        :type outfile: path object

        creates a fasta file for all proteins
        """
        with open(outfile, 'w') as fh:
            for protein in self.dictionary:
                header = '>%s\n' % protein
                sequence = self.dictionary[protein] + '\n'
                fh.write(header)
                fh.write(sequence)

    def get_sequence(self, protein_id):
        """
        :param protein_id: name of the protein, for which the sequence should be obtained. Names are case sensitive.
        :type protein_id: string
        :return: amino acid sequence of protein as string or KeyError

        Obtains the sequence corresponding to the protein_id or a KeyError if the protein_id can't be found
        """
        return self.dictionary[protein_id]


class GenericAffinityParser:
    def __init__(self):
        """
        Initializes an empty affinity matrix and an empty dictionary for compound and protein indexes in affinity matrix
        """
        self.affinity_matrix = None
        self._indexes = {
            'proteins': {},
            'compounds': {}
        }

    def convert_kd_matrix_to_pkd(self):
        """applies log transformation to affinity matrix to transform from kd to pkd values"""
        self.affinity_matrix = -np.log10(self.affinity_matrix / 1e9)

    def save_affinity_matrix_to_file(self, output_file_path, excluded_compounds=None):
        """
        :param output_file_path: file path (including file name) for the affinity matrix output file. The output file
        will be a comma seperated value (csv) file with the protein ids as column names and the compound names as row
        names. Unknown affinity values are indicated by "nan"
        :type output_file_path: path Object
        :param excluded_compounds: optional: list of compounds that should be ignored for the output file. None is used
        as default
        :type excluded_compounds: list of strings

        Saves affinity matrix to an output file
        """

        header = 'cids,' + ','.join(['>' + key[0] for key in sorted(self._indexes['proteins'].items(),
                                                                    key=lambda kv: (kv[1], kv[0]))])
        row_labels = [key[0] for key in sorted(self._indexes['compounds'].items(), key=lambda kv: (kv[1], kv[0]))]
        with open(output_file_path, 'w') as fh:
            fh.write(header + '\n')
            for count, row in enumerate(self.affinity_matrix):
                row_label = row_labels[count]
                if excluded_compounds and row_label in excluded_compounds:
                    continue
                else:
                    line = str(row_label) + ',' + ','.join(["%.10f" % number for number in row]) + '\n'
                    fh.write(line)

    def get_compounds_list(self):
        """
        :return: sorted list of compounds names in the affinity matrix
        """
        return [key[0] for key in sorted(self._indexes['compounds'].items(), key=lambda kv: (kv[1], kv[0]))]


class DictionaryParser:
    def __init__(self, file_path):
        """Generic parser to turn a dictionary, that is stored in a file into a python dictionary"""
        self.file_path = file_path
        self.dictionary = self._parse_dict()

    def _parse_dict(self):
        with open(self.file_path, 'r') as fh:
            dict_from_file = eval(fh.read())
        return dict_from_file


class BindingDBParser:
    class ProteinParser(GenericProteinParser):
        """Protein Parser for BindingDB"""
        def __init__(self, file_path_sequences):
            """
            :param file_path_sequences: file path to the .fasta file containing the protein sequences. Expected header
            format:
            >[protein identifier] mol:[protein|na] length:[sequence length] [description]
            example:
            >p9671 mol:protein length:487 Serine/threonine-protein kinase 4 (STK4)
            :type file_path_sequences: path object

            creates a dictionary containing the protein name and the corresponding sequence and a dictionary that maps
            protein descriptions to protein names
            """
            self.file_path = file_path_sequences
            self.target_name_to_protein_name = {}
            self.dictionary = self._parse_fasta()

        def _parse_fasta(self):
            return_dictionary = dict()
            fasta_sequences = SeqIO.parse(open(self.file_path), 'fasta')
            for fasta in fasta_sequences:
                name, sequence, description = fasta.id, str(fasta.seq), fasta.description
                return_dictionary[str(name)] = sequence
                target_name = re.split('length:\d+ ', description)[-1]
                self.target_name_to_protein_name[target_name] = str(name)
            return return_dictionary

        def print_fasta(self, outfile):
            """
            :param outfile: file path (including file name) for the output .fasta file
            :type outfile: path Object

            creates a fasta file for all proteins
            """
            super(BindingDBParser.ProteinParser, self).print_fasta(outfile)

        def get_sequence(self, protein_id):
            """
                :param protein_id: name of the protein, for which the sequence should be obtained. Names are case sensitive.
                :type protein_id: string
                :return: amino acid sequence of protein as string or KeyError

                Obtains the sequence corresponding to the protein_id or a KeyError if the protein_id can't be found
            """
            return super(BindingDBParser.ProteinParser, self).get_sequence(protein_id)

    class CompoundParser:
        def __init__(self, reduced_csv):
            self.dictionary = {}
            self.smiles_to_compound = {}
            self._process_reduced_csv(reduced_csv)

        def _process_reduced_csv(self, reduced_csv):
            compound_count = 0
            for index, row in reduced_csv.iterrows():
                smiles = row['Ligand SMILES']
                if smiles not in self.dictionary.values():
                    self.dictionary[compound_count] = smiles
                    self.smiles_to_compound[smiles] = compound_count
                    compound_count += 1

        def get_SMILES(self, compound):
            return self.dictionary[compound]

    class AffinityMatrixParser(GenericAffinityParser):
        def __init__(self, reduced_csv, protein_parser, compound_parser, interaction_type):
            super(BindingDBParser.AffinityMatrixParser, self).__init__()
            self._create_affinity_matrix(reduced_csv, protein_parser, compound_parser, interaction_type)

        def _create_affinity_matrix(self, reduced_csv, protein_parser, compound_parser, interaction_type):
            for index, protein in enumerate(protein_parser.dictionary):
                self._indexes['proteins'][protein] = index
            for index, compound in enumerate(compound_parser.dictionary):
                self._indexes['compounds'][compound] = index

            rows = [key[0] for key in sorted(self._indexes['compounds'].items(), key=lambda kv: (kv[1], kv[0]))]
            columns = [key[0] for key in sorted(self._indexes['proteins'].items(), key=lambda kv: (kv[1], kv[0]))]

            self.affinity_matrix = np.empty((len(rows), len(columns)))
            self.affinity_matrix[:] = np.NaN
            for _, entry in reduced_csv.iterrows():
                smiles = entry['Ligand SMILES']
                protein_description = entry['Target Name Assigned by Curator or DataSource']
                affinity = entry[interaction_type]

                protein_pos = self._indexes['proteins'][protein_parser.target_name_to_protein_name[protein_description]]
                smiles_pos = self._indexes['compounds'][compound_parser.smiles_to_compound[smiles]]

                self.affinity_matrix[smiles_pos][protein_pos] = affinity

        def get_binding_affinity(self, protein_name, compound_name):
            return self.affinity_matrix[self._indexes['compounds'][compound_name]][self._indexes['proteins'][
                protein_name]]

        def convert_kd_matrix_to_pkd(self):
            super(BindingDBParser.AffinityMatrixParser, self).convert_kd_matrix_to_pkd()

        def save_affinity_matrix_to_file(self, output_file_path, excluded_compounds=None):
            super(BindingDBParser.AffinityMatrixParser, self).save_affinity_matrix_to_file(output_file_path,
                                                                                           excluded_compounds=
                                                                                           excluded_compounds)

    class BindingDBTsvParser:
        def __init__(self, file_path_binding_db, file_path_sequences, interaction_type='ki'):
            """
                :param file_path_binding_db: path to tab separated file containing SMILES, binding strength and protein
                id
                :type file_path_binding_db: Path Object
                :param file_path_sequences: path to sequence file in fasta format
                :type file_path_sequences: Path Object
                :param interaction_type: String representing the desired interaction type. Valid values are: "ki",
                "ic50", "kd", "ec50", "kon", "koff". The default is "ki"
                :type interaction_type: String
            """
            self.binding_db_file = file_path_binding_db
            self.sequence_file = file_path_sequences
            self.interaction_type = _select_interaction_type(interaction_type)

            self.protein_parser = BindingDBParser.ProteinParser(file_path_sequences)

            relevant_data = self._parse_csv()

            self.compound_parser = BindingDBParser.CompoundParser(relevant_data)
            self.interactions_parser = BindingDBParser.AffinityMatrixParser(relevant_data, self.protein_parser,
                                                                            self.compound_parser, self.interaction_type)

            self._index_map = {}

        def _parse_csv(self, relevant_columns=None):
            if not relevant_columns:
                relevant_columns = [
                    'Ligand SMILES',
                    'Target Name Assigned by Curator or DataSource',
                    self.interaction_type
                ]
            relevant_data = pd.read_csv(self.binding_db_file, sep="\t", header=0, usecols=relevant_columns)

            # drop rows which are missing relevant information
            relevant_data.dropna(inplace=True)

            # drop rows that contain > or < sign in binding strength column
            relevant_data = relevant_data[~relevant_data[self.interaction_type].str.contains('>', na=False)]
            relevant_data = relevant_data[~relevant_data[self.interaction_type].str.contains('<', na=False)]

            # drop rows for which we don't have any sequences
            relevant_data = relevant_data[relevant_data['Target Name Assigned by Curator or DataSource'].isin(
                self.protein_parser.target_name_to_protein_name.keys())]

            return relevant_data


class DeepDTADataParser:
    class ProteinParser(DictionaryParser, GenericProteinParser):
        def __init__(self, file_path_sequences):
            super(DeepDTADataParser.ProteinParser, self).__init__(file_path_sequences)

        def print_fasta(self, outfile):
            super(DeepDTADataParser.ProteinParser, self).print_fasta(outfile)

        def get_sequence(self, protein_id):
            return super(DeepDTADataParser.ProteinParser, self).get_sequence(protein_id)

    class CompoundParser(DictionaryParser):
        def get_SMILES(self, compound):
            return self.dictionary[compound]

    # Maybe inherit from a general affinity matrix parser class? Kiba does not need kd->pkd conversion but bdb does
    class AffinityMatrixFileParser(GenericAffinityParser):
        def __init__(self, file_path_affinity_matrix, file_path_protein_names, file_path_compound_names):
            """
                :param file_path_affinity_matrix: path to the pickled affinity matrix file
                :type file_path_affinity_matrix: Path Object
                :param file_path_protein_names: path to protein names file in the order they appear in the affinity
                matrix
                :type file_path_protein_names: Path Object
                :param file_path_compound_names: path to compound names file in the order they appear in the affinity
                matrix
                :type file_path_compound_names: Path Object
            """
            super(DeepDTADataParser.AffinityMatrixFileParser, self).__init__()

            self.file_path_affinity_matrix = file_path_affinity_matrix
            self.file_path_protein_names = file_path_protein_names
            self.file_path_compound_names = file_path_compound_names

            self._parse_affinity_file()
            self._parse_index_files()

        def _parse_affinity_file(self):
            self.affinity_matrix = pickle.load(open(self.file_path_affinity_matrix, "rb"), encoding='latin1')

        def _parse_index_files(self):
            self._indexes = _insert_index_file_content_into_dictionary(self._indexes, 'compounds',
                                                                       self.file_path_compound_names)
            self._indexes = _insert_index_file_content_into_dictionary(self._indexes, 'proteins',
                                                                       self.file_path_protein_names)

        def get_binding_affinity(self, protein_name, compound_name):
            protein_index = self._indexes['proteins'][protein_name]
            compound_index = self._indexes['compounds'][compound_name]
            return self.affinity_matrix[compound_index][protein_index]

        def convert_kd_matrix_to_pkd(self):
            super(DeepDTADataParser.AffinityMatrixFileParser, self).convert_kd_matrix_to_pkd()

        def save_affinity_matrix_to_file(self, output_file_path, excluded_compounds=None):
            super(DeepDTADataParser.AffinityMatrixFileParser, self).save_affinity_matrix_to_file(output_file_path,
                                                                                                 excluded_compounds=
                                                                                                 excluded_compounds)
