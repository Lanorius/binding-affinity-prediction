import pandas as pd
import numpy as np
import hdfdict


class Hd5Converter:
    """Used to transform a vector csv and a component index to id csv int  o one hd5 file"""
    def __init__(self, embedding_file_path='tmp.csv',  output_path=None):
        """
            :param inner_vector_file_path: path to inner vector csv file
            :type inner_vector_file_path: Path Object
            :param comp_index_to_id_filepath: path to compound index to id csv file
            :type comp_index_to_id_filepath: Path Object
            :param output_path: path for output file
            :type comp_index_to_id_filepath: Path Object or None
        """

        self.embedding_csv = embedding_file_path
        self.output_path = output_path

    def convert(self):
        df = pd.read_csv(self.embedding_csv, header=None)
        cids = df.iloc[:, 0]
        vectors = df.iloc[:, 1:]

        cids = cids.tolist()
        vectors = vectors.values.tolist()
        hd5_dictionary = {}

        for i in range(len(cids)):
            hd5_dictionary[str(cids[i])] = [np.float32(val) for val in vectors[i]]

        if self.output_path:
            hdfdict.dump(hd5_dictionary, self.output_path)

        return hd5_dictionary
