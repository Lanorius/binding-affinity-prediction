import numpy as np
import pandas as pd
from silx.io.dictdump import h5todict
import torch
import random  # for shuffled labels


class Dataset:
    def __init__(self, embeddings, compound_vectors, label_file, data_type, cluster_map=None, shuffle_drugs=False,
                 shuffle_targets=False):

        """
        :param embeddings: path to reduced embedding file in h5 format
        :type embeddings: str
        :param compound_vectors: path to compound representation file in h5 format
        :type compound_vectors: str
        :param label_file: path to CSV file for interaction strength between proteins and compounds, with header.
        Columns are labeled with protein names, rows with compound names. Known interactions are indicated by float
        values.
        :type label_file: str
        :param data_type: datatype. Valid datatypes are "pkd" and "kiba"
        :type data_type: str
        :param cluster_map: path to cluster map file in h5 format.
        :type cluster_map: str, optional
        """

        # this dictionary is important for each new type of values that are being evaluated
        self.data_ranges = {"pkd": [5, 11], "kiba": [0, 18]}
        self.data_type = data_type

        self.drugs_representation = h5todict(compound_vectors)
        self.targets_embedded = h5todict(embeddings)

        if shuffle_drugs:
            self.drugs_representation = dict(zip(self.drugs_representation.keys(),
                                                 random.sample(list(self.drugs_representation.values()),
                                                               len(self.drugs_representation))))
        if shuffle_targets:
            self.targets_embedded = dict(zip(self.targets_embedded.keys(),
                                             random.sample(list(self.targets_embedded.values()),
                                                           len(self.targets_embedded))))

        self.data_to_load = []
        self.regression_labels = []
        self.binary_labels = []

        self._determine_labels_and_data_to_load_keys(label_file, cluster_map)

    def __getitem__(self, index):

        key_target, key_comp = self.data_to_load[index]
        target_comp = [torch.from_numpy(self.targets_embedded[key_target]).float(),
                       torch.from_numpy(self.drugs_representation[key_comp])]

        return target_comp, self.regression_labels[index], self.binary_labels[index], (key_target, key_comp)

    def __len__(self):
        return len(self.data_to_load)

    def _determine_labels_and_data_to_load_keys(self, label_file, cluster_map):
        binary_thresholds = {"pkd": 7, "kiba": 12.1}  # same thresholds as chosen by DeepDTA paper

        if cluster_map is not None:
            cluster_map = h5todict(cluster_map)

        interactions = pd.read_csv(label_file)
        # interactions = pd.read_csv(label_file, header=0, index_col=0, sep='\t')
        targeteins = interactions.columns.tolist()[1:]  # ignore first column name
        compounds = [str(i) for i in interactions[interactions.columns[0]].tolist()]
        interactions = interactions.drop(interactions.columns[0], axis=1)
        # why not keep it as array?
        labels = np.hstack(interactions.values)
        label_index = -1

        # get keys for compound-targetein combinations and write to list; get labels
        # compound-targetein key list avoids creating NxMx(k+l) matrix in memory for N targeteins(with embedding size k)
        # and M compounds (with embedding size l).
        for comp in compounds:
            for target in targeteins:
                target = target.replace('>', '')
                label_index += 1
                if target in self.targets_embedded and comp in self.drugs_representation:
                    if (labels[label_index] > 0) & np.isfinite(labels[label_index]):
                        if labels[label_index] < self.data_ranges[self.data_type][1]:
                            if cluster_map is None:
                                self.data_to_load.append([target.replace('>', ''), comp])
                            else:
                                self.data_to_load.append([cluster_map[target].item().decode("utf-8"), comp])
                            binding_strength = max(labels[label_index], self.data_ranges[self.data_type][0])
                            self.regression_labels.append(binding_strength)
                            self.binary_labels.append(0.0 if binding_strength < binary_thresholds[self.data_type]
                                                      else 1.0)
