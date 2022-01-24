from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu
import os


class SmilesEncoder:
    def __init__(self, smiles_dictionary, vea_model_path=os.path.join('..', '..', 'weights', 'zinc_properties')):
        """
            :param smiles_dictionary: a dictionary containing the smiles identifier and the smiles representation
            of the compound
            :type smiles_dictionary: dictionary
            :param vea_model_path: path to the model that should be used for encoding the smiles string, as a default
            the model in models/zinc_properties is used
            :type file_path: Path Object, optional
        """
        self.encoded_smiles = {}
        self.vae = VAEUtils(directory=vea_model_path)
        self._encode_smiles(smiles_dictionary)

    def _encode_smiles(self, smiles_dictionary):
        error_counter = 0
        for key in smiles_dictionary.keys():
            smiles = smiles_dictionary[key]
            try:
                canon_smiles = mu.canon_smiles(smiles.replace("'", ""))
                hot_encoded_smiles = self.vae.smiles_to_hot(canon_smiles)

                # check if smiles was to long (<120) to hot encode
                if len(hot_encoded_smiles) == 0:
                    print("Couldn't encode the smile for %s because the smiles representation is to long" % key)
                    error_counter += 1
                else:
                    encoded_smiles = self.vae.encode(hot_encoded_smiles)
                    self.encoded_smiles[key] = encoded_smiles.tolist()

            except KeyError:
                print("Couldn't encode the smile for %s due to unexpected character in smiles" % key)
                error_counter += 1

            except Exception:
                print("Encountered unknown error for %s" % key)
                error_counter += 1
        print("%d of %d compounds couldn't be encoded" % (error_counter, len(smiles_dictionary.keys())))

    def get_encoded_smiles(self, compound_id):
        """
        :param compound_id: id of the compound for which the encoded SMILES representation should be returned
        :return: SMILES encoding as a list
        """
        return self.encoded_smiles[compound_id]

    def get_encoded_smiles_ids(self):
        """
        :return: list of ids for which encoding was successful
        """
        return self.encoded_smiles.keys()
