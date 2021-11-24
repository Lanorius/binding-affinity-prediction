import configparser
import sys


def parse_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    files = config["INPUT FILES"]
    general_setting = config["GENERAL"]
    prediction_types = config["PREDICTION TYPES"]
    do_regression = prediction_types.getboolean("regression")
    nr_of_classes = prediction_types.getint("nr_classes")

    # when using data_loading_general you have to pass the files
    # (embeddings, compound_vectors, labels, mapping_file) as arguments
    if general_setting['database'] == 'davis':
        # for the Davis Data
        data_used = ["pkd", "Davis"]

    elif general_setting['database'] == 'kiba':
        # for the Kiba Data
        data_used = ["kiba", "Kiba"]

    elif general_setting['database'] == 'bdb':
        # for the BDB pkd Data
        data_used = ["pkd", "BDB Data pKd"]
    else:
        print("Invalid entry in config.ini for dataset in section GENERAL")
        sys.exit()

    if general_setting['compounds'] == 'chemVAE':
        # for the chemVAE vector
        use_model = "chemVAE"

    elif general_setting['compounds'] == 'chemBERTa':
        # for the chemBERTa vector
        use_model = "chemBERTa"
    elif general_setting['compounds'] == 'RDKit':
        # for the chemBERTa vector
        use_model = "RDKit"    
    

    else:
        print("Invalid entry in config.ini for compounds in section GENERAL")
        sys.exit()

    return data_used, use_model, files, do_regression, nr_of_classes

