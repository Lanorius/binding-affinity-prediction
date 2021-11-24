# Prediction-Of-Binding-Affinity

## Table of Contents

* [Description](#Description)
* [Setup](#Setup)
	* [BindingDB](#BindingDB)
* [Usage](#usage)
	* [Preprocess Data](#Preprocess-Data)
	* [Training a model](#Training-a-model)
		* [config.ini](#config.ini)
	* [Load pre-trained model and load new data](#Load-pre-trained-model-and-load-new-data)
* [Links](#Links)
	* [Project related presentations and publications](#Project-related-presentations-and-publications)
	* [Related resources](#Related-resources)
		* [Embeddings](#Embeddings)
		* [Databases](#Databases)
		* [Related Work](#Related-work)
* [ToDOs](#ToDOs)
* [Future ideas](#Future-ideas)

## Description

## Setup

### BindingDB

To use the preprocessing for BindingDB, please unpack "BindingDB_All.7z" in the sub "folder data/bdb"

## Usage


### Preprocess Data

To pre-process the data from the "data" sub folder the scripts in "src/data_preprocessing" can be used by calling:

```
python preprocess_files_main.py
```

The settings can be configured in the config.ini in the data_preprocessing sub folder.


### Training a model

To train a new model, according to the set configurations, the following command can be used in the src folder:
```
binding_prediction.py
```

#### config.ini

This file is for adjusting the settings that should be used for training a new model.

### Load pre-trained model and load new data

TODO - currently doesn't exist

## Links

### Project related presentations and publications

[Masterpraktikum 2020 - Affinity Prediction Slides](https://docs.google.com/presentation/d/1-VV0Z7pT1VxaWNwn9BNw8AqYKmx7lbMDIxM0r0M-G7s/edit#slide=id.ga0d2a34b63_0_9)

### Related resources

#### Embeddings

[ChemVea](https://github.com/aspuru-guzik-group/chemical_vae): Variational auto encoder that was used to create compound embeddings

[Bio Embeddings](https://github.com/sacdallago/bio_embeddings): Resource for creating protein sequence embeddings

#### Databases

[BindingDB](https://www.bindingdb.org/bind/index.jsp): Database of measured binding affinity

#### Related Work

[DeepDTA Paper](https://arxiv.org/pdf/1801.10193.pdf): Drug-target binding affinity prediction method from 2018 based on sequence information only 

[ChemBoost](https://onlinelibrary.wiley.com/doi/abs/10.1002/minf.202000212): Drug-target binding affinity method from 2020 from the same group as DeepDTA, testing different input features for compounds and proteins

[SimBoost](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z)

[Toward more realistic drug–target interaction predictions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4364066/)

## ToDOs

* automatically unpack bindingDB_All.7z?
* more documentation (doc strings, description of config file parameters, ...)
* create requirements file

## Future ideas

* investigate binned predictions more
  * remap predictions to label values
  * std error and other performance metrics for binary predictions
  * make binary predictions work with existing evaluation code
  * ...
* LSTM/RNN/Attention etc for affinity prediction?
* add other embedding preprocessing options? (Compounds and/or Proteins)
* more visualizations
  * training vs validation curve?
  * more visualizations to identify difficult pairs?
* multi threaded cross-training?