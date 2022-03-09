# My Master's Thesis: Prediction-Of-Binding-Affinity

Our aim was to predict the binding affinity of compounds and proteins by using a fully connected neural network.\
To encode both types of molecules we used ChemVAE for the compounds and ProT5 for the proteins. 

![plot](./images/mymodel.png)

The following two repos were also part of my project. All three are required to repeat my steps:\
[Data Creator](https://github.com/Lanorius/dataset_creation): filtering and clustering of DTI dataset\
[ChemVAE Fork](https://github.com/Lanorius/chemical_vae): encoding of the small molecules

Additionally PortT5 are needed:\
[Prot T5] (https://github.com/agemagician/ProtTrans): encoding of the proteins


#### How to use our Prediction Model
1. Clone this repo
2. Ensure the requirements are met
3. Check in the src/config.ini file if all files are chosen correctly
	*if you wish to run a special task, set the relevant task to true
	*if you wish to use overtraining, you have to set the parameters in the special params section
4. Run by using "python src/binding_prediction.py"

Several clustered and unclustered datasets are available in the data folder.

This algorithm currently only works with pKd scores. Leave the general section as it is. When the general setting is set to davis
the model works on both the pKd scores from the Davis set as well as those from the BindingDB.
The compound setting should not be changed as well, unless another way of encoding the compounds is implemented. 


#### Embeddings

[ChemVAE](https://github.com/aspuru-guzik-group/chemical_vae): Variational auto encoder that was used to create compound embeddings

[T5 Embeddings](https://github.com/agemagician/ProtTrans): Resource for creating protein sequence embeddings

<!--

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



#### config.ini

This file is for adjusting the settings that should be used for training a new model.
-->


