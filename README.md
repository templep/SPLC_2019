# SPLC_2019
companion webpage for SPLC 2019 submission

This page gathers scripts, results and images that were used in the SPLC 2019 paper intitled: "Generating Adversarial Configurations for Quality Assurance of Software Product Lines".
Authors: Paul TEMPLE, Mathieu ACHER, Gilles PERROUIN, Battista BIGGIO, Jean-Marc JEZEQUEL, Fabio ROLI

## Pre-requisites:
The scripts use Python 3 with compatible versions of the following libraries: Numpy, Matplotlib, Scikit-learn and pickle
They also call the J48 model of Weka.
The folder weka contains a jar (called weka.jar) which should embed weka libraries.

## Description of the folders:
### data
The "data" folder provides different files which are used as inputs in the scripts.
Model_classif.txt contains a J48 model from Weka trained on configurations provided in train.csv
train_format_weka.arff is a version of train.csv converted to a compatible format for Weka (train_format_weka.csv is the equivalent but in csv which avoids to go through all the definitions of types, etc.)

train* are files used as the training set for the J48 machine learning algorithm.
test* files are equivalent but contain configurations used to evaluate the learnt classifier.

### scripts
The scripts folder contains all the scripts used to generate results in the paper.
The hierarchy of folders tries to follow the research questions addressed in the paper.
The two python scripts directly available in the script folder are the ones used to generate boxplots and provide the visualization of valid/non-valid configurations.

The folder named "prepare_data_and_model" contains two scripts: one to use the dummification procedure discussed in the paper and the other one to initially train a J48 classifier.

### results
The "result" folder contains results that were generated and exploited in the paper.
The folder named "4000_gen_adv_config" is related to results of RQ1 while the "retrain" folder is related to results of RQ2.
"images" contains images used in the article.
"importance_features" provides the features' order of importance regarding the classifier. The following columns provide statistics measures regarding these features in, first, the legitimate configurations (without any adversarial configurations) and, second, regarding adversarial configurations only.
These measures are exploited and discussed in RQ2.

"config_pt" is an intermediate folder used by scripts in which generated configurations are saved.

## Run the scripts:
First, if a classifier needs to be trained on new data, please use the script called "script.py" in the prepare_data_and_model folder.
Then, depending on which aspect you want to reproduce, go to the desired folder and execute the python script.

RQ1.1 will produce 4000 adversarial configurations that tries to fool the previously trained classifier.
Parameters of the attack should be modified in the script directly (label from which the attacks start, number of moved/displacements and steps).

RQ1.2 evaluates whether produced adversarial configurations are valid with regards to the MOTIV Feature Model in terms of feature values.

RQ1.3 produce 4000 configurations following the same procedure as for adversarial configurations except that random modifications and decisions are drawn instead of following the evasion attack.
Parameters of the attack should be modified in the script directly (label from which the attacks start, number of moved/displacements and steps).

RQ1.4 should reuse the script from RQ1.1 and modify the label from 0 to 1 in order to produce adversarial configurations that try to generate non-acceptable configuration out of acceptable ones.

RQ2 creates 25 adversarial configurations but in the end retrain the classified on the original training set augmented with these adversarial configurations.
As for RQ1.1 parameters can be changed
