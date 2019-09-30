# Towards Making a Dependency Parser See

This is the source code for the paper "Towards Making a Dependency Parser See" that has been
accepted for EMNLP 2019.

------------------------------------------------------------------------------------------------------------------
***NEW VERSION OF THE CODE WITH PYTHON3+ AND PYTORCH 1+ IS AVAILABLE [HERE](https://github.com/mstrise/dep2label-up)***.
The code includes Sequence Labeling Parsing for Dependency parsing with Multi-Task Learning, although only on parallel data.

-------------------------------------------------------------------------------------------------------------------

## What is this paper about?

In short, the main idea is to use human eye movement data to guide a dependency parser (although in principle
you can use any complementary data). Since human data may often not be available
at the inference time, we leverage it only during training. To do so, we use a sequence labeling parser
in multi-task (MTL) setup where a model learns to predict gaze features as auxiliary task(s)
while treating dependency parsing as the main task. In addition, it is possible to evaluate a model on 
a disjoint dataset to test the impact of the gaze features extracted from a separate treebank.   

* [1. Gaze features in Sequence Labeling Parsing](#gaze-features-in-sequence-labeling-parsing)
* [2. Requirements](#requirements)
* [3. Usage](#usage)
* [4. Reference](#reference)

## Gaze features in Sequence Labeling Parsing

In our [previous work](https://github.com/mstrise/dep2label)
we recast dependency parsing as sequence labeling problem where for each input token,
a label is predicted that captures its head and dependency relation. Moreover, we have showed in other [work](https://github.com/mstrise/seq2label-crossrep)
 that a sequence labeling parser can benefit from multi-task (MTL) setup. We extend this idea by training a model 
 with gaze features as auxiliary task(s). 

An example of a dependency tree with its encoding and some of the possible gaze features:

<p align="center">
  <img src="https://github.com/mstrise/dep2label-eye-tracking-data/blob/master/pict/tree.png" width="400">
</p>


## Requirements

It is recommended to create a virtual environment in order to keep the installed packages separate to avoid conflicts
 with 
other programs.

* ```Python 2.7```
* ```PyTorch 0.3```

The program was tested on Ubuntu 16.04, Python 2.7.12, PyTorch 0.3.1.

We also rely on 
[NCRF++](https://github.com/jiesutd/NCRFpp): An Open-source Neural Sequence Labeling Toolkit that
 we have slightly modified. It is recommended to get familiar with it.


## Usage

### Data format

###### Parallel data
The initial data should be in CONLL-X format with additional columns containing gaze features. An example of a line:

```
11 beautiful _ ADJ JJ _ 12 amod _ 667.0 222.33 3.0 1.0 243.0 667.0 2.0 1.0 1.0 0.0 140.0 0.0
```
The first 9 columns follow the CONLL-X format while the remaining index of columns represent:
```
9: total fixation duration
10: mean fixation duration 
11: nb fixations
12: fixation probability
13: first fixation duration 
14: first pass duration 
15: nb re-fixations
16: re-read probability
17: w-1 fixation probability
18: w+1 fixation probability
19: w-1 fixation duration
20: w+1 fixation duration
```
A dependency tree can be encoded into labels using the [script](https://github.com/mstrise/dep2label-eye-tracking-data/blob/master/dep2label/preprocessing_human_data/encoded_labels.py).

In addition, it is necessary to prepare and preprocess the human data. You can split and extract the gaze features as we did
by using this [code](https://github.com/mstrise/dep2label-eye-tracking-data/blob/master/dep2label/preprocessing_human_data/human_data.py).

The final file should look like (the example contains only one gaze feature):

(TOKEN, POS-TAG, CONCATENATED LABELS)
```
# TREEBANK=DUNDEE
-BOS-	-BOS-	-BOS-@-BOS-{}-BOS-{}-BOS-
Can	V	+1@V{}aux{}0-20
a	D	+1@N{}det{}80-100
parser	N	+1@N{}aux{}0-20
see	V	-1@ROOT{}root{}0-20
?	P	-1@V{}punct{}60-80
-EOS-	-EOS-	-EOS-@-EOS-{}-EOS-{}-EOS-
```
###### Disjoint data

Data with the encoded dependency labels has to be concatenated with the data containing gaze features as
 labels in following way:

```
# TREEBANK=PTB
-BOS-	-BOS-	-BOS-@-BOS-{}-BOS-
Can	V	+1@V{}aux
a	D	+1@N{}det
parser	N	+1@N{}aux
see	V	-1@ROOT{}root
?	P	-1@V{}punct
-EOS-	-EOS-	-EOS-@-EOS-{}-EOS-
. 
.
. (rest of the data extracted from PTB)
.
.

# TREEBANK=DUNDEE
-BOS-	-BOS-	-BOS-
The	D	0-20
house	N	80-100
is	V	0-20
beautiful	A	0-20
.	P	60-80
-EOS-	-EOS-	-EOS-
. 
.
. (rest of the data extracted from Dundee treebank with gaze features)
.
.

```


### Train a model

```bash
python main.py --config $PATH_TO_CONFIG_FILE_FOR_TRAINING  
```
* [config file](https://github.com/mstrise/dep2label-eye-tracking-data/blob/master/config/parallel.config) 
for **parallel dataset**, where you can define your dataset as:
```
#Treebank(s), main/auxiliary task(s)
dataset=[DUNDEE] nb_tasks=3 main=True weight=1|1|0.1 metric=LAS
```

* [config file](https://github.com/mstrise/dep2label-eye-tracking-data/blob/master/config/disjoint.config)
for **disjoint datasets**, where you can define them as:
```
#Treebank(s), main/auxiliary task(s)
dataset=[PTB] nb_tasks=2 main=True weight=1|1 metric=LAS
dataset=[DUNDEE] nb_tasks=1 main=False weight=0.1
```

## Parse with a pre-trained model

```bash
python parse.py --test xx --goldDependency xx --model xx/mod --status test --gpu False --multitask --outputDependency xx --ncrfpp xx --offset True 
```
* ```--test``` path to the encoded testset
* ```--gold_dependency``` path to the gold dependency file 
* ```--model``` path to the model ended with "/mod"
* ```--gpu``` True/False
* ```--multitask``` when auxiliary task(s) are used
* ```--output_dependency``` path to dependency output file
* ```--ncrfpp``` path to the directory run_nrfpp.py
* ```--offset``` True/False. Forst best performance- True. The first two tuple are learned 
as a single task in dependency parsing (check explanation of that [here](https://github.com/mstrise/seq2label-crossrep#encoding-trees-into-labels))

## Acknowledgements

This work has received funding from the European Research Council (ERC), under the European Union's Horizon 2020 research and innovation programme (FASTPARSE, grant agreement No 714150).

## Reference

If you wish to use our work for research purposes, please cite us!
```
@InProceedings{TowardsMakingaDependencyParserSee2019,
 author    = {Strzyz, Michalina and Vilares, David and G\'omez-Rodr\'iguez, Carlos},
 title     = {Towards Making a Dependency Parser See},
 booktitle = {Proc. of the 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint 
 Conference on Natural Language Processing (EMNLP 2019)},
 year      = {2019},
 address   = {Hong Kong, China},
 publisher = {Association for Computational Linguistics},
 pages     = {to appear}
}
```

## Contact

Any questions? Bugs? Comments? Contact me using michalina.strzyz[at]udc.es
