# Sequence Labeling Parsing by Learning Across Representations

This repository contains the source code for the paper ["Sequence Labeling Parsing by Learning Across Representations"](https://arxiv.org/pdf/1907.01339.pdf) that has been accepted for ACL 2019. 

## Core idea

Recent research has shown that constituency and dependency parsing can be recast 
as sequence labeling problem resulting in a fast and accurate parsing. We extend this idea by adding multitask learning (MTL) that 
enables learning both constituency and dependency syntactic abstractions and leveraging their complementary nature. 
When adding one parsing paradigm as auxiliary loss, the performance of the other paradigm 
consistently improves. In other words, our model achieves better results for dependency parsing when information about the syntactic 
representations of constituency trees is available to the model during training and vice versa. Moreover, 
a _single_ sequence labeling model can serve for parsing _both_ constituency and dependency trees 
at almost no cost in terms of performance or speed (when both paradigms are considered as the main tasks).

* [1. Constituency and Dependency Parsing as Sequence Labeling explained](#constituency-and-dependency-parsing-as-sequence-labeling)
* [2. Requirements](#requirements)
* [3. Encoding Trees into Labels](#encoding-trees-into-labels)
* [4. Usage (train and parse)](#usage)
* [5. Reference](#reference)

## Constituency and Dependency Parsing as Sequence Labeling

Various encodings has been proposed for linearizing syntactic trees.
This code employs the relative encoding proposed in ["Constituent Parsing as Sequence Labeling"](https://aclweb.org/anthology/D18-1162) and relative PoS-based encoding from
["Viable Dependency Parsing as Sequence Labeling"](https://www.aclweb.org/anthology/N19-1077). 
Below some examples with an intuitive explanation of how constituency and dependency trees can be 
encoded as labels. More detailed descriptions are to find in the papers. 

<p align="center">
  <img src="https://github.com/mstrise/seq2label-crossrep/blob/master/fig/trees.png">
</p>

**_(a) a constituency tree_** with the relative variation of encoding. 

_Example for the word "control" with the label (-2,S,Ø)_: 
First, we count how many ancestors are shared between the target word "control" and the next token("."), 
which is 1 since they only share "S". Next, we look at the relative variation with respect to 
the previous word. The tokens "good" and "control" share 3 levels. After subtraction of relative 
variation from the absolute, we get **-2**. The second element **"S"** is the last ancestor that is
 shared between the target and next word. The third element in the tuple is empty (**Ø**)
  because the target word is not a part of a leaf unary chain. 

**_(a) a dependency tree_** with the relative PoS-based encoding.

_Example for the word "control" with the label (-1,V,dobj)_: The head of the target word is the 
first token to the left (**-1**) that has PoS tag **V** with the relation **dobj**. 

Our code is an extension of the original code for [constituency](https://github.com/aghie/tree2labels/tree/naacl-2019) and 
[dependency](https://github.com/mstrise/dep2label) parsing. We also rely on 
[NCRF++](https://github.com/jiesutd/NCRFpp): An Open-source Neural Sequence Labeling Toolkit that
 we have slightly 
modified. In MTL setup, our model learns many tasks jointly that consist of dependency and constituency parsing
 and in that way it can benefit from learning their shared syntactic representations.


<p align="center">
  <img src="https://github.com/mstrise/seq2label-crossrep/blob/master/fig/architecture.png">
</p>

## Requirements

It is recommended to create a virtual environment in order to keep the installed packages separate to avoid conflicts
 with 
other programs.

* ```Python 2.7```
* ```PyTorch 0.3```

The program was tested on Ubuntu 16.04, Python 2.7.12, PyTorch 0.3.1.

## Encoding trees into labels

In order to train and run the model, the input files have to contain encoded trees
 into the labels. It can be easily done by using the following scripts for dependency 
 and constituency trees. We follow the format setup proposed in 
 [Constituent Parsing as Sequence Labeling: better, faster, stronger sequence tagging constituent parsers](https://github.com/aghie/tree2labels/tree/naacl-2019).
### Labels for Dependency Parsing

```bash
python encode_trees2labels.py --file_to_encode $PATH_TO_THE_CONLL-X_FILE --output $PATH_TO_OUTPUTFILE --task SINGLE/COMBINED/MULTI
```

The file to encode has to be in the [CoNLL-X](https://universaldependencies.org/format.html) format:
```
1   He      _   N   N   _   2   nsubj   _   _      
2   has     _   V   V   _   0   root    _   _
3   good    _   J   J   _   4   amod    _   _ 
4   control _   N   N   _   2   dobj    _   _
5   .       _   .   .   _   2   punct   _   _
```

The output will differ depending on the chosen MTL setup:

* ```single``` a label will be treated as single task where its components are separated by the symbol "@"
```
He      N   +1@nsubj@V
has     V   -1@root@ROOT
good    J   +1@amod@N
control N   -1@dobj@V
.       .   -1@punct@V
```

* ```multi``` each component in the label separated by "{}" will be treated as a separate task
```
He      N   +1{}nsubj{}V
has     V   -1{}root{}ROOT
good    J   +1{}amod{}N
control N   -1{}dobj{}V
.       .   -1{}punct{}V
```

* ```combined``` best performance is achieved when information about the rel.position and 
word's head (f.ex. +1@V) is combined into one task and dependency relation as the second task
```
He      N   +1@V{}nsubj
has     V   -1@ROOT{}root
good    J   +1@N{}amod
control N   -1@V{}dobj
.       .   -1@V{}punct
```
### Labels for Constituency Parsing

```bash
python dataset.py
```
Check the usage [here](https://github.com/aghie/tree2labels/tree/naacl-2019).

### Multitask Labels for both Dependency and Constituency Parsing

Labels for constituency and dependency parsing have to be merged in order to train both parsing paradigms. 

```bash
python merge_dep_and_cons_files.py --encoded_file_constituency $PATH_TO_ENCODED_FILE_CONSTITUENCY --encoded_file_dependency $PATH_TO_ENCODED_DEPENDENCY --merged_output $PATH_TO_OUTPUT 
```


An example of the output:

```
Monday	NNP 	1ROOT{}S{}NP{}-1@IN{}pobj
```
The format of each line is: a token, followed by its PoS tag as a self-defined feature and a label with 5 tasks (separated by {})

## Usage

### Train a model

```bash
python main.py --config $PATH_TO_CONFIG_FILE_FOR_TRAINING  
```
[config files](https://github.com/mstrise/seq2label-crossrep/tree/master/config)
 for both single and multi-task models for constituency and dependency parsing. The names of the models 
 follow the original naming used in the paper. 
 
##### MTL setup in config files: examples

For a multi-task label:

```
multi-task label      3{}NP{}-EMPTY-{}+1@NNS{}amod
index of task         0  1      2        3     4
```

 with 5 tasks in total where the first 3 tasks are for constituency(```3{}NP{}-EMPTY-```) and 
 the last 2 for dependency(```+1@NNS{}amod```), we can specify:
 
 an MTL setup for constituency parsing as the main task and dependency parsing as auxiliary task:

```Python
###MTL setup###
index_of_main_tasks=0,1,2
tasks=5
tasks_weights=1|1|1|0.1|0.1
dependency_parsing=False
constituency_parsing=True
```
 
or an MTL setup for dependency parsing as the main task and constituency parsing as auxiliary task:

```Python
###MTL setup###
index_of_main_tasks=3,4
tasks=5
tasks_weights=0.2|0.2|0.2|1|1
dependency_parsing=True
constituency_parsing=False
```

or an MTL setup where both dependency and constituency parsing are considered as main tasks:

```Python
###MTL setup###
index_of_main_tasks=0,1,2,3,4
tasks=5
tasks_weights=1|1|1|1|1
dependency_parsing=True
constituency_parsing=True
```


## Parse with a pre-trained model

```bash
python decode.py --test $PATH_ENCODED_TESTSET [--gold_constituency] $PATH_GOLD_TESTSET_FOR_CONST [--gold_dependency] $PATH_GOLD_TESTSET_FOR_DEPEN --model $PATH_TO_MODEL --status test --gpu False [--multitask] [--output_constituency] $PATH_OUTPUT_CONS [--output_dependency] $PATH_OUTPUT_DEPEN --ncrfpp $PATH_TO_NCRFPP 
```
* ```--test``` path to the encoded testset
* ```--gold_constituency``` path to the gold constituency file. Required when  ```constituency_parsing=True``` defined in the config file
* ```--gold_dependency``` path to the gold dependency file. Required when  ```dependency_parsing=True``` 
* ```--model``` path to the model ended with "/mod"
* ```--gpu``` True/False
* ```--multitask``` when multitask labels are used 
* ```--output_constituency``` path to constituency output file. Only if ```constituency_parsing=True```
* ```--output_dependency``` path to dependency output file. Only if ```dependency_parsing=True```
* ```--ncrfpp``` path to the directory containing main.py and decode.py

## Acknowledgements

This work has received funding from the European Research Council (ERC), under the European Union's Horizon 2020 research and innovation programme (FASTPARSE, grant agreement No 714150).

## Reference

If you wish to use our work for research purposes, please cite us!
```
@inproceedings{strzyz-etal-2019-sequence,
    title = "Sequence Labeling Parsing by Learning across Representations",
    author = "Strzyz, Michalina  and
      Vilares, David  and
      G{\'o}mez-Rodr{\'\i}guez, Carlos",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1531",
    doi = "10.18653/v1/P19-1531",
    pages = "5350--5357",
    abstract = "We use parsing as sequence labeling as a common framework to learn across constituency and dependency syntactic abstractions.To do so, we cast the problem as multitask learning (MTL). First, we show that adding a parsing paradigm as an auxiliary loss consistently improves the performance on the other paradigm. Secondly, we explore an MTL sequence labeling model that parses both representations, at almost no cost in terms of performance and speed. The results across the board show that on average MTL models with auxiliary losses for constituency parsing outperform single-task ones by 1.05 F1 points, and for dependency parsing by 0.62 UAS points.",
}
```

## Contact

Any questions? Bugs? Comments? Contact me using michalina.strzyz[at]udc.es
