# Viable Dependency Parsing as Sequence Labeling

We recast dependency parsing as a sequence labeling problem 
where each input word from a sentence is represented as a label 
using one of four encodings. Our BILSTM-based models yield fast and accurate dependency 
parsing without using any traditional parsing algorithms. The repository contains source 
code for the paper "Viable Dependency Parsing as Sequence Labeling" that has been accepted
by NAACL 2019. 

#### How does Dependency Parsing as Sequence Labeling work in short?

###### ENCODING

Each input token is encoded into a label that preserves information about its head and relation.

##### Types of encoding with examples
 
![encodings](https://github.com/mstrise/seq2label/blob/master/figs/encodings.png)

1. __Naive positional encoding__: encodes the real index position in the sentence of the head. A label
 ```2_nsubj``` for the token _Alice_
means "the head of the word _Alice_ is at index 2 with the relation nsubj"
2. __Relative positional encoding__: encodes the difference between the index position of the head
and its dependent (- sign when the head is on the left side of the word and + sign otherwise).
A label  ```+1_nsubj``` for the token _Alice_
means "the head of the word _Alice_ is at index position +1 to the right from that word with the relation nsubj"
3. __Relative PoS-based encoding__: encodes the the distance of the head's PoS tag from its dependent 
(- sign when the head is on the left side of the word and + sign otherwise).
A label ```V_+1_nsubj``` for the token _Alice_
means "the head of the word _Alice_ is the first word to the right with a PoS tag V and with the relation nsubj"
4. __Bracketing-based encoding__: represents words referring to the form of regular expressions. 
A label ```<\>_dobj``` for the token _apple_
means "the word _apple_ is the head of the preceding word indicated by```<\ ``` and has an incoming arc from a word
somewhere to the left ```>``` with the relation dobj".

More detailed explanation of the encodings can be found in the paper.

An example line from a file in the 
[CONLL format](https://universaldependencies.org/format.html): ```1    Alice   _   NNP NNP _   2   nsubj   _   _       ```
can be tansformed with ```encoding 3``` into a new format (token + its label) that will be fed into [NCRF++](https://github.com/jiesutd/NCRFpp):```Alice    V_+1_nsubj    ```


###### NCRF++

We use [NCRF++](https://github.com/jiesutd/NCRFpp): An Open-source Neural Sequence Labeling Toolkit that
 we slightly 
modified. It is recommended to familiarize oneself with this system and its architecture for better understanding.


###### DECODING

As the last step, the output of [NCRF++](https://github.com/jiesutd/NCRFpp) is decoded and transformed back to the 
CONLL 
format. Additionally, the output is postprocessed in order to assure that each sentence is well-formed (for instance 
that the 
system only outputs acyclic syntactic trees or if no root is predicted the system searches throught the 
top 3 most probably labels for each token). A more detailed explanation can be found in the paper.


## Requirements

It is recommended to create a virtual environment in order to keep the installed packages separate to avoid conflicts
 with 
other programs.

* ```Python 2.7```
* ```PyTorch 0.3```

The program was tested on Ubuntu 16.04, Python 2.7.12, PyTorch 0.3.1.

## Usage

#### Train a model

```bash
python main.py --train-config $PATH_TO_CONFIG_FILE_FOR_TRAINING --decode-config $PATH_TO_CONFIG_FILE_FOR_DECODING 
```
* ```--train-config``` an example of a [config file for training](https://github.com/mstrise/seq2label/blob/master/config/train.config), where


```Python
### I/O ###
train_gold=...  # gold file for training 
dev_gold=...    # gold file for evaluation on the development set
train_enc_dep2label=... # file to which the system will write the encoded dependency tree with its labels for training set
dev_enc_dep2label=...   # file to which the system will write the encoded dependency tree with its labels for dev set
model_dir=... # directory where model will be stored
word_emb_dir=...    # extrernal word embeddings
encoding=...    # type of encoding that one wants to use: 1,2,3 or 4. Encoding 3 is set as default for the best performance
eval_type=...   # format of the file: CONLL (for PTB) or CONLLU (for UD). Different scripts are used to evaluate them. The first one excludes the punctuation. 
postag_type=... # type of PoS tags that one wants to use as a self-defined feature: UPoS: Universal part-of-speech tag or XPOS: Language-specific part-of-speach tag  
```

* ```--decode-config``` an example of a [config file for decoding](https://github.com/mstrise/seq2label/blob/master/config/decode.config), where

```Python
### Decode ###
test_gold=... # gold file for testing
input=... # file in the CoNLL-U format with either gold or predicted segmentation and PoS tags
raw_dir=... # file to which the system will write the encoded dependency tree with its labels for testing
output_nn=... # output from NCRF++ in the form of: TOKEN---POS FEATS---TOP 3 MOST PROBABLE LABELS FOR A GIVEN TOKEN
parsedTree=...  # final parsed file 
dset_dir=... # directory for .dset file (should be the same as model_dir defined in train.config)
load_model_dir=... # directory for .model file (should be the same as model_dir defind in train.config)  
```




In this work, the same [pretrained word embeddings](https://github.com/clab/lstm-parser/) for English and 
Chinese were used as in the paper: 
[Transition-Based Dependency Parsing with Stack Long Short-Term Memory](https://arxiv.org/abs/1505.08075). For other 
languages we used [pretrained word embeddings](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1989)
 from CoNLL 2017 Shared Task.
 
 The training config file above contains the hyperparameters for the 
 ![](https://latex.codecogs.com/gif.latex?$P^{\mathrm{C}}_{\mathrm{2,800}}$) model presented in the paper. The best 
 model will be saved as "XXX_best.model" based on the highest UAS score on the development set.

#### Parse with a pre-trained model

```bash
python main.py  --decode-config $PATH_TO_CONFIG_FILE_FOR_DECODING 
```
* ```--decode-config``` an example of a [config file for decoding of the best model](https://github.com/mstrise/seq2label/blob/master/config/decode_best_model.config)


#### Script for tree encoding

You can easily encode your trees into labels using the following script:

```bash
python encode_trees2labels.py --fileToEncode $PATH_TO_THE_CONLLU_FILE --output $PATH_TO_THE_OUTPUT_FILE --encoding 
$TYPE_OF_ENCODING:1,2,3,4 --pos $UPOS/XPOS 
```


## Reference

If you wish to use our work for research purposes, please cite us!
```
@InProceedings{ViableDependencyParsing2019,
 author    = {Strzyz, Michalina and Vilares, David and G\'omez-Rodr\'iguez, Carlos},
 title     = {Viable Dependency Parsing as Sequence Labeling},
 booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
 year      = {2019},
 address   = {Minneapolis, Minnesota, USA},
 publisher = {Association for Computational Linguistics},
 pages     = {to appear}
}
```

Original paper for [NCRF++](https://github.com/jiesutd/NCRFpp)

```
@inproceedings{yang2018ncrf,  
 title={NCRF++: An Open-source Neural Sequence Labeling Toolkit},  
 author={Yang, Jie and Zhang, Yue},  
 booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
 Url = {http://aclweb.org/anthology/P18-4013},
 year={2018}  
}
```

## Contact

Any questions? Bugs? Comments? Contact me using michalina.strzyz[at]udc.es
