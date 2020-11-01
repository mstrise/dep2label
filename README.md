# Dependency Parsing as Sequence Labeling

This repository contains the code with a collection of encodings proposed for Dependency Parsing as Sequence Labeling with [NCRF++](https://github.com/jiesutd/NCRFpp). 


**Dependency Parsing as Sequence Labeling is also available with BERT** 🠊 [repository](https://github.com/mstrise/dep2label-bert/tree/master) 


This is the source code for the following papers accepted at COLING2020:

* "Bracketing Encodings for 2-Planar Dependency Parsing"
* "A Unifying Theory of Transition-based and Sequence Labeling Parsing"

### Overview of the Encoding Family for Dependency Parsing in SL

| Name       | Type of encoding         | supports non-projectivity?  | 
| ------------- |:-------------:| :-------------:|
| ```rel-pos```     | Relative Part-of-Speech-based | :heavy_check_mark: |
| ```1-planar-brackets```     | Bracketing-based      |  :heavy_check_mark: / &#10007; | 
| ```2-planar-brackets-greedy``` |   Second-Plane-Averse Greedy Plane Assignment    |   :heavy_check_mark:  | 
| ```2-planar-brackets-propagation```  |   Second-Plane-Averse Plane Assignment based on Restriction Propagation on the Crossings Graph   |   :heavy_check_mark:  |
| ```arc-standard```  |  Arc Standard in Transition-based   |   &#10007;  | 
| ```arc-eager```  |  Arc Eager in Transition-based   |   &#10007;  | 
| ```arc-hybrid```   | Arc Hybrid in Transition-based    |   &#10007;  | 
| ```covington```  |  Covington in Transition-based    |   :heavy_check_mark:  | 



Relative PoS-based and 1-planar bracketing-based encoding are described in *"Viable Dependency Parsing as Sequence Labeling"* (NAACL2019): [[paper]](https://www.aclweb.org/anthology/N19-1077.pdf) | [[code for results reproducibility]](https://github.com/mstrise/dep2label/tree/naacl2019).

In addition, one may be interested in using SL with:
* constituency parsing representations as described in *"Sequence Labeling Parsing by Learning Across Representations"* (ACL2019): [[paper]](https://www.aclweb.org/anthology/P19-1531.pdf) | [[code]](https://github.com/mstrise/dep2label/tree/acl2019)
* human data (gaze features) as described in *"Towards Making a Dependency Parser See"*(EMNLP2019): [[paper]](https://www.aclweb.org/anthology/D19-1160.pdf) | [[code]](https://github.com/mstrise/dep2label/tree/emnlp2019)

## Requirements

The code is based on [NCRF++](https://github.com/jiesutd/NCRFpp) and Constituency Parsing as Sequence Labeling [code](https://github.com/aghie/tree2labels).

It is recommended to create a virtual environment in order to keep the installed packages separate to avoid conflicts with other programs.

```sh
pip install -r requirements.txt
```


## Scripts for encoding and decoding dependency labels


To encode a CoNNL-X file to SL file:

```bash
python encode_dep2labels.py --input --output --encoding [--mtl] 
```
where
```bash
input=...    # file to encode (CoNNL-X format)
output=...   # output file with encoded dependency trees as labels (SL format)
encoding=... # encoding type= ["rel-pos", "1-planar-brackets", "2-planar-brackets-greedy","2-planar-brackets-propagation","arc-standard", "arc-eager","arc-hybrid", "covington","zero"]
mtl=...      # optionally, nb of multi-tasks= ["1-task","2-task","2-task-combined","3-task"]. By default, type that gives the best results is chosen
```

To decode a SL file to a CoNNL-X file:
```bash
python decode_labels2dep.py --input [--conllu_f] --output --encoding
```
where
```bash
input=...    # file to decode (SL format) 
conllu_f=... # optionally, the corresponding CoNNL-X file (in case of special indexing i.e. 1.1 or 1-2)
output=...   # output file with decoded dependency trees (CoNNL-X format)
encoding=... # encoding type= ["rel-pos", "1-planar-brackets", "2-planar-brackets-greedy","2-planar-brackets-propagation","arc-standard", "arc-eager","arc-hybrid", "covington"]
```
## Training a model

Modify [config file](https://github.com/mstrise/dep2label/tree/master/dep2label/config/train.config) and run the following script:

```bash
python main.py --config 
```
where
```bash
config=...   # path to the config file 
```

## Parsing with a trained model

run the following script:

```bash
python decode.py --test --gold [--predicted] --model --gpu --output --encoding --ncrfpp
```

where
```bash
test=...     # test file with encoded dependency trees (SL format)
gold=...     # gold test (CoNNL-X format)
predicted=...# optionally, CONNL-X file with with the predicted segmentation/tokenization/PoS in case the SL test file is also predicted
model=...    # path to the model (/mod)
gpu=...      # [True,False]
output=...   # output file with decoded trees (CoNNL-X format)
encoding=... # encoding type= ["rel-pos", "1-planar-brackets", "2-planar-brackets--greedy","2-planar-brackets-propagation","arc-standard", "arc-eager","arc-hybrid", "covington"]
ncrf=...     # path to NCRF
```

## Acknowledgements

This work has received funding from the European Research Council (ERC), under the European Union's Horizon 2020 research and innovation programme (FASTPARSE, grant agreement No 714150).

## Reference

If you wish to use our work for research purposes, please cite us!

* Strzyz, Michalina and Vilares, David and Gómez-Rodríguez, Carlos. "Bracketing Encodings for 2-Planar Dependency Parsing". To appear in COLING220.

* Gómez-Rodríguez, Carlos and Strzyz, Michalina and Vilares, David. "A Unifying Theory of Transition-based and Sequence Labeling Parsing". To appear in COLING2020.
