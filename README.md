# Evidential Relational-Graph Convolutional Networks for Entity Classification in Knowledge Graphs
This repository contains a Pytorch-based implementation to reproduce [Evidential Relational-Graph Convolutional Networks for Entity Classification in Knowledge Graphs](#link.pdf), as published in [CIKM 2021](https://www.cikm2021.org/), as well as more general code to leverage evidential learning to train Graph Convolutional Networks to measures uncertainty in knowledge graphs.
The code is adpated from [Kipf's Keras-based implementation](https://github.com/tkipf/relational-gcn). See details in [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (2017).


![Overview of the E-R-GCN Pipeline](https://github.com/TobiWeller/E-R-GCN/blob/main/assets/overview.png?raw=true)


## Setup
To use this package, you must install the following dependencies first: 
- Compatible with PyTorch 1.4.0 and Python 3.7.3.
- Dependencies can be installed using `requirements.txt`.

## Training
You can learn the representations on the datasets AIFB, MUTAG, BGS and AM using E-RGCN by specifying the dataset with the argument --dataset. This file loads the knowledge graphs and learns a model for node classification.
We include early-stopping  mechanisms in `pytorchtools.py` to pick the optimal epoch.


- AIFB: 
```shell
python run.py --data aifb --epochs 50 --bases 0 --hidden 16 --lr 0.01 --l2 0
```

- MUTAG: 
```shell
python run.py --data mutag --epochs 50 --bases 30 --hidden 16 --lr 0.01 --l2 5e-4
```

- BGS: 
```shell
python run.py --data bgs --epochs 50 --bases 40 --hidden 16 --lr 0.01 --l2 5e-4
```
- AM:
```
python run.py --data am --epochs 50 --bases 40 --hidden 10 --lr 0.01 --l2 5e-4
```
Note: Results depend on random seed and will vary between re-runs.
* `--bases` for RGCN basis decomposition
* `--data` denotes training datasets
* `--hidden` is the dimension of hidden GCN Layers
* `--lr` denotes learning rate
* `--l2` is the weight decay parameter of L2 regularization
* `--drop` is the dropout value for training GCN Layers
* Rest of the arguments can be listed using `python run.py -h`



## Results
The following image shows the results reported in the [Paper](#link.pdf). Considering the results for entity classification we observe on the one hand that our model E-R-GCN achieves state-of-the-art results on AIFB and AM, outperforming R-GCN in particular, and on the other hand that the results reported in [R-GCN](https://github.com/tkipf/relational-gcn) are confirmed. The experimental results show a performance increase compared to the previous R-GCN model.
![Results of E-R-GCN for node classification in KG](https://github.com/TobiWeller/E-R-GCN/blob/main/assets/results.png?raw=true)



The use of the evidential learning approach makes no overconfident predictions and allows to specify uncertainties of the model and thus to accurately determine whether a model is suitable for prediction on a subgraph, which is currently not considered in classical ML approaches. This is particularly important when considering dynamic changes in knowledge graphs or to detect out-of-distribution samples.


![Uncertainty predictions of E-R-GCN and R-GCN with increasing number of triples](https://github.com/TobiWeller/E-R-GCN/blob/main/assets/robust.png?raw=true)



## Citation
If you use this code for evidential learning as part of your project or paper, please cite the following work:  

@inproceedings{weller2021ergcn,
    author = {Weller, Tobias and Paulheim, Heiko},
    title = {Evidential Relational-Graph Convolutional Networks for Entity Classification in Knowledge Graphs},
    year = {2021},
    isbn = {9781450384469},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3459637.3482102},
    doi = {10.1145/3459637.3482102},
    booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
    pages = {3533–3537},
    numpages = {5},
    keywords = {evidential learning, graph convolutional neural network, knowledge graph, entity classification},
    location = {Virtual Event, Queensland, Australia},
    series = {CIKM '21}
}
    

## Licence
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
