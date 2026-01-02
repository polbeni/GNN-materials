# Graph Neural Networks (GNNs) for materials properties prediction

Doing materials calculations with first-principles methods like DFT is computationally expensive, usually requiring to use supercomputing facilities. Machine learning methods arise as an interesting alternative to speed up these calculations in certain contexts. Some examples could be for regression problems, such as fast determination of forces between atoms in a molecular system (this is what MLIPs do), or classifications tasks, such as distinguish materials by their insulator or conductor behaviour.  

In this repository we provide with a sort of scripts to perform materials property prediction by using graph neural networks (GNNs), from constructing graphs of materials structures, to train GNN models and use them for prediction. (Note that modify these scripts for prediction problems is quite straightforward) 

By using GNNs, the information of the structure (lattice parameters and ion positions) is expressed by graphs (node, edge, and adjacency tensors), and the periodicity of the material unit cell is ensured [[1]](#1). For each atom in the unit cell we generate a node with four different features: atomic number, electronegativity, ion weight, and ion radius. Two nodes $i$ and $j$ are connected by an edge if their Euclidean distance is less than a cutoff radius, and the Euclidean distance is considered as the edge feature. Then we can train a GNN to predict material properties. An excellent introduction to the topic of graph convolutional networks, with interactive figures, can be found in [[2]](#2).

## Functionalities

The available functionalities are:
- Create a materials database using [The Materials Project](https://next-gen.materialsproject.org/).
- Create graphs from materials structure files (such as POSCAR or cif files), and normalize or standardize their features.
- Create a GNN model (model architecture and machine learning parameters (learning rate, batch size, ...) can be easily modified from the scripts).
- Perform hyperparameter testing, screening for different models and parameters such as the learnin rate or dropout. The models are trained on the training set and evaluated on the validation set. Then the models are ranked using different metrics from their performance in the validation set. The performance of the final model can be verified on the test set.
- Train the model with the created database and re-train the model (as many times as you want) with your own DFT results.
- Use the trained (or re-trained) model to predict materials properties.
- Use explainability tools to get the most important graph edges when using a trained model.
- Get the embedding of a graph from a pretrained model.

When using the scripts, GPU (with CUDA) will be used preferably over CPU. However, if not CUDA detected CPU will be used. For now, there is no compatibility with Apple Silicon GPU (MPS).


## Installation

To download the repository, use:

```bash
$ git clone https://github.com/polbeni/GNN-materials
```


## Requirments

The required Python packages to execute the different scripts are:
- matplotlib
- mp-api
- numpy
- pandas
- pymatgen
- scikit-learn
- torch
- torch_geometric

The different modules can be downloaded manually, or even easier, installed by using the `requirements.txt` files that can be found in the `env` dir:
```bash
$ pip install -r requirements.txt
```
It should work in MacOS and GNU/Linux machines.


## How to cite

If you use this repository, please cite it as follows:
```
@article{benitez2025physics,
  title={Why Physics Still Matters: Improving Machine Learning Prediction of Material Properties with Phonon-Informed Datasets},
  author={Ben{\'\i}tez, Pol and L{\'o}pez, Cibr{\'a}n and Saucedo, Edgardo and Mizoguchi, Teruyasu and Cazorla, Claudio},
  journal={arXiv preprint arXiv:2511.15222},
  year={2025}
}
```

## Authors

This code and repository are being developed by:
- Pol Benítez Colominas (pol.benitez@upc.edu)

## References

<a id="1">[1]</a> 
XIE, Tian; GROSSMAN, Jeffrey C. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. <em>Physical review letters</em>, 2018, 120.14: 145301.

<a id="2">[2]</a> 
SANCHEZ-LENGELING, Benjamin, et al. A Gentle Introduction to Graph Neural Networks. DOI: 10.23915/distill.00033
