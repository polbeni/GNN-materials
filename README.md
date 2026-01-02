# Graph Neural Networks (GNNs) for materials properties prediction
Doing materials calculations with first-principles methods like Density Functional Theory (DFT) is computationally expensive, usually requiring supercomputing clusters and large time frames to compute. Machine learning methods arise as an interesting alternative to speed up these calculations in certain contexts. For example, we could use classification machine learning techniques to predict if a molecule is toxic or non-toxic, or if a given material is an insulator or conductor. Another use could be for regression problems, such as predicting energies for a given structure of a crystal material.

We are interested in being able to predict band gaps to account for the thermal effect on the band gap in anharmonic semiconductor materials. These computations require hundreds of thousands of hours of computation; thus, using machine learning prediction models, we could speed up these calculations. However these scripts and formalism can be used to any general regression task (or even classification with some modifications).

The main problem is how to express the information of the unit cell (lattice parameters and ion positions) in a way that we can feed into a machine learning method. The best method is to use Graph Neural Networks (GNNs) [[1]](#1). Historically, molecules were mapped to graph structures for quantum chemistry machine learning applications. However, mapping a unit cell of crystal material to a graph is not as easy. The main problem is how to express the periodicity of the cell (molecules do not have this problem).

In this approach, we generate graphs with as many nodes as there are atoms in the unit cell. Each node has four different features: atomic number, electronegativity, ion weight (in u), and ion radius (in pm). To account for the periodicity of the unit cell, we consider that two nodes $i$ and $j$ are connected by an edge if their Euclidean distance is less than a cutoff radius (typically a few angstroms), i.e., if $d_{i,j} < R_{cutoff}$. For each atom in the unit cell, it is verified if the other atoms are inside a sphere of radius $R_{cutoff}$ centered on the atom of interest. A supercell big enough is created to account for all the possible connections inside the cutoff sphere. The edge feature will be the Euclidean distance.

Once we have the graphs, convolutional graph networks are used to perform graph regression. An excellent introduction to the topic of graph convolutional networks, with interactive figures, can be found in [[2]](#2).


## Functionalities

The available functionalities are:
- Create a materials database using [The Materials Project](https://next-gen.materialsproject.org/).
- Create graphs from materials structure files (such as POSCAR or cif files), and normalize or standardize their features.
- Create a GCN model (model architecture and machine learning parameters (learning rate, batch size, ...) can be easily modified from the scripts).
- Perform hyperparameter testing, screening for different models and parameters such as the learnin rate or dropout. The models are trained on the training set and evaluated on the validation set. Then the models are ranked using different metrics from their performance in the validation set. The performance of the final model can be verified on the test set.
- Train the model with the created database and re-train the model (as many times as you want) with your own DFT results.
- Use the trained (or re-trained) model to predict materials properties.

When training the GCNN model, GPU (with CUDA) will be used preferably over CPU. However, if not CUDA detected the model will train over CPU. For now not compatible with Apple Silicon GPU (MPS).


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
