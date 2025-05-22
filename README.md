# Crystal Graph Convolutional Neural Networks (CGCNNs) for materials properties prediction
Doing materials calculations with first-principles methods like Density Functional Theory (DFT) is computationally expensive, usually requiring supercomputing clusters and large time frames to compute. Machine learning methods arise as an interesting alternative to speed up these calculations in certain contexts. For example, we could use classification machine learning techniques to predict if a molecule is toxic or non-toxic, or if a given material is an insulator or conductor. Another use could be for regression problems, such as predicting energies for a given structure of a crystal material.

We are interested in being able to predict band gaps to account for the thermal effect on the band gap in anharmonic semiconductor materials. These computations require hundreds of thousands of hours of computation; thus, using machine learning prediction models, we could speed up these calculations.

The main problem is how to express the information of the unit cell (lattice parameters and ion positions) in a way that we can feed into a machine learning method. The best method is to use Crystal Graph Convolutional Neural Networks (CGCNNs) [[1]](#1). Historically, molecules were mapped to graph structures for quantum chemistry machine learning applications. However, mapping a unit cell of crystal material to a graph is not as easy. The main problem is how to express the periodicity of the cell (molecules do not have this problem).

In this approach, we generate graphs with as many nodes as there are atoms in the unit cell. Each node has four different features: atomic number, electronegativity, ion weight (in u), and ion radius (in pm). To account for the periodicity of the unit cell, we consider that two nodes $i$ and $j$ are connected by an edge if their Euclidean distance is less than a cutoff radius (typically a few angstroms), i.e., if $d_{i,j} < R_{cutoff}$. For each atom in the unit cell, it is verified if the other atoms are inside a sphere of radius $R_{cutoff}$ centered on the atom of interest. A supercell big enough is created to account for all the possible connections inside the cutoff sphere. The edge feature will be the Euclidean distance.

Once we have the graphs, convolutional graph neural networks are used to perform graph regression. An excellent introduction to the topic of graph convolutional neural networks, with interactive figures, can be found in [[2]](#2).


## Installation

To download the repository, use:

```bash
$ git clone https://github.com/polbeni/cgcnn
```

## Requirments

The required Python packages to execute the different scripts are:
- mp-api
- pymatgen
- matplotlib
- numpy
- pandas
- scikit-learn
- torch
- torch_geometric

The different modules can be downloaded manually, or you can execute the following command to install them automatically in a python environment:
```bash
$ cd env/train-cgcnn/
$ pip install -r requirements.txt
```
I have verified they work properly in MacOS and GNU/Linux machines.


## Functionalities

The available functionalities are:
- Create a materials database using [The Materials Project](https://next-gen.materialsproject.org/).
- Create graphs from materials structure files (such as POSCAR or cif files), and normalize or standardize their features.
- Create a CGCNN model (model architecture and machine learning parameters (learning rate, batch size, ...) can be easily modified from the scripts).
- Perform hyperparameter testing, screening for different models and parameters such as the learnin rate or dropout. The models are trained on the training set and evaluated on the validation set. Then the models are ranked using different metrics from their performance in the validation set. The performance of the final model can be verified on the test set.
- Train the model with the created database and re-train the model (as many times as you want) with your own DFT results.
- Use the trained (or re-trained) model to predict materials properties.

When training the CGCNN model, GPU (with CUDA) will be used preferably over CPU. However, if not CUDA detected the model will train over CPU. For now not compatible with Apple Silicon GPU (MPS).

## How to use it

### Create the materials database
To create a materials database to train a CGNN model that predicts crystal structure properties, we can use materials calculation databases. Here we use the API of [The Materials Project](https://next-gen.materialsproject.org/). We are interested in the structure and band gap values of all materials that contain at least one of the chemical species of interest and have a non-null band gap (insulators and semiconductors). In the `api-materials-project` directory, edit the `find-materials-mp.py` script to download the materials and features of interest (IMPORTANT: you should provide your Materials Project API key, which you can find [here](https://next-gen.materialsproject.org/api#api-key)). Then execute the script:
```bash
$ python3 find-materials-mp.py
```

### Generate and normalize graphs
Once we have our materials database, we need to convert our structures into graphs to train the graph convolutional neural network model. To create the graphs, execute the following scripts in the `create-graphs` directory:
```bash
$ python3 create-graphs.py
$ python3 create-csv.py
```
The file `create-graphs/atoms_dict.json` contains the features for each atom ('atomic' nuumber, electronegativity, ion weight (in u), and ion radius (in pm)), and it is used to create the node features. Edges only have one feature, the euclidean distance.

Then, we can normalize (uniform distribution) or standardize (Gaussian distribution) the nodes and edges features with:
```bash
$ python3 normalize-graphs.py
```
or
```bash
$ python3 standerize-graphs.py
```
respectively.

After normalization (or standardization), output files with the normalization (or standardization) values are generated to be used later, such as when re-training the model with new data or predicting new materials.

### Create and train the CGCNN model
The next step is to create a graph convolutional neural network model and train it. We used [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). Go to the `models-gpu` directory and edit `models-gpu/cgcnn-model.py` with the desired parameters for the training (training set size, learning rate, number of epochs, etc.) as well as the model architecture, and the correct path to graphs and related files. Then execute the file:
```bash
$ python3 cgcnn-model.py
```
The provided outputs are: a file with the final errors and epoch losses, a plot with the evolution of the losses, a plot with the prediction results, and a file with the trained model named `trained_model`.

To re-train the model with new DFT data, use the script in the `models-gpu` directory:
```bash
$ python3 cgcnn-retrain.py
```
The graphs can be normalized (or standardized) using the same parameters used for the original database, or using new parameters.

### Predict properties of new materials
Once we have a model, we can use the script in `predict-bandgaps` to make predictions (in our case, predict band gaps):
```bash
$ python3 compute-bg.py
```
Now the graphs should be normalized (or standardized) using the same parameters used for the trained model used.

## Authors

This code and repository are being developed by:
- Pol Benítez Colominas (pol.benitez@upc.edu)

## References

<a id="1">[1]</a> 
XIE, Tian; GROSSMAN, Jeffrey C. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. <em>Physical review letters</em>, 2018, 120.14: 145301.

<a id="2">[2]</a> 
SANCHEZ-LENGELING, Benjamin, et al. A Gentle Introduction to Graph Neural Networks. DOI: 10.23915/distill.00033
