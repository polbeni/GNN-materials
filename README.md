# Crystal Graph Neural Networks (CGNNs) for materials properties prediction
Code for Crystal Graph Neural Networks


## Installation

To download the repository, use:

```bash
$ git clone https://github.com/polbeni/cgnn
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

The different modules can be downloaded manually, or you can execute the following command to install them automatically:
```bash
$ pip install -r requirements.txt
```


## Functionalities

The available functionalities are:
- Create a materials database using [The Materials Project](https://next-gen.materialsproject.org/).
- Create graphs from materials structure files (such as POSCAR or cif files), and normalize or standardize their features.
- Create a CGNN model.
- Train the model with the created database and re-train the model with your own DFT results.
- Use the trained (or re-trained) model to predict materials properties.

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

### Create and train the CGNN model
The next step is to create a graph convolutional neural network model and train it. We used [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). Go to the `convolutional-graph-neural-networks` directory and edit `convolutional-graph-neural-networks/cgnn-model.py` with the desired parameters for the training (training set size, learning rate, number of epochs, etc.) as well as the model architecture, then execute the file:
```bash
$ python3 cgnn-model.py
```
The provided outputs are: a file with the final errors and epoch losses, a plot with the evolution of the losses, a plot with the prediction results, and a file with the trained model named `convolutional-graph-neural-networks/trained_model`.

Alternatively, if you want to train the model with normalized output, you can do:
```bash
$ python3 cgnn-model-normalized-output.py
```

To re-train the model with new DFT data, use the script in the `retrain-model` directory:
```bash
$ python3 cgnn-retrain.py
```
Remember that the graphs should be normalized (or standardized) using the same parameters used for the original database.

### Predict properties of new materials
Once we have a model, we can use the script in `predict-bandgaps` to make predictions (in our case, predict band gaps):
```bash
$ python3 compute-bg.py
```
Again, remember that the graphs should be normalized (or standardized) using the same parameters used for the original database.

## Authors

This code and repository are being developed by:
- Pol Benítez Colominas (pol.benitez@upc.edu)

## References

<a id="1">[1]</a> 
XIE, Tian; GROSSMAN, Jeffrey C. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. <em>Physical review letters</em>, 2018, 120.14: 145301.

<a id="2">[2]</a> 
SANCHEZ-LENGELING, Benjamin, et al. A Gentle Introduction to Graph Neural Networks. DOI: 10.23915/distill.00033
