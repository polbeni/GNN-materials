# Pol Benítez Colominas, August 2025
# Universitat Politècnica de Catalunya

# Plot the performance of the specified model for the training, validation, and test sets


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde


##### Functions
def plot_results(type_set, type_data, units, real_values, predictions, min_value, max_value, ticks_vals, color, output_save):
    """
    Plots the predictions vs DFT value

    Inputs:
        type_set: training, validation, or test
        type_data: the predicted value (band gap, energy, fermi level, or hydrostatic stress)
        units: units of the magnitude
        real_values: list with the DFT computed values
        predictions: list with the GNN predictions
        min_value: minimum value of the range
        max_value: maximum value of the range
        ticks_vals: values for the number of ticks [major, minor]
        color: color for the points
        output_save: name of the output file
    """

    # Compute the density of points
    xy = np.vstack([real_values, predictions])
    z = gaussian_kde(xy)(xy)

    z = np.log1p(z)

    idx = z.argsort()
    x, y, z = real_values[idx], predictions[idx], z[idx]

    # Create the plot
    _, axs = plt.subplots(figsize=(3, 3))

    axs.set_title(type_set)

    axs.set_xlabel('DFT ' + type_data + ' ' + units)
    axs.set_ylabel('Predicted ' + type_data + ' ' + units)

    axs.set_xlim(min_value, max_value)
    axs.set_ylim(min_value, max_value)

    axs.scatter(x, y, c=z, s=10, cmap=color)

    axs.plot([min_value, max_value], [min_value, max_value], linestyle='--', color='black')

    major_locator = MultipleLocator(ticks_vals[0])
    minor_locator = MultipleLocator(ticks_vals[1])
    axs.xaxis.set_major_locator(major_locator)
    axs.xaxis.set_minor_locator(minor_locator)
    axs.yaxis.set_major_locator(major_locator)
    axs.yaxis.set_minor_locator(minor_locator)

    plt.tight_layout()
    plt.savefig(output_save)


def plot_dist(type_data, units, real_values, predictions, lims_dist, ticks_vals_dist, output_save):
    """
    Plots the distribution of the difference between real and predicted value

    Inputs:
        type_data: the predicted value (band gap, energy, fermi level, or hydrostatic stress)
        units: units of the magnitude
        real_values: list with the list of DFT computed values
        predictions: list with the list of GNN predictions
        lims_dist: x and y limits for the distributions [[min_x, max_x], [min_y, max_y]]
        ticks_vals_dist: values for the number of ticks for the distributions [[major_x, minor_x], [major_y, minor_y]]
        output_save: name of the output file
    """

    # Training
    real = real_values[0]
    pred = predictions[0]

    errors = pred - real

    kde = gaussian_kde(errors)
    x_vals_train = np.linspace(errors.min(), errors.max(), 500)
    y_vals_train = kde(x_vals_train)

    # Validation
    real = real_values[1]
    pred = predictions[1]

    errors = pred - real

    kde = gaussian_kde(errors)
    x_vals_val = np.linspace(errors.min(), errors.max(), 500)
    y_vals_val = kde(x_vals_val)

    # Test
    real = real_values[2]
    pred = predictions[2]

    errors = pred - real

    kde = gaussian_kde(errors)
    x_vals_test = np.linspace(errors.min(), errors.max(), 500)
    y_vals_test = kde(x_vals_test)

    # Plot
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel(type_data + '$^{GNN}-$' + type_data + '$^{DFT}$'  + ' ' + units)
    axs.set_ylabel('Frequency (a.u.)')

    axs.axvline(0, color='black', linestyle='--')
    #axs.axhline(0, color='black', lw=1)

    axs.plot(x_vals_train, y_vals_train, color='mediumpurple', lw=2, alpha=0.7, label='Training')
    axs.fill_between(x_vals_train, y_vals_train, color='mediumpurple', alpha=0.2)

    axs.plot(x_vals_val, y_vals_val, color='cornflowerblue', lw=2, alpha=0.7, label='Validation')
    axs.fill_between(x_vals_val, y_vals_val, color='cornflowerblue', alpha=0.2)

    axs.plot(x_vals_test, y_vals_test, color='mediumseagreen', lw=2, alpha=0.7, label='Test')
    axs.fill_between(x_vals_test, y_vals_test, color='mediumseagreen', alpha=0.2)

    axs.set_xlim(lims_dist[0][0], lims_dist[0][1])
    axs.set_ylim(lims_dist[1][0], lims_dist[1][1])

    major_locator = MultipleLocator(ticks_vals_dist[0][0])
    minor_locator = MultipleLocator(ticks_vals_dist[0][1])
    axs.xaxis.set_major_locator(major_locator)
    axs.xaxis.set_minor_locator(minor_locator)
    major_locator = MultipleLocator(ticks_vals_dist[1][0])
    minor_locator = MultipleLocator(ticks_vals_dist[1][1])
    axs.yaxis.set_major_locator(major_locator)
    axs.yaxis.set_minor_locator(minor_locator)

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_save)


def read_data_and_plot(input_path, type_data, units, min_value, max_value, ticks_vals, lims_dist, ticks_vals_dist, output_save):
    """
    Reads the data for the desired magnitude and plot the results

    Inputs:
        input_path: path to the values
        type_data: the predicted value (band gap, energy, fermi level, or hydrostatic stress)
        units: units of the magnitude
        min_value: minimum value of the range
        max_value: maximum value of the range
        ticks_vals: values for the number of ticks [major, minor]
        lims_dist: x and y limits for the distributions [[min_x, max_x], [min_y, max_y]]
        ticks_vals_dist: values for the number of ticks for the distributions [[major_x, minor_x], [major_y, minor_y]]
        output_save: name of the output file
    """

    list_preds = []
    list_reals = []

    # Training
    data_train = np.loadtxt(input_path + '/prediction_vs_DFT_train.txt', skiprows=1)

    pred_vals = data_train[:, 0]
    real_vals = data_train[:, 1]

    plot_results('Training', type_data, units, real_vals, pred_vals, min_value, max_value, ticks_vals, 'Purples', output_save + '_train.pdf')

    list_preds.append(pred_vals)
    list_reals.append(real_vals)

    # Validation
    data_val = np.loadtxt(input_path + '/prediction_vs_DFT_validation.txt', skiprows=1)

    pred_vals = data_val[:, 0]
    real_vals = data_val[:, 1]

    plot_results('Validation', type_data, units, real_vals, pred_vals, min_value, max_value, ticks_vals, 'Blues', output_save + '_validation.pdf')

    list_preds.append(pred_vals)
    list_reals.append(real_vals)

    # Test
    data_test = np.loadtxt(input_path + '/prediction_vs_DFT_test.txt', skiprows=1)

    pred_vals = data_test[:, 0]
    real_vals = data_test[:, 1]

    plot_results('Test', type_data, units, real_vals, pred_vals, min_value, max_value, ticks_vals, 'Greens', output_save + '_test.pdf')

    list_preds.append(pred_vals)
    list_reals.append(real_vals)

    # Distributions
    plot_dist(type_data, units, list_reals, list_preds, lims_dist, ticks_vals_dist, output_save + '_dist.pdf')



##### Main
read_data_and_plot('hyper-energy', '$E_0$', '(eV/atom)', -3.35, -2.7, [0.2, 0.04], [[-0.4, 0.4], [0, 20]], [[0.2, 0.04], [5, 1]], 'E0')
read_data_and_plot('hyper-fermi', '$E_F$', '(eV)', 4.5, 5.9, [0.4, 0.08], [[-0.16, 0.16], [0, 15]], [[0.05, 0.01], [3, 0.6]], 'EF')
read_data_and_plot('hyper-hydrostatic', '$\\sigma_h$', '(GPa)', -1, 11, [5, 1], [[-6, 6], [0, 0.6]], [[2, 0.4], [.2, 0.04]], 'hydrostatic')