# Pol Benítez Colominas, August 2025
# Universitat Politècnica de Catalunya

# Plot the performance of the trained models in the hyperparameter study


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


##### Functions
def get_data(path_to_file):
    """
    It gets the name of the model, the r^2, MAE, and MSE, from the model_performance.txt file

    Inputs:
        path_to_file: path to the model_performance.txt file
    """

    model_name_list = []
    r2_list = []
    MAE_list = []
    MSE_list = []

    with open(path_to_file, 'r') as file:
        next(file)

        for line in file:
            model_name = line.split()[0]
            r2 = float(line.split()[1])
            MAE = float(line.split()[2])
            MSE = (float(line.split()[3]))**0.5 # for RMSE

            if r2 > -1:
                model_name_list.append(model_name)
                r2_list.append(r2)
                MAE_list.append(MAE)
                MSE_list.append(MSE)

    return model_name_list, r2_list, MAE_list, MSE_list


def plot_violins(model_names, r2_list, MAE_list, MSE_list, units, output_name):
    """
    Plots the violin plots for the models performance

    Inputs:
        model_names: list with the model names
        r2_list: list with the r2 metric values
        MAE_list: list with the MAE metric values
        MSE_list: list with the MSE metric values
        units: units of the predicted magnitude
        output_name: name of the magnitude to save the final plots
    """

    models = []
    lr = []
    hidden = []
    dropout = []

    model1 = []
    model2 = []
    model3 = []
    model4 = []
    model5 = []
    model6 = []

    lr2 = []
    lr3 = []
    lr4 = []
    lr5 = []
    lr6 = []

    hidden32 = []
    hidden64 = []
    hidden128 = []
    hidden256 = []

    dropout00 = []
    dropout02 = []
    dropout04 = []
    dropout06 = []

    for model in range(len(model_names)):
        if model_names[model][5] == '1':
            model1.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][5] == '2':
            model2.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][5] == '3':
            model3.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][5] == '4':
            model4.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][5] == '5':
            model5.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][5] == '6':
            model6.append([r2_list[model], MAE_list[model], MSE_list[model]])

        if model_names[model][10] == '2':
            lr2.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][10] == '3':
            lr3.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][10] == '4':
            lr4.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][10] == '5':
            lr5.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][10] == '6':
            lr6.append([r2_list[model], MAE_list[model], MSE_list[model]])

        if model_names[model][12] == '3':
            hidden32.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][12] == '6':
            hidden64.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][12] == '1':
            hidden128.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][12] == '2':
            hidden256.append([r2_list[model], MAE_list[model], MSE_list[model]])

        if (model_names[model][12] == '3') or (model_names[model][12] == '6'):
            correct_in = 16
        else:
            correct_in = 17
        if model_names[model][correct_in] == '0':
            dropout00.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][correct_in] == '2':
            dropout02.append([r2_list[model], MAE_list[model], MSE_list[model]])
        elif model_names[model][correct_in] == '4':
            dropout04.append([r2_list[model], MAE_list[model], MSE_list[model]])
        if model_names[model][correct_in] == '6':
            dropout06.append([r2_list[model], MAE_list[model], MSE_list[model]])

    models.append(model1)
    models.append(model2)
    models.append(model3)
    models.append(model4)
    models.append(model5)
    models.append(model6)

    lr.append(lr2)
    lr.append(lr3)
    lr.append(lr4)
    lr.append(lr5)
    lr.append(lr6)

    hidden.append(hidden32)
    hidden.append(hidden64)
    hidden.append(hidden128)
    hidden.append(hidden256)

    dropout.append(dropout00)
    dropout.append(dropout02)
    dropout.append(dropout04)
    dropout.append(dropout06)

    # Architecture
    metric = []
    for value in models:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][0])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Architecture')
    axs.set_ylabel('$R^2$')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('mediumpurple')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('mediumpurple')
    vp['cmedians'].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_name + '_architecture_r2.pdf')
    plt.close()

    metric = []
    for value in models:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][1])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Architecture')
    axs.set_ylabel('MAE (' + units + ')')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('cornflowerblue')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('cornflowerblue')
    vp['cmedians'].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_name + '_architecture_MAE.pdf')
    plt.close()

    metric = []
    for value in models:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][2])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Architecture')
    axs.set_ylabel('RMSE (' + units + ')')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('mediumseagreen')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('mediumseagreen')
    vp['cmedians'].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_name + '_architecture_RMSE.pdf')
    plt.close()

    # Learning rate
    metric = []
    for value in lr:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][0])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Learning rate')
    axs.set_ylabel('$R^2$')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('mediumpurple')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('mediumpurple')
    vp['cmedians'].set_linewidth(2)

    axs.set_xticks(ticks=[1, 2, 3, 4, 5], labels=['$10^{-2}$', '$10^{-3}$', '$10^{-4}$', '$10^{-5}$', '$10^{-6}$'])

    plt.tight_layout()
    plt.savefig(output_name + '_lr_r2.pdf')
    plt.close()

    metric = []
    for value in lr:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][1])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Learning rate')
    axs.set_ylabel('MAE (' + units + ')')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('cornflowerblue')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('cornflowerblue')
    vp['cmedians'].set_linewidth(2)

    axs.set_xticks(ticks=[1, 2, 3, 4, 5], labels=['$10^{-2}$', '$10^{-3}$', '$10^{-4}$', '$10^{-5}$', '$10^{-6}$'])

    plt.tight_layout()
    plt.savefig(output_name + '_lr_MAE.pdf')
    plt.close()

    metric = []
    for value in lr:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][2])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Learning rate')
    axs.set_ylabel('RMSE (' + units + ')')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('mediumseagreen')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('mediumseagreen')
    vp['cmedians'].set_linewidth(2)

    axs.set_xticks(ticks=[1, 2, 3, 4, 5], labels=['$10^{-2}$', '$10^{-3}$', '$10^{-4}$', '$10^{-5}$', '$10^{-6}$'])

    plt.tight_layout()
    plt.savefig(output_name + '_lr_RMSE.pdf')
    plt.close()


    # Number of hidden channels
    metric = []
    for value in hidden:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][0])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Number of hidden channels')
    axs.set_ylabel('$R^2$')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('mediumpurple')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('mediumpurple')
    vp['cmedians'].set_linewidth(2)

    axs.set_xticks(ticks=[1, 2, 3, 4], labels=['$32$', '$64$', '$128$', '$256$'])

    plt.tight_layout()
    plt.savefig(output_name + '_hidden_r2.pdf')
    plt.close()

    metric = []
    for value in hidden:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][1])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Number of hidden channels')
    axs.set_ylabel('MAE (' + units + ')')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('cornflowerblue')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('cornflowerblue')
    vp['cmedians'].set_linewidth(2)

    axs.set_xticks(ticks=[1, 2, 3, 4], labels=['$32$', '$64$', '$128$', '$256$'])

    plt.tight_layout()
    plt.savefig(output_name + '_hidden_MAE.pdf')
    plt.close()

    metric = []
    for value in hidden:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][2])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Number of hidden channels')
    axs.set_ylabel('RMSE (' + units + ')')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('mediumseagreen')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('mediumseagreen')
    vp['cmedians'].set_linewidth(2)

    axs.set_xticks(ticks=[1, 2, 3, 4], labels=['$32$', '$64$', '$128$', '$256$'])

    plt.tight_layout()
    plt.savefig(output_name + '_hidden_RMSE.pdf')
    plt.close()


    # Dropout
    metric = []
    for value in dropout:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][0])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Dropout')
    axs.set_ylabel('$R^2$')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('mediumpurple')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('mediumpurple')
    vp['cmedians'].set_linewidth(2)

    axs.set_xticks(ticks=[1, 2, 3, 4], labels=['$0.0$', '$0.2$', '$0.4$', '$0.6$'])

    plt.tight_layout()
    plt.savefig(output_name + '_dropout_r2.pdf')
    plt.close()

    metric = []
    for value in dropout:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][1])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Dropout')
    axs.set_ylabel('MAE (' + units + ')')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('cornflowerblue')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('cornflowerblue')
    vp['cmedians'].set_linewidth(2)

    axs.set_xticks(ticks=[1, 2, 3, 4], labels=['$0.0$', '$0.2$', '$0.4$', '$0.6$'])

    plt.tight_layout()
    plt.savefig(output_name + '_dropout_MAE.pdf')
    plt.close()

    metric = []
    for value in dropout:
        metric_sub = []
        for val in range(len(value)):
            metric_sub.append(value[val][2])
        
        metric.append(metric_sub)
    
    _, axs = plt.subplots(figsize=(4, 3))

    axs.set_xlabel('Dropout')
    axs.set_ylabel('RMSE (' + units + ')')

    vp = axs.violinplot(metric, showmeans=False, showmedians=True, showextrema=False)

    for body in vp['bodies']:
        body.set_facecolor('mediumseagreen')
        body.set_alpha(0.5)

    vp['cmedians'].set_color('mediumseagreen')
    vp['cmedians'].set_linewidth(2)

    axs.set_xticks(ticks=[1, 2, 3, 4], labels=['$0.0$', '$0.2$', '$0.4$', '$0.6$'])

    plt.tight_layout()
    plt.savefig(output_name + '_dropout_RMSE.pdf')
    plt.close()






model_name_list, r2_list, MAE_list, MSE_list = get_data('bandgap/model_performance.txt')
plot_violins(model_name_list, r2_list, MAE_list, MSE_list, 'eV', 'bandgap')

model_name_list, r2_list, MAE_list, MSE_list = get_data('fermi/model_performance.txt')
plot_violins(model_name_list, r2_list, MAE_list, MSE_list, 'eV', 'fermi')
