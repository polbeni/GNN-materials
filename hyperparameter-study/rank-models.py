# Pol Benítez Colominas, May 2025
# Universitat Politècnica de Catalunya

# Rank models after a hyperparameter study based on the obtained metrics

### Parameters
models = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6']   # CGCNN model
learning_rates = ['1e-2', '1e-3', '1e-4', '1e-5', '1e-6']               # Value of the learning rate step
hidden_array = ['32', '64', '128', '256']                               # Number of hidden channels in the convolutional layers
dropout_array = ['00', '02', '04', '06']                                # Fraction of elements to dropout

outputs_dir = 'outputs_file/'                                           # Path to dir where outputs are saved


### Evaluate the models and rank them by their metrics
model_candiates = []

for model_it in models:
    for lr_it in learning_rates:
        for hidden_it in hidden_array:
            for dropout_it in dropout_array:
                model_metrics = []

                path_to_model = model_it + '_' + lr_it + '_' + hidden_it + '_' + dropout_it

                model_metrics.append(path_to_model)

                with open(path_to_model + '/' + outputs_dir + 'metrics_model.txt', 'r') as file:
                    for _ in range(10):
                        file.readline()

                    line = file.readline()
                    model_metrics.append(float(line.split()[1]))

                    line = file.readline()
                    model_metrics.append(float(line.split()[1]))

                    line = file.readline()
                    model_metrics.append(float(line.split()[1]))

                    line = file.readline()
                    model_metrics.append(float(line.split()[2]))

                    line = file.readline()
                    model_metrics.append(float(line.split()[3]))

                model_candiates.append(model_metrics)

sorted_candidates = sorted(model_candiates, key=lambda x: x[3], reverse=True)


### Save the list with candidates
with open('model_performance.txt', 'w') as file:
    file.write('Model        r2        MAE        MSE        max_error        epoch_min_loss\n')

    for model_num in range(len(sorted_candidates)):
        file.write(f'{sorted_candidates[model_num][0]}        {sorted_candidates[model_num][3]}        {sorted_candidates[model_num][2]}        {sorted_candidates[model_num][1]}        {sorted_candidates[model_num][4]}        {sorted_candidates[model_num][5]}\n')