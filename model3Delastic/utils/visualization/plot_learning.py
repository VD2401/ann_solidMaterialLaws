import matplotlib
import matplotlib.pyplot as plt
import torch

OUTPUT_DIR = 'test/test_parameters/training/'

def learning(training_indicator, testing_indicator, training_time, save_key, **kwargs):
    
    data_min = kwargs.get('vmin', 0)
    data_max = kwargs.get('vmax', max([max(training_indicator), max(testing_indicator)]))

    plt.figure(figsize=(10, 6), dpi=200)

    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams.update({'font.size': 12})
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    plt.plot(range(1, len(testing_indicator) + 1), training_indicator, label='Training error', linewidth=1, markersize=7, marker='+')
    plt.plot(range(1, len(testing_indicator) + 1), testing_indicator, label='Testing error', linewidth=1, markersize=7, marker='+')

    plt.xlabel('Epochs')
    plt.xlim([0, len(testing_indicator) + 1])

    plt.yscale('log')
    plt.ylabel('Mean Absolute Error')
    plt.ylim([data_min, data_max])

    plt.grid()
    plt.legend()
    plt.annotate('Epoch duration = ' + str(round(training_time, 2)) + ' s',
                 (0.05 * (len(testing_indicator) + 1), 0.95 * max([max(training_indicator), max(testing_indicator)])))
    
    # save figure
    plt.savefig(OUTPUT_DIR + 'training_' +save_key + '.png')
    
    plt.close()