import os
import csv 

#TODO implement a writer class
def write_results(results):
    #write to results.csv
    if not os.path.exists('results.csv'):
        fields = ['n_samples',
                'epoch',
                'stress_number',
                'load_number' ,
                'augment_files',
                'testing_loss',
                'training_loss',
                'training_time',
                'testing_time']
        
        with open('results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            file.close()
    with open('results.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(results)
        file.close()