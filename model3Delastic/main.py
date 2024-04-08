from data.parser import Parser
from data.dataset import Dataset
from model.model import UNet3D, save_model
from model.training import TrainerTester
import os
import csv

def main():
    parser = Parser()
    print(f"Arguments of parser are: {parser.get_args()}")
    print(f"Number of samples: {parser.get_n_samples()}")
    print(f"Number of epochs: {parser.get_n_epochs()}")
    print(f"Stress number: {parser.get_stress_number()}")
    print(f"Load number: {parser.get_load_number()}")
    print(f"Augment: {parser.get_augment()}")
    print(f"Model name: {parser.get_model_name()}")
    print(f"Data path: {parser.get_data_path()}\n")
    
    dataset = Dataset(parser.get_data_path(),\
                        n_samples=parser.get_n_samples(),\
                        stress_number=parser.get_stress_number(),\
                        load_number=parser.get_load_number(),\
                        augment=parser.get_augment())

    print(f"Dataset is created with {dataset.n_samples} samples, stress number {dataset.stress_number} and load number {dataset.load_number}.\n")
    if dataset.augment:
        dataset.augmentate()
        print("Augmentation of dataset is done.")
    dataset.load_data()
    print("Data of dataset is loaded.")
    print(f"Lenght of dataset: {len(dataset)}\n")
    
    model = UNet3D()
    print(f"Type of model: UNet3D with activation function LeakyReLU.")
    
    training = TrainerTester(model, dataset)
    print("Training infrastructure is set.")
    
    print("Start training.")
    training.train(n_epochs=parser.get_n_epochs()) # Train the model
    print("End of training. Number of epochs: ", training.epochs)
    # save the model
    save_key = f"N{dataset.n_samples}_stress{dataset.stress_number}_loading{dataset.load_number}"
    save_model(model, save_key=save_key, epochs=training.epochs)
    
    results =  [str(parser.get_n_samples()),
                str(parser.get_n_epochs()),
                str(parser.get_stress_number()),
                str(parser.get_load_number()),
                str(parser.get_augment()),
                str(parser.get_model_name()),
                str(parser.get_data_path()),
                'maxLoss_testing',
                str(training.epochs),
                'maxLoss_training',
                'true_samples']
    #write to results.csv
    if not os.path.exists('results.csv'):
        fields = ['n_samples',
                'input_epochs',
                'stress_number',
                'load_number' ,
                'augment_files',
                'model_name' ,
                'data_path',
                'maxLoss_testing',
                'output_epochs',
                'maxLoss_training',
                'true_samples']
        
        with open('results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            file.close()
    with open('results.csv', 'a', newline='') as file:
        writer = csv.DictWriter(file)
        writer.writerow(results)
        file.close()
    
    
    
    
        
    
    del dataset
        
if __name__ == '__main__':
    main()