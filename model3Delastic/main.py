from data.parser import Parser
from data.dataset import Dataset
from model.model import UNet3D, save_model, UNet3D_2
from model.training import TrainerTester
import os
import csv

def main():
    N_SAMPLES = [16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
    DATA_PATH = 'data/data_files/'
    STRESS_NUMBER = 0
    LOAD_NUMBER = 0
    AUGMENT = False
    
    for n_sample in N_SAMPLES:
        print(f"\n\n——————————————————————\nFor number of samples: {n_sample}\n——————————————————————\n")
        # parser = Parser()
        # print(f"Arguments of parser are: {parser.get_args()}")
        # print(f"Number of samples: {parser.get_n_samples()}")
        # print(f"Number of epochs: {parser.get_n_epochs()}")
        # print(f"Stress number: {parser.get_stress_number()}")
        # print(f"Load number: {parser.get_load_number()}")
        # print(f"Augment: {parser.get_augment()}")
        # print(f"Model name: {parser.get_model_name()}")
        # print(f"Data path: {parser.get_data_path()}\n")
        
        # dataset = Dataset(parser.get_data_path(),\
        #                     n_samples=parser.get_n_samples(),\
        #                     stress_number=parser.get_stress_number(),\
        #                     load_number=parser.get_load_number(),\
        #                     augment=parser.get_augment())
        
        dataset = Dataset(DATA_PATH,\
                            n_samples=n_sample,\
                            stress_number=STRESS_NUMBER,\
                            load_number=LOAD_NUMBER,\
                            augment=AUGMENT)

        print(f"Dataset is created with {dataset.n_samples} samples, stress number {dataset.stress_number} and load number {dataset.load_number}.\n")
        if dataset.augment:
            dataset.augmentate()
            print("Augmentation of dataset is done.")
        dataset.load_data()
        print("Dataset loaded.")
        print(f"Lenght of dataset: {len(dataset)}\n")
        
        model = UNet3D()
        print(f"Type of model: UNet3D with activation function LeakyReLU.")
        
        training = TrainerTester(model, dataset)
        print("Training infrastructure is set.")
        
        print("Start training.")
        training.train() # Train the model
        print("End of training. Number of epochs: ", training.epochs)
        # save the model
        save_key = f"N{dataset.n_samples}_stress{dataset.stress_number}_loading{dataset.load_number}"
        save_model(model, save_key=save_key, epochs=training.epochs)
        
        
        del dataset
        
if __name__ == '__main__':
    main()