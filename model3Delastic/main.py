from data.parser import Parser
from data.dataset import Dataset
from model.model import UNet3D, save_model
from model.training import TrainerTester

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
    
    dataset = Dataset(parser.get_data_path(), n_samples=parser.get_n_samples(), stress_number=parser.get_stress_number(), load_number=0, augment=0)

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
    
    del dataset
        
if __name__ == '__main__':
    main()