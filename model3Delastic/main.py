from data.parser import Parser
from data.dataset import Dataset
from model.model import UNet3D, save_model
from model.training import TrainerTester

def main():
    parser = Parser()
    print("Arguments of parser are: ", parser.get_args())
    print("Parser is: ", parser.get_parser())
    print("Number of samples: ", parser.get_n_samples())
    print("Number of epochs: ", parser.get_n_epochs())
    print("Stress number: ", parser.get_stress_number())
    print("Retrain: ", parser.get_retrain())
    print("Model name: ", parser.get_model_name())
    print("Data path: ", parser.get_data_path())
    
    dataset = Dataset(parser.get_data_path(), n_samples=parser.get_n_samples(), stress_number=parser.get_stress_number(), load_number=0, augment=0)
    print("Data path of dataset is: ", dataset.data_path)
    print("Device of dataset is: ", dataset.device)
    print("Number of samples of dataset is: ", dataset.n_samples)
    print("Stress number of dataset is: ", dataset.stress_number)
    print("Load number of dataset is: ", dataset.load_number)
    print("Maxfile of dataset is: ", dataset._maxfile)
    
    if dataset.augment:
        dataset.augmentate()
        print("Augmentation of dataset is done.")
    dataset.load_data()
    print("Data of dataset is loaded.")
    
    print("Length of dataset is: ", len(dataset))
    print("Item 0 of dataset is: ", dataset[0])
    
    model = UNet3D()
    
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