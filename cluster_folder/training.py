import torch
import time
import mlflow

from model import UNet3D
from augmentation import augment1, augment2, augment3, augment4, augment5, augment6
from dataset import Dataset
from visualization import plot_ITOE, volume

class TrainerTester:
    def __init__(self, model: UNet3D, dataset: Dataset, seed=0, augmentation_mode = 0, batch_size=8) -> None:
        # Set the model, the dataset and the device without initialization
        self.model = model
        self.dataset = dataset
        self.device = self.dataset.device
        self.model.to(self.device)

        # TODO: set these parameters from a configuration file
        self.criterion = torch.nn.MSELoss()
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.split_ratio = 0.75
        self.batch_size = batch_size
        self.stop_criteria = 0.02
        self.max_epochs = int(500*128/self.dataset.n_samples) # Set the same computing time for each dataset

        self.training_time = list()
        self.testing_time = list()
        self.training_indicator = list()
        self.testing_indicator = list()
        self.epochs = 0
        
        self.seed = self.set_seed(seed)
        
        self.augmentation_mode = augmentation_mode

        self.train_loader, self.test_loader = self.split_data()

    def split_data(self):
        split = int(len(self.dataset) * self.split_ratio)

        train_inputs, test_inputs = self.dataset.input[:split], self.dataset.input[split:]
        train_outputs, test_outputs = self.dataset.output[:split], self.dataset.output[split:]
        
        train_inputs, train_outputs = self.augmentation(input=train_inputs, output=train_outputs)

        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_outputs)
        test_dataset = torch.utils.data.TensorDataset(test_inputs, test_outputs)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

# train function
    def train_loop(self):
        indicator = torch.zeros((len(self.train_loader),))
        self.model.train()
        for batch, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            mlflow.log_metric("loss", loss.item(), step=self.epochs)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            indicator[batch] = torch.abs(output - target).mean()
        train_indicator = indicator.mean().item()
        mlflow.log_metric("training MAE", train_indicator, step=self.epochs)
        return train_indicator

# test function
    def test_loop(self):
        test_MAE = torch.zeros((len(self.test_loader),))
        test_RMSE = torch.zeros((len(self.test_loader),))
        self.model.eval()
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_MAE[batch] = torch.abs(output - target).mean()
                test_RMSE[batch] = torch.sqrt(torch.mean((output - target)**2))
        test_indicator = test_MAE.mean().item()
        test_RMSE = test_RMSE.mean().item()
        mlflow.log_metric("testing MAE", test_indicator, step=self.epochs)
        mlflow.log_metric("testing RMSE", test_RMSE, step=self.epochs)
        return test_indicator

    def train(self):
        self.epochs = 0
        while (self.epochs < self.max_epochs):
            
            self.epochs += 1
            
            start = time.time()
            train_indicator = self.train_loop()
            self.training_indicator.append(train_indicator)
            self.training_time.append(time.time() - start)

            start = time.time()
            test_indicator = self.test_loop()
            self.testing_indicator.append(test_indicator)
            self.testing_time.append(time.time() - start)

            print(f"Epoch {self.epochs} - Training Loss: {train_indicator:.4f} - Testing Loss: {test_indicator:.4f}")

            # learning(self.training_indicator, self.testing_indicator, self.training_time[-1])

            if (self.testing_indicator[-1] < self.stop_criteria) or (self.epochs == self.max_epochs):
                # self.save_training_visualization()
                mlflow.pytorch.log_model(self.model, "model@ep" + str(self.epochs))
                break
            if (self.epochs == int(self.max_epochs/4) or self.epochs == int(self.max_epochs/2) or self.epochs == int(3*self.max_epochs/4)):
                # self.save_training_visualization()
                mlflow.pytorch.log_model(self.model, "model@ep" + str(self.epochs))
            
            # End training if the loss explodes
            if self.training_indicator[-1] > 10:
                print("The training loss exploded. The training is stopped.")
                break

        print("Total training time: ", sum(self.training_time))

    def get_max_loss_training(self):
        if len(self.training_indicator) == 0:
            print("No training indicator computed")
            return
        indicator = torch.zeros((len(self.train_loader),))
        print("Computing the maximum loss for training")
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                indicator[batch] = torch.abs(output - target).mean()

        index = torch.argmax(indicator)
        return index, indicator.flatten()[index].item()

    def get_max_loss_testing(self):
        if len(self.testing_indicator) == 0:
            print("No testing indicator computed")
            return
        indicator = torch.zeros((len(self.test_loader),))
        print("Computing the maximum loss for testing")
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                indicator[batch] = torch.abs(output - target).mean()

        index = torch.argmax(indicator)
        return index, indicator.flatten()[index].item()

    def save_training_visualization(self):
        index, indicator = self.get_max_loss_testing()

        input = self.test_loader.dataset[index][0]
        target = self.test_loader.dataset[index][1]
        output = self.model(input.to(self.device).unsqueeze(0)).detach()

        x, y, z = torch.arange(64), torch.arange(64), torch.arange(64) # create grid
        volume(x ,y ,z , input[0, ...], data_type='input', epoch=self.epochs, vim=0, vmax=3.5)
        volume(x ,y ,z , target[0, ...], data_type='target', epoch=self.epochs, vim=0, vmax=3.5)
        volume(x ,y ,z , output[0, 0, ...], data_type='output', epoch=self.epochs, vim=0, vmax=3.5)
        print(f"The maximum loss is {indicator:.4f} at index {index}. The figure displays the input, target and output of this worst sample.")

        plot_ITOE(input[0, ...], target[0, ...], output[0, 0, ...], index, 0, self.epochs)

    def get_mean_loss_training(self):
        if len(self.training_indicator) == 0:
            print("No training indicator computed")
            return
        indicator = torch.zeros((len(self.train_loader), 1))
        print("Computing the maximum loss for training")
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                indicator[batch, :] = torch.abs(output - target).mean()
        return indicator.mean().item()


    def get_mean_loss_testing(self):
        if len(self.testing_indicator) == 0:
            print("No testing indicator computed")
            return
        indicator = torch.zeros((len(self.test_loader), 1))
        print("Computing the maximum loss for testing")
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                indicator[batch, :] = torch.abs(output - target).mean()
        return indicator.mean().item()
    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        print("Seed changed to ", seed)
        return seed

    def augmentation(self, input, output):
        # The augmentation doubles according to the required augmentation type
        
        new_inputs, new_outputs = input, output
        
        # Only add the flip in yz
        if self.augmentation_mode == 5:
            input_flip_in_yz, output_flip_in_yz = augment4(input, output)
            return torch.cat((input, input_flip_in_yz), dim=0), torch.cat((output, output_flip_in_yz), dim=0)
        
        # Add the the rotation in x
        if self.augmentation_mode:
            input_rotated_in_x, output_rotated_in_x = augment1(input, output)
            new_inputs, new_outputs = torch.cat((input, input_rotated_in_x), dim=0), torch.cat((output, output_rotated_in_x), dim=0)
        
        # Add the rotation in y
        if self.augmentation_mode > 1:
            input_rotated_in_y, output_rotated_in_y = augment2(input, output)
            new_inputs, new_outputs = torch.cat((new_inputs, input_rotated_in_y), dim=0), torch.cat((new_outputs, output_rotated_in_y), dim=0)
        
        # Add the rotation in z
        if self.augmentation_mode > 2:
            input_rotated_in_z, output_rotated_in_z = augment3(input, output)
            new_inputs, new_outputs = torch.cat((new_inputs, input_rotated_in_z), dim=0), torch.cat((new_outputs, output_rotated_in_z), dim=0)
        
        # Add the flip in yz
        if self.augmentation_mode > 3:
            input_flip_in_yz, output_flip_in_yz = augment4(input, output)
            new_inputs, new_outputs = torch.cat((new_inputs, input_flip_in_yz), dim=0), torch.cat((new_outputs, output_flip_in_yz), dim=0)
        
        
        return new_inputs, new_outputs