import torch
import time
from model.model import UNet3D
from data import dataset
from utils.visualization import plot_learning, plot_slice, plot_volume


class TrainerTester:
    def __init__(self, model: UNet3D, dataset: dataset.Dataset) -> None:
        # Set the model, the dataset and the device without initialization
        self.model = model
        self.dataset = dataset
        self.device = self.dataset.device
        self.model.to(self.device)
        
        # TODO: set these parameters from a configuration file
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.lr = 1e-3
        self.split_ratio = 0.75
        self.batch_size = 8
        
        self.training_time = list()
        self.testing_time = list()
        self.training_indicator = list()
        self.testing_indicator = list()
        self.epochs = 0
        
        self.train_loader, self.test_loader = self.split_data()
        
    def split_data(self):
        split = int(len(self.dataset) * self.split_ratio)

        train_inputs, test_inputs = self.dataset.input[:split], self.dataset.input[split:]
        train_outputs, test_outputs = self.dataset.output[:split], self.dataset.output[split:]

        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_outputs)
        test_dataset = torch.utils.data.TensorDataset(test_inputs, test_outputs)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

# train function
    def train_loop(self):
        indicator = torch.zeros((len(self.dataset),))
        self.model.train()
        for batch, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            indicator[batch] = torch.abs(output - target).mean()
        return indicator.mean().item()

# test function
    def test_loop(self):
        indicator = torch.zeros((len(self.dataset),))
        indicator = torch.zeros((len(self.test_loader),))
        self.model.eval()
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                indicator[batch] = torch.abs(output - target).mean()
        return indicator.mean().item()
    
    def train(self, n_epochs: int=100):
        epoch = 0
        while (epoch < n_epochs):
            start = time.time()
            train_indicator = self.train_loop()
            self.training_indicator.append(train_indicator)
            self.training_time.append(time.time() - start)
            
            start = time.time()
            test_indicator = self.test_loop()
            self.testing_indicator.append(test_indicator)
            self.testing_time.append(time.time() - start)
            
            print(f"Epoch {epoch+1}/{n_epochs} - Training Loss: {train_indicator:.4f} - Testing Loss: {test_indicator:.4f}")
            
            save_key = f"N{self.dataset.n_samples}_stress{self.dataset.stress_number}_loading{self.dataset.load_number}"
            plot_learning.learning(self.training_indicator, self.testing_indicator, self.training_time[-1], save_key)
            if epoch+1 % 25 == 0 or epoch == n_epochs - 1:
                self.compute_lowest_loss()
            epoch += 1
            # Create a condition if the last ten indicators have less than 1% of decrease then it should be true. It should work even if testing_indicator does not have 10 elements.
            zero_gradient_condition =len(self.testing_indicator) >= 20 and \
                    (0.99*self.testing_indicator[-20:].mean() - self.testing_indicator[-1] > 0)
            if zero_gradient_condition:
                print("ZERO GRADIENT CONDITION. The training stops because the testing indicator has not decreased by more than 1% in the last 20 epochs.")
            if  self.testing_indicator[-1] < 0.02 or zero_gradient_condition:
                self.compute_lowest_loss()
                break
        print("Total training time: ", sum(self.training_time))
        self.epochs = epoch
    
    def get_max_loss_training(self):
        if len(self.training_indicator) == 0:
            print("No testing indicator computed")
            return
        indicator = torch.zeros((len(self.train_loader), 1))
        print("Computing the maximum loss")
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                indicator[batch, :] = torch.abs(output - target).mean()

        index = torch.argmax(indicator)
        print(f"Maximum loss at index {index.item()}")
        return index, indicator[index]
    
    def get_max_loss_testing(self):
        if len(self.testing_indicator) == 0:
            print("No testing indicator computed")
            return
        indicator = torch.zeros((len(self.test_loader), 1))
        print("Computing the maximum loss")
        with torch.no_grad():
            for batch, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                indicator[batch, :] = torch.abs(output - target).mean()

        index = torch.argmax(indicator)
        print(f"Maximum loss at index {index.item()}")
        return index, indicator[index]
    
    def compute_lowest_loss(self):
        index = self.get_max_loss_testing()
        
        input = self.test_loader.dataset[index][0]
        target = self.test_loader.dataset[index][1]
        output = self.model(input.to(self.device).unsqueeze(0)).detach()

        save_key = f"N{self.dataset.n_samples}_stress{self.dataset.stress_number}_loading{self.dataset.load_number}"
        x, y, z = torch.arange(64), torch.arange(64), torch.arange(64) # create grid
        plot_volume.volume(x ,y ,z , input[0, ...], save_key=save_key + '_input')
        plot_volume.volume(x ,y ,z , target[0, ...], vmin=0, vmax=3.5, save_key=save_key + '_target')
        plot_volume.volume(x ,y ,z , output[0, 0, ...], vmin=0, vmax=3.5, save_key=save_key+ '_output')
        print(indicator.flatten()[index].item())

        plot_slice.plot_ITOE(input[0, ...], target[0, ...], output[0, 0, ...], index, 0, save_key=save_key)

