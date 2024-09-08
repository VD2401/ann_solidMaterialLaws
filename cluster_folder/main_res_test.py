import torch
from numpy import ceil, floor
import time
import mlflow
from os import path
from operator import itemgetter

from dataset import Dataset
from model import UNet3D
from training import TrainerTester

# %%
#DATAPATH = '../model3Delastic/data/data_files/' # temporary
DATAPATH = 'data_files/' 

EXPERIMENT_NAME = 'test-resolution'

# Set the experiment
mlflow.set_experiment(EXPERIMENT_NAME)
print('The experiment ', EXPERIMENT_NAME, ' is set. Experiment ID: ', mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id)

STRESS_NUMBER = 0
LOAD_NUMBER = 0

DATAPATH = "data_files128/"

starting_time = time.time()
n_sample = 128
batch_size = 16
print(f"\n\n——————————————————————\nFor number of samples: {n_sample}\n——————————————————————\n")

dataset = Dataset(DATAPATH,\
                    n_samples=n_sample,\
                    stress_number=STRESS_NUMBER,\
                    load_number=LOAD_NUMBER,\
                    resolution=128
                    )

print(f"Dataset is created with {dataset.n_samples} samples, stress number {dataset.stress_number} and load number {dataset.load_number}.\n")

key_dataset = DATAPATH + "data_elasticity_3D_res128.pt"
if path.exists(key_dataset):
    input, output = itemgetter('young_modulus', 'stress')(torch.load(key_dataset, map_location=dataset.device))
    input, output = input.unsqueeze(1), output.unsqueeze(1)
    dataset.input = input
    dataset.output = output
else:
    raise ValueError("Data not preprocessed")
print(f"After " + str(dataset.augment) + f" augmentation the input & output size is {dataset.input.size()}")

print("Dataset loaded.")
print(f"Lenght of dataset: {len(dataset)}\n")

model = UNet3D()
print(f"Type of model: UNet3D with activation function LeakyReLU.")

# run_name with n_sample
with mlflow.start_run(run_name="Resolution = 128"):
    print("The run is launched. Run ID: ", mlflow.active_run().info.run_id)
    training = TrainerTester(model, dataset, seed=1, batch_size=batch_size)
    # FOR DEBUGGING
    training.max_epochs = 800
    print(f"Batch size is set to {training.batch_size}. Maximum number of epochs is set to {training.max_epochs}.")
    #training.max_epochs = 120
    print("Training infrastructure is set.")
    
    params = {
        "n_samples": dataset.n_samples,
        "stress_number": dataset.stress_number,
        "load_number": dataset.load_number,
        "device": torch.cuda.get_device_name() if dataset.device.type == 'cuda' else dataset.device.type,
        "seed": training.seed,
        "criterion": training.criterion.__class__.__name__,
        "learning_rate": training.lr,
        "optimizer": training.optimizer.__class__.__name__,
        "split_ratio": training.split_ratio,
        "batch_size": training.batch_size,
        "stop_criteria": training.stop_criteria,
        "max_epochs": training.max_epochs,
        "model_name": model.__class__.__name__,
        "activation": model.activation.__class__.__name__,
        "depth": model.depth,
        "retrained_model": model.retrain,
        "width1": model.width1,
        "width2": model.width2,
        "width3": model.width3,
        "width4": model.width4,
        "width5": model.width5,
        "kernel_size_conv1": model.kernel_size_conv1,
        "kernel_size_conv2": model.kernel_size_conv2,
        "stride_conv1": model.stride_conv1,
        "stride_conv2": model.stride_conv2,
        "padding_conv1": model.padding_conv1,
        "padding_conv2": model.padding_conv2,
        "padding_mode_conv1": model.padding_mode_conv1,
        "padding_mode_conv2": model.padding_mode_conv2,
        "kernel_size_pool": model.kernel_size_pool,
        "stride_pool": model.stride_pool,
        "resolution": 128
    }
    
    mlflow.log_params(params)
    print("Parameters are logged.")
    
    print("Start training.")
    training.train() # Train the model
    print("End of training. Number of epochs: ", training.epochs)
    
    mlflow.log_param("last_epoch", training.epochs)
    mlflow.log_param("total_training_examples", training.epochs*n_sample)
    mlflow.log_metric("training_time", sum(training.training_time))
    mlflow.log_metric("testing_time", sum(training.testing_time))
    mlflow.log_metric("training_indicator", training.training_indicator[-1])
    
    end_time = time.time()
    mlflow.log_metric("total_time", end_time - starting_time)
    mlflow.log_param("mean_time_per_epoch", (end_time - starting_time)/training.epochs)
    print("Total time: ", end_time - starting_time)
    

starting_time = time.time()
n_sample = 128
batch_size = 16
print(f"\n\n——————————————————————\nFor number of samples: {n_sample}\n——————————————————————\n")

dataset = Dataset(DATAPATH,\
                    n_samples=n_sample,\
                    stress_number=STRESS_NUMBER,\
                    load_number=LOAD_NUMBER,\
                    )

print(f"Dataset is created with {dataset.n_samples} samples, stress number {dataset.stress_number} and load number {dataset.load_number}.\n")

dataset.load_data()
print("Dataset loaded.")
print(f"Lenght of dataset: {len(dataset)}\n")

model = UNet3D()
print(f"Type of model: UNet3D with activation function LeakyReLU.")

with mlflow.start_run(run_name="Resolution = 64"):
    print("The run is launched. Run ID: ", mlflow.active_run().info.run_id)
    training = TrainerTester(model, dataset, seed=1, batch_size=batch_size)
    training.max_epochs = 800
    print(f"Batch size is set to {training.batch_size}. Maximum number of epochs is set to {training.max_epochs}.")
    # FOR DEBUGGING
    #training.max_epochs = 100
    print("Training infrastructure is set.")
    
    params = {
        "n_samples": dataset.n_samples,
        "stress_number": dataset.stress_number,
        "load_number": dataset.load_number,
        "device": torch.cuda.get_device_name() if dataset.device.type == 'cuda' else dataset.device.type,
        "seed": training.seed,
        "criterion": training.criterion.__class__.__name__,
        "learning_rate": training.lr,
        "optimizer": training.optimizer.__class__.__name__,
        "split_ratio": training.split_ratio,
        "batch_size": training.batch_size,
        "stop_criteria": training.stop_criteria,
        "max_epochs": training.max_epochs,
        "model_name": model.__class__.__name__,
        "activation": model.activation.__class__.__name__,
        "depth": model.depth,
        "retrained_model": model.retrain,
        "width1": model.width1,
        "width2": model.width2,
        "width3": model.width3,
        "width4": model.width4,
        "width5": model.width5,
        "kernel_size_conv1": model.kernel_size_conv1,
        "kernel_size_conv2": model.kernel_size_conv2,
        "stride_conv1": model.stride_conv1,
        "stride_conv2": model.stride_conv2,
        "padding_conv1": model.padding_conv1,
        "padding_conv2": model.padding_conv2,
        "padding_mode_conv1": model.padding_mode_conv1,
        "padding_mode_conv2": model.padding_mode_conv2,
        "kernel_size_pool": model.kernel_size_pool,
        "stride_pool": model.stride_pool,
        "resolution": 64
    }
    
    mlflow.log_params(params)
    print("Parameters are logged.")
    
    print("Start training.")
    training.train() # Train the model
    print("End of training. Number of epochs: ", training.epochs)
    
    mlflow.log_param("last_epoch", training.epochs)
    mlflow.log_param("total_training_examples", training.epochs*n_sample)
    mlflow.log_metric("training_time", sum(training.training_time))
    mlflow.log_metric("testing_time", sum(training.testing_time))
    mlflow.log_metric("training_indicator", training.training_indicator[-1])
    
    end_time = time.time()
    mlflow.log_metric("total_time", end_time - starting_time)
    mlflow.log_param("mean_time_per_epoch", (end_time - starting_time)/training.epochs)
    print("Total time: ", end_time - starting_time)
    
