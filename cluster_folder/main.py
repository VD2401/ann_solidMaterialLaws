import torch
from numpy import ceil, floor
import time
import mlflow
import sys

from dataset import Dataset
from model import UNet3D
from training import TrainerTester

# %%
#DATAPATH = '../model3Delastic/data/data_files/' # temporary
DATAPATH = 'data_files/' 

# EXPERIMENT_NAME = 'test1-best_augmentation_fraction'
EXPERIMENT_NAME = 'test2-best_augmentation_fraction'
# EXPERIMENT_NAME = 'test1-best_n_sample'
# EXPERIMENT_NAME = 'test2-best_n_sample'
# EXPERIMENT_NAME = 'test-best_batch_size'
# EXPERIMENT_NAME = 'test-best_learning_rate'

# Set the experiment
mlflow.set_experiment(EXPERIMENT_NAME)
print('The experiment ', EXPERIMENT_NAME, ' is set. Experiment ID: ', mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id)

STRESS_NUMBER = 0
LOAD_NUMBER = 0

## TESTING AUGMENTATION FRACTIONS
# if contain best_augmentation_fraction in the name
if 'best_augmentation_fraction' in EXPERIMENT_NAME:
    n_sample = 512
    batch_size = 64
    dict_augmentation_mode = {
        0: 'no augmentation',
        1: 'rotation in x',
        2: 'rotation in y',
        3: 'rotation in z',
        4: 'flip in y,z',
        5: 'flip in y,z only',
    }
    for seed in [3, 4, 5]:
        for augmentation_mode, augmentation_type in dict_augmentation_mode.items():
            if seed == 3 and (augmentation_mode in [0, 1, 2]):
                continue
            starting_time = time.time()
            
            print(f"\n\n——————————————————————\nFor number of samples: {n_sample}\n——————————————————————\n")
            print(f"Augmentation mode: {augmentation_type}")
            first_round = True if augmentation_mode == 3 else False
            
            if first_round:
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
            
            # run_name with n_sample
            with mlflow.start_run(run_name="{}".format(augmentation_type)):
                print("The run is launched. Run ID: ", mlflow.active_run().info.run_id)
                training = TrainerTester(model, dataset, seed=seed, augmentation_mode= augmentation_mode, batch_size= batch_size)
                # FOR DEBUGGING
                training.max_epochs = 200
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
                    "augmentation_mode": augmentation_type,
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

