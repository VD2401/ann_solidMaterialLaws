import torch
import os
from numpy import ceil

def _data_key(i: int, l, s) -> str:
    return "data_elasticity_3D_128_" + str(i) + "_L" + str(l) + "_S" + str(s) + "_input.pt"

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path='data/data_files/',
                 n_samples=128,
                 stress_number=0,
                 load_number=0,
                 augment=0,
                 keep_prep=0,
                 resolution=64):
        # Assume number of files is 1, if not, we will adapt
        # 128 samples per file
        m = n_samples//128
        
        # Adapt if more is available
        while os.path.exists(data_path + _data_key(m, load_number, stress_number)):
            m += 1
            
        # We save the number of datafiles available.
        self.number_of_files = m

        # store the indicator of how many files we use for data augmentation
        self.augment = int(augment)
        # if not specified, we use all the data available
        self.n_samples = n_samples

        self.data_path = data_path

        # adapt the device to the available hardware
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print("Device: ", self.device)

        # Specify the stress number and the load number
        self.stress_number = stress_number
        self.load_number = load_number
        
        self.resolution = resolution

        # this indicator will be used to delete the created datafiles if the preprocessed data is not kept
        self.keep_prep = keep_prep

        # We augment the data if not 0 file augmentation
        if augment:
            self.total_samples = 6*self.n_samples #TODO define a number of augmentations
        else:
            self.total_samples = self.n_samples

        self.input = torch.zeros((0, 1, resolution, resolution, resolution), device=self.device)
        self.output = torch.zeros((0, 1, resolution, resolution, resolution), device=self.device)


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx < self.total_samples:
            return self.input[idx], self.output[idx]
        else:
            raise IndexError("Index out of range")

    def load_data(self):
        # For each file we add the data
        for i in range(int(ceil(self.n_samples/128))):
            # datapath 
            key_input = self.data_path + _data_key(i, self.load_number, self.stress_number)
            key_output = key_input.replace('input', 'output')
            print(key_input)
            print(key_output)
            
            if os.path.exists(key_input) and os.path.exists(key_output):
                # load the data
                input = torch.load(key_input, map_location=self.device)['input']
                output = torch.load(key_output, map_location=self.device)['output']
                
                # add to the dataset
                self.input = torch.cat((self.input, input), 0)
                self.output = torch.cat((self.output, output), 0)
            else:
                raise ValueError("Data not preprocessed")
            
            # we only keep the necessary data
            self.input = self.input[:self.n_samples]
            self.output = self.output[:self.n_samples]
        print(f"After " + str(self.augment) + f" augmentation the input & output size is {self.input.size()}")

    def __del__(self):
        del self.input
        del self.output
        print("Deletion of input/output")
        # Optional: automatic deletion of preprocessed data if memory issues
        
        # if (not self.keep_prep) or self.augment: # We delete the data if chosen or if the file were changed due to data_augmentation
        #     print("Deleting files")
        #     for i in range(int(ceil(self.n_samples/128))):
        #         print("Deletion of file ", i+1, " out of ", int(ceil(self.n_samples/128)))
        #         key_input = self.data_path + "data_elasticity_3D_128_" + str(i) + "_input.pt"
        #         key_output = self.data_path + "data_elasticity_3D_128_" + str(i) + "_output.pt"
        #         if os.path.exists(key_input):
        #             print("Deletion of ",key_input)
        #             os.remove(key_input)
        #         if os.path.exists(key_output):
        #             print("Deletion of ",key_output)
        #             os.remove(key_output)

