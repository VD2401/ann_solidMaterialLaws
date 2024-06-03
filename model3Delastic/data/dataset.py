import torch
import os
from . import rotate
from numpy import ceil

def _data_key(i: int, l, s) -> str:
    return "data_elasticity_3D_128_" + str(i) + "_L" + str(l) + "_S" + str(s) + "_input.pt"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, n_samples=None, stress_number=0, load_number=0, augment=0, keep_prep=0):
        # # Check that the data path is correct.
        # if not os.path.exists(data_path + _data_key(3, load_number, stress_number)):
        #     print(os.listdir(data_path))
        #     raise ValueError('The data path must lead to a folder with at least one file "' + _data_key(3, load_number, stress_number)
        #                      + ' and here the search path is ' + data_path + '"')
        m = 1
        while os.path.exists(data_path + _data_key(m, load_number, stress_number)):
            m += 1
        # We save the number of datafiles available. Files are named data_elasticity_3D_128_i.pt
        self.number_of_files = m
        
        # TODO: change max file to nb file in functions
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

        # this indicator will be used to delete the created datafiles if the preprocessed data is not kept
        self.keep_prep = keep_prep
        
        # We augment the data if not 0 file augmentation
        if augment:
            self.total_samples = 6*self.n_samples #TODO define a number of augmentations
        else:
            self.total_samples = self.n_samples
    
        self.input = torch.zeros((0, 1, 64, 64, 64), device=self.device)
        self.output = torch.zeros((0, 1, 64, 64, 64), device=self.device)
        
            
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if idx < self.total_samples:
            return self.input[idx], self.output[idx]
        else: 
            raise IndexError("Index out of range")
    
    def augmentate(self):
        # Augment the data
        for i in range(int(ceil(self.n_samples/128))):
            key_input = self.data_path + "data_elasticity_3D_128_" + str(i) + "_input.pt"
            key_output = self.data_path + "data_elasticity_3D_128_" + str(i) + "_output.pt"
            
            if os.path.exists(key_input) and os.path.exists(key_output):
                # Original data
                input0 = torch.load(key_input, map_location=self.device)['input']
                output0 = torch.load(key_output, map_location=self.device)['output']
                
                # Rotate the data around the y-axis
                input = torch.cat((input0, rotate.rotate180(input0, 2)), 0)
                output = torch.cat((output0, rotate.rotate180(output0, 2)), 0)
                
                # Rotate the data around the z-axis
                input = torch.cat((input, rotate.rotate180(input0, 3)), 0)
                output = torch.cat((output, rotate.rotate180(output0, 3)), 0)
                
                # Rotate the data around the x-axis for 90 degrees
                input = torch.cat((input, torch.rot90(input0, k=1, dims=(2, 3))), 0)
                output = torch.cat((output, torch.rot90(input0, k=1, dims=(2, 3))), 0)
                
                # Rotate the data around the x-axis for 180 degrees
                input = torch.cat((input, torch.rot90(input0, k=2, dims=(2, 3))), 0)
                output = torch.cat((output, torch.rot90(input0, k=2, dims=(2, 3))), 0)
                
                # Rotate the data around the x-axis for 270 degrees
                input = torch.cat((input, torch.rot90(input0, k=3, dims=(2, 3))), 0)
                output = torch.cat((output, torch.rot90(input0, k=3, dims=(2, 3))), 0)
                
                torch.save({'input': input}, key_input)
                torch.save({'output': output}, key_output)
                print(f"After augmentation the input/output size is {input.size()}") # Size total_samples x 1 x 64 x 64 x 64
            else:
                raise ValueError("Data not preprocessed")
        
    def load_data(self):
        for i in range(int(ceil(self.n_samples/128))):
            key_input = self.data_path + _data_key(i, self.load_number, self.stress_number)
            key_output = key_input.replace('input', 'output')
            print(key_input)
            print(key_output)
            os.listdir(self.data_path)
            if os.path.exists(key_input) and os.path.exists(key_output):
                input = torch.load(key_input, map_location=self.device)['input']
                output = torch.load(key_output, map_location=self.device)['output']
                self.input = torch.cat((self.input, input), 0)
                self.output = torch.cat((self.output, output), 0)
            else:
                raise ValueError("Data not preprocessed")   
            self.input = self.input[:self.n_samples]
            self.output = self.output[:self.n_samples]
        print(f"After " + str(self.augment) + " augmentation the input & output size is {self.input.size()}")

    def __del__(self):
        del self.input
        del self.output
        print("Deletion of input/output")
        if 0:#(not self.keep_prep) or self.augment: # We delete the data if chosen or if the file were changed due to data_augmentation
            print("Deleting files")
            for i in range(int(ceil(self.n_samples/128))):
                print("Deletion of file ", i+1, " out of ", int(ceil(self.n_samples/128)))
                key_input = self.data_path + "data_elasticity_3D_128_" + str(i) + "_input.pt"
                key_output = self.data_path + "data_elasticity_3D_128_" + str(i) + "_output.pt"
                if os.path.exists(key_input):
                    print("Deletion of ",key_input)
                    os.remove(key_input)
                if os.path.exists(key_output):
                    print("Deletion of ",key_output)
                    os.remove(key_output)
