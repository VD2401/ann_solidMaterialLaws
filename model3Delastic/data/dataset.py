import torch
import os
from . import rotate
from numpy import ceil

def _data_key(i: int) -> str:
    return "data_elasticity_3D_128_" + str(i) + ".pt"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, n_samples=None, stress_number=0, load_number=0, augment=0):
        # Check that the data path is correct.
        if not os.path.exists(data_path + _data_key(0)):
            raise ValueError('The data path must lead to a folder with at least one file"' + _data_key(0))
        m = 1
        
        # We save the number of datafiles available
        while True and not augment:
            if not os.path.exists(data_path + _data_key(m)): 
                break
            m += 1
        
        # TODO: change max file to nb file in functions
        # store the indicator of how many files we use for data augmentation
        self.augment = int(augment)
        # if not specified, we use all the data available
        if n_samples is None:
            self.n_samples = self._maxfile*128
            self._maxfile = self.augment if self.augment else m # if augment is 0/False, then self._maxfile = m 
        else:
            self.n_samples = n_samples
            self._maxfile = self.augment if self.augment else int(ceil(self.n_samples/128)) # if augment is 0/False, then self._maxfile = m 
        
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
        
        print(type(self.n_samples))  # Should be <class 'int'>
        print(self.n_samples)  # Should be a positive integer

        print(type(self.device))  # Should be <class 'torch.device'>
        print(self.device)  # Should be something like device(type='cuda', index=0)
        # We create two tensors to store the input and the output
        
        # this indicator will be used to delete the created datafiles if the preprocessed data is not kept
        self.keep_prep = False
        
        # We augment the data if not 0 file augmentation
        self.total_samples = 6*self.n_samples if augment else self.n_samples
        
        self.input = torch.zeros((0, 1, 64, 64, 64), device=self.device)
        self.output = torch.zeros((0, 1, 64, 64, 64), device=self.device)
        
        # We preprocess the data
        self.preprocessing()
        
        # Need for augmentation
        # Need for loading
            
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.total_samples > idx:
            return self.input[idx], self.output[idx]
        else: 
            return None, None
    
    def preprocessing(self):
        # Load every datafiles and create two new files. One for the input, the other for the output.
        for i in range(self._maxfile):
            
            print("Preprocessing of file " + str(i+1) + " out of " + str(self._maxfile))
            key_input = self.data_path + "data_elasticity_3D_128_" + str(i) + "_input.pt"
            key_output = self.data_path + "data_elasticity_3D_128_" + str(i) + "_output.pt"
            
            if os.path.exists(key_input) and os.path.exists(key_output):
                print("Data already preprocessed")
                self.keep_prep = True
                continue
            else:
                n = min(128, self.n_samples - i*128)
                print("Value of n: ", n, "Value of i: ", i)
                data = torch.load(self.data_path + _data_key(i), map_location=self.device)
                print("File " + str(i+1) + " loaded")
                input = data["young_modulus"].view(128, 1, 64, 64, 64)\
                                                .detach().clone().cpu()[:n]
                torch.save({'input': input}, key_input)
                print("Input of file " + str(i+1) + " saved. It corresponds to the young modulus.")
                output = data["stress"].select(5, self.stress_number)\
                                        .select(1, self.load_number)\
                                            .view(128, 1, 64, 64, 64)\
                                                .detach().clone().cpu()[:n]
                torch.save({"output": output}, key_output)
                print("Output of file " + str(i+1) + 
                      " saved. It corresponds to the stress number " +
                      str(self.stress_number) + " and the load number " +
                      str(self.load_number))
                del data # Free memory
    
    def augmentate(self):
        # Augment the data
        for i in range(self._maxfile):
            key_input = self.data_path + "data_elasticity_3D_128_" + str(i) + "_input.pt"
            key_output = self.data_path + "data_elasticity_3D_128_" + str(i) + "_output.pt"
            if os.path.exists(key_input) and os.path.exists(key_output):
                input0 = torch.load(key_input, map_location=self.device)['input']
                output0 = torch.load(key_output, map_location=self.device)['output']
                input = torch.cat((input0, rotate.rotate180(input0, 2)), 0)
                output = torch.cat((output0, rotate.rotate180(output0, 2)), 0)
                input = torch.cat((input, rotate.rotate180(input0, 3)), 0)
                output = torch.cat((output, rotate.rotate180(output0, 3)), 0)
                input = torch.cat((input, torch.rot90(input0, k=1, dims=(2, 3))), 0)
                output = torch.cat((output, torch.rot90(input0, k=1, dims=(2, 3))), 0)
                input = torch.cat((input, torch.rot90(input0, k=2, dims=(2, 3))), 0)
                output = torch.cat((output, torch.rot90(input0, k=2, dims=(2, 3))), 0)
                input = torch.cat((input, torch.rot90(input0, k=3, dims=(2, 3))), 0)
                output = torch.cat((output, torch.rot90(input0, k=3, dims=(2, 3))), 0)
                torch.save({'input': input}, key_input)
                torch.save({'output': output}, key_output)
                print("Input size: ", input.size()) # Size total_samples x 1 x 64 x 64 x 64
                print("Output size: ", output.size()) # Size total_samples x 1 x 64 x 64 x 64
            else:
                raise ValueError("Data not preprocessed")
        
    def load_data(self):
        # Load the data with the correct number of samples
        
        for i in range(self._maxfile):
            key_input = self.data_path + "data_elasticity_3D_128_" + str(i) + "_input.pt"
            key_output = self.data_path + "data_elasticity_3D_128_" + str(i) + "_output.pt"
            if os.path.exists(key_input) and os.path.exists(key_output):
                input = torch.load(key_input, map_location=self.device)['input']
                output = torch.load(key_output, map_location=self.device)['output']
                self.input = torch.cat((self.input, input), 0)
                self.output = torch.cat((self.output, output), 0)
                print("Input size: ", self.input.size()) # Size total_samples x 1 x 64 x 64 x 64
                print("Output size: ", self.output.size()) # Size total_samples x 1 x 64 x 64 x 64
            else:
                raise ValueError("Data not preprocessed")          

    def __del__(self):
        del self.input
        del self.output
        print("Deletion of input/output")
        if (not self.keep_prep) or self.augment: # We delete the data if chosen or if the file were changed due to data_augmentation
            print("Deleting files")
            for i in range(self._maxfile):
                print("Deletion of file ", i+1, " out of ", self._maxfile)
                key_input = self.data_path + "data_elasticity_3D_128_" + str(i) + "_input.pt"
                key_output = self.data_path + "data_elasticity_3D_128_" + str(i) + "_output.pt"
                if os.path.exists(key_input):
                    print("Deletion of ",key_input)
                    os.remove(key_input)
                if os.path.exists(key_output):
                    print("Deletion of ",key_output)
                    os.remove(key_output)
