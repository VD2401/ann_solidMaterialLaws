# This class will parse a command to save the parameters of the simulation
# The parameters will be --n_samples, --n_epochs, --stress_number, --retrain, --model_name

import argparse
import os

class Parser:
    def __init__(self, args=None):
        self.parser = argparse.ArgumentParser(description='Parser for the simulation parameters')
        self.parser.add_argument('--n_samples', type=int, help='Number of samples for the simulation. Mandatory.', required=True)
        self.parser.add_argument('--n_epochs', type=int, help='Number of epochs for the simulation. Default is 1000.', default=1000)
        self.parser.add_argument('--stress_number', type=int, help='Number of the stress for the simulation. Default is 0.', default=0)
        self.parser.add_argument('--retrain', type=bool, help='Retrain the model. Default is False.', default=False)
        self.parser.add_argument('--model_name', type=str, help='Name of the model. If --retrain is False: set to None. Else: Mandatory.', default=None)
        self.parser.add_argument('--data_path', type=str, help='Path to the data. Mandatory.', default='data/')
        
        # If from cell or notebook
        if args is not None:
            self.args = vars(self.parser.parse_args(args))
        
        # Save in a dictionary if from command line
        else:
            self.args = vars(self.parser.parse_args())
        
        # Check that the stress number is correct
        if self.args['stress_number'] < 0 or self.args['stress_number'] > 5:
            raise ValueError('The stress number must be between 0 and 5')
        
        # Check that the model name is correct
        if self.args['retrain']:
            if self.args['model_name'] is None:
                raise ValueError('The model name must be set if retrain is True')
            
            # Model must check this format "model_N + n_samples '_stress' + str(stress_number)"
            elif not os.path.exists(self.args['model_name']):
                raise ValueError('The model name must be a valid path; It should suit the format "model_N + n_samples \'_stress\' + str(stress_number)')
            
        # Check that the data path is correct. The data_path should lead to a folder with at least one file "data_elasticity_3D_128_0.pt".
        if not os.path.exists(self.args['data_path'] + 'data_elasticity_3D_128_0.pt'):
            raise ValueError('The data path must lead to a folder with at least one file "data_elasticity_3D_128_0.pt"')
        
        
    def get_args(self):
        return self.args
    
    def get_parser(self):
        return self.parser
    
    def get_n_samples(self):
        return self.args['n_samples']
    
    def get_n_epochs(self):
        return self.args['n_epochs']
    
    def get_stress_number(self):
        return self.args['stress_number']
    
    def get_retrain(self):
        return self.args['retrain']
    
    def get_model_name(self):
        return self.args['model_name']
    
    def get_data_path(self):
        return self.args['data_path']
    
    