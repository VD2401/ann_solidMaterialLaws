import torch
import sys

def data_key(i):
    return "data_elasticity_3D_128_" + str(i)

def main():
    IDX_FILES = [3, 4, 5]
    DATA_PATH = "../data/data_files/"
    STRESS_NUMBER = 0
    LOAD_NUMBER = 0

    for i in IDX_FILES:
        key_source = DATA_PATH + data_key(i) + ".pt"
        key_input = DATA_PATH + data_key(i) + "_L" + str(LOAD_NUMBER) + "_S" + str(STRESS_NUMBER) + "_input.pt"
        key_output = DATA_PATH + data_key(i) + "_L" + str(LOAD_NUMBER) + "_S" + str(STRESS_NUMBER) + "_output.pt"
        
        try:
            data = torch.load(key_source)
        except:
            raise ValueError(f"Data file {data_key(i)} not found in {DATA_PATH}")
        print("File " + str(i+1) + " loaded")
        
        # Save input and output
        input = data["young_modulus"].view(128, 1, 64, 64, 64)\
                                        .detach().clone().cpu()
        torch.save({'input': input}, key_input)
        print("Input of file " + str(i+1) + " saved. It corresponds to the young modulus.")
        output = data["stress"].select(5, STRESS_NUMBER)\
                                .select(1, LOAD_NUMBER)\
                                    .view(128, 1, 64, 64, 64)\
                                        .detach().clone().cpu()
        torch.save({"output": output}, key_output)
        print("Output of file " + str(i+1) + 
                " saved. It corresponds to the stress number " +
                str(STRESS_NUMBER) + " and the load number " +
                str(LOAD_NUMBER))
        
        del data # Free memory
    

if __name__ == "__main__":
    main()