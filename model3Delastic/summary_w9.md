---
Meeting: Follow-up
Date: 2024-05-01
Participants:
  - Paul Devianne
  - Lori Graham-Brady
---
- ML Flow
    - Paul still learning on how to apply to this project
    - Very helpful for research to build up on top after the project

- Model
	- Data augmentation
		- 6 implemented
            - 3 rotations of 180° around each axis
            - 3 flips over each axis
            - The augmentation seems to accelerate the training process
                - For 2 times more samples from one augmentation
                - The number of epochs until a given criterion is 3.8 times lower
                - To study this correctly Lori suggests to use the same number of training samples but only change the fraction of augmented samples to see if the training process is accelerated   
            - Flipping seems to be less effective than rotation for the training process
    - Training for N_samples > 128 for non augmented data
        - There is a jump in the loss function after 40 epochs for all N > 128
        - Then the loss saturates at a value higher than before the jump
        - Study maybe the effect of the learning rate

- Cluster
    - Access cluster
        - Email Anne-Françoise Suter after confirmation with Jean-François Molinari
    - Robust results
        - To build robust results for the end of the project
