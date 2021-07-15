# The Large-Batch Parabolic Approximation optimizer (LABPAL)  
Approximates the full-batch loss in negative gradient direction with a one-dimensional parabolic function.
This approximation is done rarely and measured with a large batch size.  
Then, the learning rate is derived form the position of the minimum of the approximation and reused for
 multiple steps.  
Either SGD or normalized SGD (NSGD) using the unit gradient can be used. For SGD a learning rate is measured, whereas, for NSGD the step size is measured. In the following we just use the term learning rate.

##The paper
The LABPAL optimizer is introduced in the "Using a one dimensional parabolic model of the full-batch loss to estimate learning rates during training"  following paper:
todo link

## The performance
LAPBAL works on the considered experiments without any hyper parameter tuning needed.
It automatically infers suitable learning rates and keeps the gradient noise low by increaseing the batch size by iteratively sampling batches.
TODO write version with noise adaptation factor replaced with dataset size and batch size.
TODO figure of LABPAL Resnet cifar10 and cifar 100. TODO plots are not externalized for some reason.


## LABPAL reference implementation  
The LABPAL reference implementation is found in 'source/optimizer/lap_pal'.  
A closure method is needed for the 'optimizer.step()' method. This closure is given by the 'get_closure' method of the 'LABPAL' class.

# Run experiments  
- adapt the hyper parameters and paths in 'configuration_lab_pal.txt'  
- run main.py  
