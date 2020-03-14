# Introduction
As digital video consumption is becomimg an increasingly crucial entertainment method nowdays, people deserve smooth and high quailty streaming experience. However, with the growing resolution, frame rate and time duration, the data storage, and network bandwidth used by video increases rapidly. As the result, video compression becames a pratical problem. A good video compressor and decompressor offers much lower memory cost for videos in the way that keeps the original quaility as much as possible. In our project we will briefly introduce and implement two Convolutional Neural Network architectures proposed by other researchers \cite{}. Also, we will experiment a Generative Adversarial Network with Convolutional Neural architechture \cite{}. 


different network architectures.

# Setup Environment
conda install -c conda-forge keras

conda install -c conda-forge matplotlib

conda install -c anaconda pillow

conda install -c anaconda scikit-image

# Proposed Method 1

For the training and test data frames we spilited each single frame into muliple 3*16*16 frames as first step. Then we process the network seperately for every patches and merge the patches at the end.
The figure shows the stucture of 1/16 ratio compressor working process. The layers are follow: a 64*7*7 filter, a 2*2 Max-Pooling, a 32*5*5 filter, a 2*2 Max-Pooling, a 16*1*1 filter, a 8*3*3 filter, a 3*3*3 filter. We used the RaceHorses_416*240_30 as training data frames and BlowingBubbles_416*240_50 as test data frames. Since it's a regression problem we choose MSE as the loss function. 
For the traning process the shape changes are as follow: input 3*16*16 -> 64*16*16 -> 64*8*8 -> 32*8*8 -> 32*4*4 ->16*4*4 -> 8*4*4 -> 3*4*4 and the total parameters been calculated is 9472+51232+528+1160

For different ratio we use PSNR value to evaluate the performance of the network. As we can see from the figure the PSNR value grows as the ratio increases. In conclusion, the processed images have higher quailty under less layers copmress.

# Proposed Method 2

# Proposed Method 3

# Experimental Studies 
## Dataset description

## Quantitative evaluation

## Perceptual quality evaluation

## Complexity and model size analysis

# Conclusions and Future Work
