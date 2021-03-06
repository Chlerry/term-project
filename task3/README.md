# Introduction
As digital video consumption is becomimg an increasingly crucial entertainment method nowdays, people deserve smooth and high quailty streaming experience. However, with the growing resolution, frame rate and time duration, the data storage, and network bandwidth used by video increases rapidly. As the result, video compression becames a pratical problem. A good video compressor and decompressor offers much lower memory cost for videos in the way that keeps the original quaility as much as possible. In our project we will briefly introduce and implement two Convolutional Neural Network architectures proposed by other researchers \cite{}. Also, we will experiment a Generative Adversarial Network with Convolutional Neural Architechture \cite{} to generate chroma components for luminance images. 


# Setup Environment
conda install -c conda-forge keras

conda install -c conda-forge matplotlib

conda install -c anaconda pillow

conda install -c anaconda scikit-image

# Proposed Method 1

The figure shows the stucture of 1/16 ratio compressor working process. The layers are follow: a 64*7*7 filter, a 2*2 Max-Pooling, a 32*5*5 filter, a 2*2 Max-Pooling, a 16*1*1 filter, a 8*3*3 filter, a 3*3*3 filter. We used the RaceHorses_416*240_30 as training data frames and BlowingBubbles_416*240_50 as test data frames. Since it's a regression problem we choose MSE as the loss function. 
For the traning process the shape changes are as follow: input 3*16*16 -> 64*16*16 -> 64*8*8 -> 32*8*8 -> 32*4*4 ->16*4*4 -> 8*4*4 -> 3*4*4 and the total parameters been calculated is 9472+51232+528+1160+219. 

# Proposed Method 2

The figure shows the stucture of 1/16 ratio compressor working process. The layers are follow: a 64*7*7 with 2*2 strides filter, a 32*5*5 with 2*2 strides filter, a 16*1*1 filter, a 8*3*3 filter, a 3*3*3 filter. We used the RaceHorses_416*240_30 as training data frames and BlowingBubbles_416*240_50 as test data frames. Since it's a regression problem we choose MSE as the loss function. 
For the traning process the shape changes are as follow: input 3*16*16 -> 64*8*8 -> 32*4*4 ->16*4*4 -> 8*4*4 -> 3*4*4 and the total parameters been calculated is 9472+51232+528+1160+219

For different ratio we use PSNR value to evaluate the performance of the network. As we can see from the figure the PSNR value grows as the ratio increases. In conclusion, the processed images have higher quailty under less layers copmress.

# Proposed Method 3

First, in this method, images are converted from RGB colorspace to YCbCr color space. 
This method uses the Convolutional Neural Network proposed in Method 1 to compress the images' Y channel, which is the images' luminance. After this, the predicted (decoded) luminance images will be passed to a generator to predict their CbCr color mask. After Colorization, the images will be feeded to the discriminator. The generator and discriminator will be trained together to compete with each other. 

# Experimental Studies 
## Dataset Description
There are three sets of images available: RaceHorses_416x240_30 (300 240x416 images), BlowingBubbles_416x240_50 (500 240x416 images), BasketballDrill_832x480_50 (500 480x831 images)

First each image was depatched into multiple 16x16x3 frames. After finishing the training progress and testing progress, patches will merge back to image, which will used to lost evaluation (calculating MSE and PSNR). 

Each image set will be splitted as 75% training data and 25% testing data, used for each model to evaluate. 

## Quantitative evaluation

## Perceptual quality evaluation

## Complexity and model size analysis

# Conclusions and Future Work
In conclusion, we tried three different network architectures and we learned the importance of different factors for compress and depress video images. The results are acceptible till the ratios became rather small and as the frames divide into smaller patches the visulization performance getting better. In the sametime, we spent a decent amount of time to train and predict the network which could be improved in the future with faster calculate capability. In other way, the relationship between frames could be predicted to reduce the training dataset size and increase the visulization performance.
