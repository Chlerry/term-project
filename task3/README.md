# Introduction
As video became one of the most crucial entertainment appoarch nowdays, people deserve smooth and high quailty watching experience. However, with the resolution, frame rate and duration grows, the data memory use of video increases rapidly. As the result, it becames a pratical problem that how to use there data efficiently. The video compressor and decompressor offers a much lower memory cost video in the way that keep the original quaility as much as possible by using neural network techinology.
In our project we will briefly introduce two different network architectures:
图
For the training and test data frames we spilited each single frame into muliple 16*16*3 frames as first step. Then we process the network seperately for every small frames.
The figure shows the stucture of 1/16 ratio compressor working process. The layers are follow: a 64*7*7 filter, a 2*2 Max-Pooling, a 32*5*5 filter, a 2*2 Max-Pooling, a 16*1*1 filter, a 8*3*3 filter, a 3*3*3 filter. We used the RaceHorses_416*240_30 as training data frames and BlowingBubbles_416*240_50 as test data frames. Since it's a regression problem we choose MSE as the loss function. 

For different ratio we use PSNR value to evaluate the performance of the network. As we can see from the figure the PSNR value grows as the ratio increases. In conclusion, the processed images have higher quailty under less layers copmress.
# Setup Environment
conda install -c conda-forge keras

conda install -c conda-forge matplotlib

conda install -c anaconda pillow

conda install -c anaconda scikit-image
