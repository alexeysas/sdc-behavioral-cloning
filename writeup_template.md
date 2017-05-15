# **Behavioral Cloning** 

## **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/lenet.png "LeNeT Model"
[image2]: ./examples/cropped.png "Cropped"
[image3]: ./examples/nvidia.png "NVIDIA architecture"
[image4]: ./examples/hist1.png "Angles histogram"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### Solution Design Approach

The overall strategy for deriving a model architecture was to start from a simple convolutional network similar to LeNeT arcjirecture: 

![alt text][image1]

I thought this model might be appropriate because convolutional neural networks are pretty good for image recognition and classification. So I've started from one of the simpliest well-known architectures as a base.

Also as a simple dataset for training - I've collected two laps of data for the firest track. My goal for now was not to fight overfiting but make sure that model able to learn the track. 

Additinaly, I've normalized model imputs to fit values to the range [-0.5, 0.5] which is proven to work a lot beteer than initial color range [0, 255]. Also cropped image data above the horizont (50 pixels from top) and car image (20 pixel from bottom). So final image looks like: 

![alt text][image2]

I split my image and steering angle data into a training and validation set. I found that my first model had a pretty comparable validation and training loss.  However, the driving behavior was auful. Car mostly go stright forward and leaves the track immidiatly.

My initial idea was to improve model by using more complex structure. I've tried adding more convolutional levels, pooling layers, non-linearities, playing with fully collected layers - hardly got any better results. 

Finnaly, I've changed model to the NVIDIA model. According to their paper: https://arxiv.org/pdf/1604.07316v1.pdf this model was selected as top perfomer among all other models they tried for the self-driving car and worked well for the real tests. 

![alt text][image3]

The result was terrible. Car hardly able to drive through the first simple turn and left the road..  As this model was proved performer for the self-driving car - I realised that I need to seek issue somehwere else keeping model fixed. 

#### Data processing and augumentation

So, obviously, I had to collect and pre-process training data in a right way to achive good results. I've tried different data pre-processing techniques such as:

* Color space convertion
* Noise reduction
* Edge detection - to emphsize road borders.
* Tried regions of interests using different forms and shapes.

Finaly - I've realzied that the main issue of the model is that it is biased to the small angles and almost always left the road heading stright forward:

So I've build the histogram to visualize the colelcted data:

![alt text][image4]

So, now it is clear that model is biased to to the zero angles.  Also in the NVIDIA paper they mentioned that they had to add a lot of curved truns data as model is biased to the straight roads to make it works correctly.

So, I've just tried simple solution: if np.abs(angle) < 0.1 I am not adding this data to the model. This worked as a charm - car was able to complete half of the road succsesfuly.



The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.



####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Cropping 90x320     	| Outputs: 90x320x3	|
| Normalization     	| Normilize values to fit [-0.5, 0.5] range,  outputs: 90x320x3	|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 43x158x24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 20x77x36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 8x37x48 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 8x37x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 8x37x64 	|
| RELU					|												|
| Fully connected		| 2256x100        									|
| Dropout | keep_prob = 0.8        									|
| Fully connected		| 100x50        									|
| Dropout | keep_prob = 0.8        									|
| Fully connected		| 50x10        									|
| Fully connected		| 10x1        									|

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Also I've tryed to 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Also I've recorded two laps of data using second track driving to prevent overfeating for the first track (it idealy drives over first track after training on three laps data collected from first track only, but it is clearly overfitting as it is not able to drive a single turn for the second truck) and in attempt to eveluate model .

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
