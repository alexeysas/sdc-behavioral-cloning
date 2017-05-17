# **Behavioral Cloning project**  

### Project Goal
The goal of this project is to:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track without leaving the road

[//]: # (Image References)

[image1]: ./examples/lenet.png "LeNeT Model"
[image2]: ./examples/cropped.png "Cropped"
[image3]: ./examples/nvidia.png "NVIDIA architecture"
[image4]: ./examples/hist1.png "Angles histogram"
[image5]: ./examples/original.png "Original Image"
[image6]: ./examples/flipped.png "Flipped Image"
[image7]: ./examples/recovery.png "Recovery Image"
[image8]: ./examples/sample.png "Three images sample"
[image9]: ./examples/final_dist.png "Model training data distribution"
[image9]: ./examples/loss.png "Loss"

Technicaly we need to predict steering angle based on the camera images captured from the vehicle. Actualy vehicle has three cameras which appeared to be extremly useful addition as the result.

Images samples captured by vehicle cameras:

![alt text][image8]

### Solution design approach

The overall strategy to complete the project was to start from a simple convolutional network (for example similar to LeNeT architecture: 

![alt text][image1]

And then try to improve model additionally applying various data processing techniques.

I think that simple LeNeT model might be appropriate because convolutional neural networks are pretty good for image recognition and classification. In spite of the fact that we have regression task still this model should be a good start.

Also as a simple dataset for training - I've collected two laps of data using first track. My goal for now was not to fight with over fitting but make sure that model able to drive the track in a some way. 

Additionally, I've normalized model inputs to fit values to the range [-0.5, 0.5] which is proven to work a lot better than initial color range [0, 255]. Also cropped image data above the horizon (50 pixels from top), removed the car image (20 pixel from bottom), and cropped 20 pixels from thr left and right side.  So initial and final images looks like: 

![alt text][image2]

As usual, data was split to training and validation set (20% dedicated to validation) Model had pretty low validation and training loss.  However, the driving behavior was awful. Car mostly go straight forward and leaves the track immediately.

Initial idea was to improve model by using more complex layers. I've tried adding more convolutional levels, pooling layers, different non-linearities, playing with fully collected layers and dropouts hardly got any resonable better results. Car was ableto drive a little bit longer.

Finally, I've changed model to the NVIDIA model. According to the paper: https://arxiv.org/pdf/1604.07316v1.pdf this model was selected as a top performer among all other models for self-driving cars and worked well for real tests. So it should be great model for this project as well.

NVIDIA model:

![alt text][image3]

The result was terrible. Car was hardly able to drive through the first simple turn and left the road...  As this model was proven performer for the self-driving car we need to address issue from a different angle.. 

## Data processing and augumentation

So, obviously training data need to be enchanced and collected in a right way to achieve good results.

Different data pre-processing techniques were tried which worked well for previous computer vision projects such as:

* Color space convertion (HSV, YUV)
* Noise reduction
* Edge detection (to emphsize road borders)
* Tried regions of interests using different forms and shapes.

Nothing worked acceptable well to improve model behavior. 

Finally, I've realized that the main issue of the model is that it is biased to the small angles and almost always left the road heading straight forward:

So when histogram was build to visualize the collected data - the issue become obvious:

![alt text][image4]

It is clear that model is biased to to the zero angles.  Also in the NVIDIA paper they mentioned that they had to add a lot of curved turns data tp deal with this issue and make it works correctly for the turns.

So as a first step, simple solution was applied: if np.abs(angle) < 0.1 then data is not added to tteh training set for to the model. This worked as a charm - car was able to complete almost half of the road successfully.

Additionaly, as a road is a conter-clockwise lap it is also biased to the left turn - so to reduce this impact image is randomply flipped with 0.5 probability with reversing corresponding angle:

Original image:

![alt text][image5]

Flipped image:

![alt text][image6]

At the end  vehicle is able to drive autonomously around the track without leaving the road with default speed (9). However, speed increase still caused some issues for driving behaviour. 

## Data colection and training stategy 

So the next goal was to make model drive autonomously with reasonable high speed on the first track. Also model is clearly over fitting for the first track as car drives out of the road as soon as simulation started for the second track. 

So to prevent over fitting and train model to drive more solidly. I've collected two laps of data for the second track and addition data for the first track.  After training on the new data - model was able to complete first turn for the second track.

Realized that model is poorly recovered when it is above to cross road borders, to solve this I've collected so called recovery data for the places where model fall out of the road during my test runs. Starting from the road borders from the left side and right sides back to center so that the vehicle would learn how to recover back to the center.

Here is some example of images:

![alt text][image7]

Strangling for a couple of days still was not able to achieve acceptable results for both trucks. It appeared that then recovery data was collected, additional partially "bad driving behavior" was collected as well - we need to drive to the road border to collect recovery data - which is not desirable behavior.  As it is almost not possible to distinguish this bad data, I've decided to go with completely different approach:

* Collect ideal driving data using center of the road for both tracks.
* Use additional right and left cameras to simulate recovery behavior 

As images from these cameras as slightly shifted we need to include corresponding angle adjustment as well. So it was introduced as additional hyper parameter for the model to tune. 

This approach did the trick.

## Final Model Architecture and Training Process

The final model architecture (main.py lines 159-174) consist of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Cropping 90x320     	| Outputs: 90x320x3	|
| Normalization     	| Normilize values to fit [-0.5, 0.5] range,  outputs: 90x280x3	|
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

Basically I've used slightly modified NVIDIA network by adding dropout layers to prevent over fitting. 

I've used MSE as accuracy metric and 'Adam' optimizer. As Adam optimizer is used instead of SGD we don't have to tune learning rate parameter. 

However, there are a couple of additional parameters were tuned to make model work well:

* ignore_threshold - angles less than this value are ignored for the training data with probability below. Was set to 0.1 for the final model
* remove_probability - remove data from training with small steering angles with this probability. 0.7 value was selected for thr final model.
* correction angle - angle adjustment for left and right camera models. 0.3 was selected for final model. 
* number of epoch = 10
* batch size = 32

Surprisingly, after data pre-processing and augmentation I've end up with 2920 data points with following angles distribution:

![alt text][image9]

It looks like this small amount of data is enough to get pretty good driving behaviour. 

Finally I randomly shuffled the data set and put 0.2% of the data into a validation set.  Although, I was able to fit all my training data in my PCs memory and used in-memory training (main.py lines 224-256) as it a way faster to run. I've provided an ability to use generator based approach in case I have a lot of data available (main.py lines 194-218), however it works a bit slower than in-memory approach.

The ideal number of epochs was 10 as evidenced by the fact that validation loss stopped decreasing. And model is not over-fit as validation and training losese are similar:

![alt text][image10]

## Results

Here is resulting video of the model driving for the first track. Driving is pretty smooth:

https://github.com/alexeysas/sdc-behavioral-cloning/tree/master/examples/Track1.mp4

Additionally, model was almost able to complete a lot more complex track - just stacked in the end on the complicated turn:

https://github.com/alexeysas/sdc-behavioral-cloning/tree/master/examples/Track1.mp4

I believe that it can be fixed by collecting more data and with additional parameters tuning.

As a conclusion of this project I've figured out that data quality, data processing and augmentation are often more important than a model itself to archive good results.
