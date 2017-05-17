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

Technicaly we need to predict steering angle based on the camera images captured from the vehicle. Actualy vehicle has three cameras which appeared to be extremly useful addition as the result.

Images samples captured by vehicle cameras:

![alt text][image8]

### Solution Design Approach

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

### Data processing and augumentation

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

#### Data colection and training stategy 

So the next goal was to make model drive autonomously with resonable high speed on the first track. Also model is clearly over fitting for the first track as car drives out of the road as soon as simulation started for the second track. 

So to prevent overfitting and train model to drive more solidly. I've collected two laps of data for the second track and additionl data for the first track.  After training on the new data - model was able to complete first turn for the second track.

Realized that model is poorly recovered when it is above to cross road borders, I've collect so called recovery data for the places where model fall out of the road during my test runs. Startimg from the road borders from the left side and right sides back to center so that the vehicle would learn how to reecover back to the center.

Here is some exaple of images:

![alt text][image7]
![alt text][image8]
![alt text][image9]


Stragling during couple of days still was not able to

* Trying to add images from left and right camera with corresponding angle adjustment (not working well for me though with any   parameters - so removed this in final model, it looks like to use this technique correctly - more complex algorithm is required)



#### Final Model Architecture

The final model architecture (main.py lines 162-179) consisted of a convolution neural network with the following layers and layer sizes 

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

Basicly I've used sligtly modified NVIDIA network by adding dropout layers to prevent overfitting. I beleive that reason why NVIDIA dont have drouput layer is becouse they collecteed tons of real data. As we only have two tracks - it can be benefitial for our model.

#### Training Process
When data is collected - I've used MSE as accuracy metric and 'Adam' optimizer. As I used Adam optimizer instead of SDG I dont have to tune learning rate. However, I have addtional parameters to wwork with instead:

* ignore_threshold - angles less than this value are ignored with probability: remove_probability - also additional parameter to tune.
* number of epoch
* Either to use left and right camera images or not.

After the collection process, I and up with X data points with following angles distribution:

![alt text][image6]

Finally I randomly shuffled the data set and put 0.1% of the data into a validation set.  Although, I was able to fit all my training data in my PCs memory and used in memory training (main.py lines 224-256) as it a way faster to run. I've provide avility to use generator based approach in case I have a lot of data avalible (main.py lines 198-222), however it works a bit slower than in-memory approah.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by the fact that validation loss stopped decreasing:

![alt text][image6]

#### Results

Res

