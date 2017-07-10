# deeplearning-food

Deep learning with little data using bottlenecking.

The challenge here was to build a deep learning model that provided a binary classification between either a picture of a 
sandwich or a picture of sushi. 

This was a particularly interesting challenge, given the dataset contained just 402 examples of each image, and so generally 
we would be training the model based on around ~300 instances of each image - which is a challenging prospect with deep 
learning. Typically deep learning works best with a very large set of data. 

This problem reminded me of a blog post I had read [1] on building a powerful classifier based on little data. This is the 
approach I have used in the code I have attached. I have built a classifier based on retraining the top layer of the 
pre-trained ImageNet dataset, using a VGG16 architecture. The ImageNet dataset is a sensible choice to use as it provides 
good features for a number of computer vision problems. This is not necessarily the best choice for this problem, though. If I 
had more time to research this problem, it may well be that there is a pre-trained model based just on food, which would most 
likely have been more effective.

The code is split into four parts. The first part is simply building the dataset. I have created a directory structure for
training and validating the data. Before constructing the dataset, first I randomised the images in the original directory and
then split the data ~63:37 into a training set and a validation set. Again, if I had more time to spend on this problem, I 
would have explored some image processing options here. Given the geometrical nature of the problem, for instance sushi may 
generally exhibit circular features, whereas sandwichs may have more pointed features; some pro-processing to emphasise these 
feature may have been beneficial. 

The second part creates the bottleneck features for the model. Here I have used Keras' data generator to augment the training 
set by adding rotations, shears, flips and translations to artificially increase the training size. We also normalise the 
images into the range [0,1]. The third part trains the top layer of the model sequentially. We use a sigmoid activation 
at the final step as this is ideal for binary problems. We also use a binary cross entropy loss function and Adam's optimiser 
for the same reason. 

Finally, we can now use the function predict_model to input new data into the model to receive a classification. From my 
experimentation with parameters, such as the size of the images, image augmentation and training/test set sizes, this model 
has an accuracy of ~80-85%. This is not a bad result for code which only takes a small amount of time to run, but with more 
time and analysis, I think this approach could gain some excellent results.

Example running of the code:

      create_dataset() # here you have to specify the path you want the directory to be created in
  
      save_bottlebeck_features() # again you have to specify where you want to save the features
  
      train_top_model() # check you are loading features from the correct directory
      
      predict_image(file) # file is the path to the image you wish to classify

[1] - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
