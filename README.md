# object-detection-with-deep-learning

This project demostrates use of deep neural networks for object detection. Here I am using the neural network to detect car in an image or video frame.

This can be easily extended to detect others like bike, truck, pedestrian, traffic light from the test data by changing the searchObject.py as the model is already trained to identify these objects.

The model can also be easily trained to detect other things by changing the setup_data function in the model.py. This is where we setup the train data and label and also categorize the labels.

To use the code for multiple type object detection we can extend the search_windows function in searchObject.py 
