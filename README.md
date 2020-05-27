###Files
The following programs run in Python 3.6 from the terminal using "python3 insert_program_name_here". 
To runt training and testing model files PyTorch must be installed but CUDA doesn't have to be (CUDA allows it to be run on GPU which is faster).
###sorting_feeder_images.py
* Functions to sort images from given structure into a structure to ease training
* Functions to print the amount of files in all created directories
###training_model.py
* Contains ImageClassification class to fine tune the ResNet18 model
* Call init method with the location of training data, model save location and amount of epochs
* Call fine_tune_model_train method to train the model 
###test_model.py
* Contains an ImageClassificationTest class to test the accuracy of a saved model
* All incorrectly classified images are saved to a wrong_images directory
* Call the init method with list of class names, testing data location, saved model location
* Call the set_up_test method to test the model
