import numpy as np
from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import load_model
 
def load_new_image(path):
    # Loading our new image
	newImage = load_img(path, color_mode="grayscale", target_size=(28, 28))
	
 	# Converting our n image to an array
	newImage = img_to_array(newImage)
	
 	# Reshaping to single sample with 1 channel 
	newImage = newImage.reshape(1, 28, 28, 1).astype('float32')
	
	# Normalizing our new image
	newImage = newImage / 255

	return newImage	
 
# load a new image and classifying it
def test_model_performance(path):
	img = load_new_image(path)
	
 	# Loading our saved model
	loaded_model = load_model('digitRecognizer.h5')

 	# Prediction for our image in array format
	imageClass = (loaded_model.predict(img) > 0.5).astype("int32")
	print(imageClass[0])
	
 	# Prediction for our image in number format
	imageClass = np.argmax(loaded_model.predict(img), axis = -1)
	print(imageClass[0])
 
# Testing our model with test images for each digit
print("Test for Digit 0:")
test_model_performance('test_images/digit0.png')

print("Test for Digit 1:")
test_model_performance('test_images/digit1.png')

print("Test for Digit 2:")
test_model_performance('test_images/digit2.png')

print("Test for Digit 3:")
test_model_performance('test_images/digit3.png')

print("Test for Digit 4:")
test_model_performance('test_images/digit4.png')

print("Test for Digit 5:")
test_model_performance('test_images/digit5.png')

print("Test for Digit 6:")
test_model_performance('test_images/digit6.png')

print("Test for Digit 7:")
test_model_performance('test_images/digit7.png')

print("Test for Digit 8:")
test_model_performance('test_images/digit8.png')

print("Test for Digit 9:")
test_model_performance('test_images/digit9.png')