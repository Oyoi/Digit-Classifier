import tensorflow as tf
from keras.datasets import mnist 
from keras.utils import np_utils 

def load_dataset():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	
    # reshaping x_train and x_test
	X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
	
    # Normalizing our input data
	X_train = X_train / 255
	X_test = X_test / 255
	
    # Converting y_train and y_test to categorical classes 
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)

	return X_train, y_train, X_test, y_test
 
def digit_recognition_cnn():
	cnn = tf.keras.models.Sequential()	
	
    # Convolution layer
	cnn.add(tf.keras.layers.Conv2D(filters=30, kernel_size=(5, 5), activation='relu', input_shape=[28,28,1]))
	
    # Pooling layer
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
	
    # Convolution layer
	cnn.add(tf.keras.layers.Conv2D(filters=15, kernel_size=(3, 3), activation='relu'))
	
    # Pooling layer
	cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
	cnn.add(tf.keras.layers.Dropout(0.2))
	
    # Flattening
	cnn.add(tf.keras.layers.Flatten())
	
    # Full connection
	cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
	cnn.add(tf.keras.layers.Dense(units=50, activation='relu'))
	
    # Output layer
	cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))
	
    # Compiling our model with Conv + ReLU + Flatten + Dense layers
	cnn.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])
 
	return cnn

# Creating and building our model
X_train, y_train, X_test, y_test = load_dataset();
model = digit_recognition_cnn()

#Training our model over 20 epochs with batch size 200 
model.fit(X_train, y_train, epochs = 20, batch_size = 200)

# Evaluating the accuracy of our trained model
accuracy = model.evaluate(X_test, y_test, verbose = 0)

print("Accuracy:", end = ' ')
print(accuracy[1])

# Saving our trained model, to be loaded later by our classifier
model.save('digitRecognizer.h5')