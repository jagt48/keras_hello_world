from numpy import loadtxt
from keras.models import Sequential
from keras.layer import Dense

# Load the dataset.
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input( (X) and output (y) variables.
X = dataset[:,0:8]    # Input data in columns 0-7.
y = dataset[:,8]      # Output data in column 8.

# Create a model using Sequential, adding one layer at a time.
# Define keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))    # First hidden layer has 12 nodes, model expect 8 variables, ReLU activation function (as opposed to Sigmoid or Tanh).
model.add(Dense(8, activation='relu'))    # Second hidden layer has 8 nodes and uses ReLU.
model.add(Dense(1, activation='sigmoid'))    # Output later has one node and uses sigmoid activation function.

# Compile keras model.
model.comppile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Epochs - one pass through all of the rows in the training dataset.
# Batch - one or more samples considered by the model within an epoch before weights are updates.
# Both epoch and batch determined by trial-and-error. Once enough iterations have completed, the error will
# level out after some point for a given model configuration (called model convergence).

# Fit the keras model to the dataset.
model.fit(X, y, epochs=150, batch_size=10)