from numpy import loadtxt
from keras.models import Sequential
from keras.layer import Dense

# Load the dataset.
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split into input( (X) and output (y) variables.
X = dataset[:,0:8]    # Input data in columns 0-7.
y = dataset[:,8]      # Output data in column 8.