import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
def get_new_model(input):
    model = Sequential
    model.add(Dense(100, 'relu', input_shape=input))
    model.add(Dense(100,'relu'))
    model.add(Dense(2,'relu'))
    return (model)
lr_to_test = [.000001,0.01,1]
for lr in lr_to_test:
    model = get_new_model()
    my_optimizer = SGD(lr=lr)
    model.compile(my_optimizer, 'categorical_crossentropy')
model.compile(optimizer = 'adam',
              loss='categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(predictors, target, validation_split=0.3)
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)
model.fit(predictors, target,
          validation_split = .3, epochs = 20,
          callbacks= [early_stopping_monitor])
######## MNIST #########################
# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50,activation = 'relu', input_shape= (784,)))

# Add the second hidden layer
model.add(Dense(50, activation = 'relu'))

# Add the output layer
model.add(Dense(10, activation='softmax') )

# Compile the model
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit the model
model.fit(X, y, validation_split = .3)
