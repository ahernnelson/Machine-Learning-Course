### import libraries for building the model
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

### Read in data and specify number nodes in layer
predictors = np.loadtxt('predictorss_data.csv', delimeter=',')
n_cols = predictors.shape[1]

#### model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(1))

#### Here is an example of compiling a model
n_cols = predictors.shape[1]
model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(n_cols,)))
model.add(Dense(100,activation='relu',input_shape=(n_cols,)))
model.add(Dense)
model.compile(optimizer ='adam', loss ='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

### adn to fit we run
model.fit(predictors, target)

### Saving the model
from keras.models import load_model
model.save('model_file.h5')
my_model = load_model('my_model.h5')
predictions = my_model.predict(data_to_predict_with)
probability_true = predictions[:,1]
my_model.summary()