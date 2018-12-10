import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
data = pd.read_csv("basketball_shot_log.csv")
predictors = data.drop(['shot_result'], axis=1).matrix()
n_cols = predictors.shape[1]
target = to_categorical(data.shot_result)
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(2, activation= 'softmax'))
model.compile(optimizer = 'adam',
              loss ='categorical_crossentropy', metrics = ['accuracy'])
model.fit(predictors,target)