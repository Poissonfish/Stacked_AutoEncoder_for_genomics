
#%%
#importing shit for keras
import pylab as plt
import numpy as np
import seaborn as sns; sns.set()

import keras

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam

#becareful about input shape
shape = (3093, 1, 1)

#%%

#%%
#setting the 
from sklearn import preprocessing


min_max_scaler = preprocessing.MinMaxScaler()
x_train =  min_max_scaler.fit_transform(df)

#%%

#the keras model and compilation
m = Sequential()
m.add(Dense(500, activation = 'relu', input_shape=(700,)))
m.add(Dense(100, activation = 'relu'))
m.add(Dense(2, activation = 'linear', name = "bottleneck"))
m.add(Dense(100, activation = 'relu'))
m.add(Dense(500, activation = 'relu'))
m.add(Dense(700, activation = 'sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())
m.summary()
#%%
#training the keras model
history = m.fit(x_train, x_train, batch_size=8, epochs=20, verbose=1, 
                validation_data=(x_train, x_train))

#%% now to time to do predictions

#makes model that gets what is at the bottle neck layer
encoder = Model(m.input, m.get_layer('bottleneck').output)

#making awesome prediction
preds = encoder.predict(x_train)

#%%
#normalizing prediction
from sklearn import preprocessing


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(preds)
#df68 = pd.DataFrame(x_scaled)


#%%
#loading the groups of the test data
groups1 = groups


#%%
#now I will plot the points and see what happens
df69 = pd.DataFrame(x_scaled, columns = ['AE1', 'AE2'])

#now binding the column onto the data
pltDat = pd.concat([df69, groups1], axis = 1)
pl1 = df69.plot.scatter(x = 'AE1', y = 'AE2', c = 'DarkBlue')
#pl2 = pl1.get_figure()
#pl2.savefig("/Users/samuelrevolinski/Dropbox/AE2norm.pdf")


import seaborn as sns
#sns.regplot(x = "AE1", y = "AE2", col = 'groups', data = pltDat)
#sns.plot.show()


#%%
#writing temp file to use ggplot2 for pltting
pltDat.to_csv("/Users/samuelrevolinski/Dropbox/temp.csv")

#writing 