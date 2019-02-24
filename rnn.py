# Recurrent Neural Network
# Part 1 - Data Preprocessing

# Importing the libraries
#Basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#Scaling library
from sklearn.preprocessing import MinMaxScaler

#NN library
from keras.models import Sequential #For creating an Neural Network object
from keras.layers import Dense #adds output layer
from keras.layers import LSTM #adds LSTM layer
from keras.layers import Dropout

#Callbacks keras library
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
#random.seed(0)
#Find correlations between data. If high, then add to training set. 

# Importing the data along with its correlated.
#Aqui agrega los indices correlacionados y agregalos a la lista data_train
#Es importante que dataset_train sea el del indice que queramos predecir
dataset_train = pd.read_csv('Google_Stock_Price_Train_10y.csv')
dataset_corr_1 = pd.read_csv('Yahoo_Stock_Price_Train.csv')
data_train = (dataset_train['Open'], dataset_corr_1['Open'])


# Extracting only the "Open Value" Colunm of the Stocks
training_set = pd.concat(data_train, axis = 1).dropna()
training_set = training_set.iloc[:,:].values
#training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling: Standardisation or Normalization? 
#Better normalization for RNN, specially because we have sigmoid act. f.

# Getting the real stock price of 2017
dataset_test_google = pd.read_csv('Google_Stock_Price_Test.csv')
dataset_test_yahoo = pd.read_csv("Yahoo_Stock_Price_Test.csv")
data_test = (dataset_test_google["Open"],dataset_test_yahoo["Open"])

test_set = pd.concat(data_test,axis=1).dropna().values

real_stock_price = dataset_test_google.iloc[:, 1:2].values


#Creating the scaler object
sc = MinMaxScaler(feature_range = (0, 1)) #Q: When is it convenient to change feature range?
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure; how many timesteps to remember and how many days to predict into the future

#What the RNN needs to remember to predict. This is a very important step which if
#not done properly can lead to overfitting or nonsense predictions

#It is possible to add more dimensions, such as other indicators: stock prices
n_future = 1 # number of days to predict
n_past = 120 #number past days to remember for the prediction
X_train = []
y_train = []
for i in range(n_past, len(training_set_scaled)-n_future+1):
    X_train.append(training_set_scaled[i-n_past:i, :])
    y_train.append(training_set_scaled[i + n_future - 1: i + n_future , 0])

X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping
if len(data_train)==1:
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Initialising the RNN
n_blocks=4
n_units=60

regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = n_units, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))
#units, lstm cells or memory units.

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = n_units, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = n_units, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = n_units, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = n_units, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = n_units))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

#Define a loss function for our particular problem

#import keras.backend as K
#def derivative_squared_error(y_true, y_pred):
#    y_true_f = y_true
#    y_pred_f = K.flatten(y_pred)
   # derivate_true = y_true_f-y_true_f
    #derivate_pred = y_pred[i]-y_pred[i-1]    
#    return y_true_f


# Compiling the RNN. RMSprop is usually a good optimizer for RNN
regressor.compile(optimizer = 'adam', loss = "mean_absolute_percentage_error")

es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=40, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
 
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 2, callbacks = [rlr, tb, es,mcp],validation_split = 0.2, verbose = 1, batch_size = 64)



# Part 3 - Making the predictions and visualising the results


# Getting the predicted stock price of 2017
dataset_total = np.concatenate((training_set, test_set), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_set) - n_past:]
#inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(n_past, len(inputs)):
    X_test.append(inputs[i-n_past:i, :])

X_test = np.array(X_test)

#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))
predicted_stock_price = regressor.predict(X_test)
z=np.zeros((len(predicted_stock_price),training_set.shape[1]-1))
predicted_stock_price = np.concatenate((predicted_stock_price,z),axis =1) 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)[:,0]

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
