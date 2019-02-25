
# Importing the libraries
#Basic libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
import pandas as pd
import random

#Scaling library
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#NN library
from keras.models import Sequential #For creating an Neural Network object
from keras.layers import Dense #adds output layer
from keras.layers import LSTM #adds LSTM layer
from keras.layers import Dropout

#Callbacks keras library
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
#random.seed(0)
#Find correlations between data. If high, then add to training set. 

# Importing the data along with its correlated.
#Aqui agrega los indices correlacionados y agregalos a la lista data_train
#Es importante que dataset_train sea el del indice que queramos predecir
path="data/"
dataset_train = pd.read_csv('data/AAPL.csv')
dataset_corr_1 = pd.read_csv('data/GOOG.csv')
dataset_corr_2 = pd.read_csv('data/BA.csv')
dataset_corr_3 = pd.read_csv('data/MSFT.csv')

data_train = (dataset_train['Open'], dataset_corr_1['Open'], dataset_corr_2['Open'], dataset_corr_3['Open'])


# Extracting only the "Open Value" Colunm of the Stocks
data_total = pd.concat(data_train, axis = 1).dropna()
training_set = data_total.iloc[:,:].values[:-30,:]
#training_set = dataset_train.iloc[:, 1:2].values

# Getting the real stock price of 2017

test_set = data_total.iloc[:,:].values[-30:,:]

real_stock_price = dataset_train.iloc[-30:, 1:2].values

# Feature Scaling: Standardisation or Normalization? 
#Better normalization for RNN, specially because we have sigmoid act. f.
#Creating the scaler object
#sc = MinMaxScaler(feature_range = (0, 1)) #Q: When is it convenient to change feature range?
sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure; how many timesteps to remember and how many days to predict into the future

#What the RNN needs to remember to predict. This is a very important step which if
#not done properly can lead to overfitting or nonsense predictions

#It is possible to add more dimensions, such as other indicators: stock prices
n_future = 1 # number of days to predict
n_past = 240 #number past days to remember for the prediction
X_train = []
y_train = []
for i in range(n_past, len(training_set_scaled)-n_future+1):
    X_train.append(training_set_scaled[i-n_past:i, :])
    y_train.append(training_set_scaled[i + n_future - 1: i + n_future , 0])

X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping
if len(data_train)==1:
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# import keras.backend as K
# from keras.layers import Lambda

 
# # def customLoss(y_true,y_pred):

# # 	def squared_differences(pair_of_tensors):
# # 		x, y = pair_of_tensors
# # 		return K.square(x - y)
	

# # 	y_numpy=K.eval(y_true)
# # 	y_desfasada=K.constant(value=np.append(y_numpy[0],y_numpy)[:-1])

# # 	y_true_slope=y_numpy - y_desfasada

# # 	y_numpy=K.eval(y_pred)
# # 	y_desfasada=K.constant(value=np.append(y_numpy[0],y_numpy)[:-1])

# # 	y_pred_slope = y_numpy - y_desfasada
# # 	y_desfasada=K.constant(value=np.append(y_numpy[0],y_numpy)[:-1])
# # 	square_diff = Lambda(squared_differences)([slope(y_true), slope(y_pred)])
# # 	return K.sum(square_diff)

# # def SlopeLoss():
# # 	def customLossOutput(y_true,y_pred):
# # 		return customLoss(y_true,y_pred)
# # 	return K.function((y_true,y_pred),customLossOutput)

# Initialising the RNN
def rnn_net(n_blocks,n_units):
	regressor = Sequential()
	# Adding the first LSTM layer and some Dropout regularisation
	regressor.add(LSTM(units = n_units, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
	regressor.add(Dropout(0.2))
	#units, lstm cells or memory units.
	for i in range(n_blocks):
	# Adding LSTM layers and some Dropout regularisation
		regressor.add(LSTM(units = n_units, return_sequences = True))
		regressor.add(Dropout(0.2))
	# Adding the output layer
	regressor.add(LSTM(units = n_units))
	regressor.add(Dropout(0.2))
	regressor.add(Dense(units = 1))
	#SlopeLossFunc=SlopeLoss()
	loss_f=RMSprop(lr=0.1)
	regressor.compile(optimizer = 'adam', loss = "mean_absolute_percentage_error")
	
	return regressor

n_blocks_list=[4,8,16,32,50]
n_units_list=[64,128,256,512,1024,2048]


for n_blocks in n_blocks_list:
	for n_units in n_units_list:
		regressor = rnn_net(n_blocks,n_units)
		#Define a loss function for our particular problem

		#import keras.backend as K
		#def derivative_squared_error(y_true, y_pred):
		#    y_true_f = y_true
		#    y_pred_f = K.flatten(y_pred)
		   # derivate_true = y_true_f-y_true_f
		    #derivate_pred = y_pred[i]-y_pred[i-1]    
		#    return y_true_f


		es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=40, verbose=1)
		rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1)
		mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
		tb = TensorBoard('logs')
		 
		# Fitting the RNN to the Training set
		history = regressor.fit(X_train, y_train, epochs = 100, callbacks = [rlr, tb, es,mcp],validation_split = 0.2, verbose = 1, batch_size = 64)

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
		mp.savefig('figures/'+str(n_blocks)+ ' blocks and '+ str(n_units) + ' units.png' )
		#plt.show()
		f=open("history/history of LSTM of "+str(n_blocks) + " blocks and " + str(n_units) + " units.txt",'w')
		f.write(str(history.history))
		f.close()