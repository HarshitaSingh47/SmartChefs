import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

swipes = []
sid = []

def get_data(filename):
	with open(filename,'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) #skipping column names
		for row in csvFileReader:
			sid.append(int(row[0]))
			swipes.append(int(row[1]))
	return

def show_plot(sid,swipes):
	linear_mod = linear_model.LinearRegression()
	sid = np.reshape(sid,(len(sid),1)) # converting to matrix of n X 1
	swipes = np.reshape(swipes,(len(swipes),1))
	linear_mod.fit(sid,swipes) #fitting the data points in the model
	plt.scatter(sid,swipes,color='yellow') #plotting the initial datapoints 
	plt.plot(sid,linear_mod.predict(sid),color='blue',linewidth=3) #plotting the line made by linear regression
	plt.show()
	return

def predict(sid,swipes,x):
	linear_mod = linear_model.LinearRegression() #defining the linear regression model
	sid = np.reshape(sid,(len(sid),1)) # converting to matrix of n X 1
	swipes = np.reshape(swipes,(len(swipes),1))
	linear_mod.fit(sid,swipes) #fitting the data points in the model
	prediction = linear_mod.predict(x)

	return prediction[0][0],linear_mod.coef_[0][0] ,linear_mod.intercept_[0]

get_data('monday.csv') # calling get_data method by passing the csv file to it
#print sid
#print swipes
print "\n"

show_plot(sid,swipes) 
#image of the plot will be generated. Save it if you want and then Close it to continue the execution of the below code.
for i in range(55, 60):
	prediction, coefficient, constant = predict(sid,swipes, i)  
	#print mean_squared_error(swipes,predictio)

	print "The number of swipes on " + str(i) +"th Monday: ",str(prediction)
	print "The regression coefficient is ",str(coefficient),", and the constant is ", str(constant)


#print "the relationship equation between dates and prices is: price = ",str(coefficient),"* date + ",str(constant) 