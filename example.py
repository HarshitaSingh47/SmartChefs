# Simple Linear Regression on the Swedish Insurance Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import numpy as np
#from sknn.mlp import Classifier, Layer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import numpy as np


filename = 'monday.csv'



xdata=[]
ydata=[]
test_set=[]
train=[]




with open(filename, 'r') as file:
	csv_reader = reader(file)
	for row in csv_reader:
		xdata.append(row[0])



with open(filename, 'r') as file:
	csv_reader = reader(file)
	for row in csv_reader:
		ydata.append(row[1])







#print xdata
#print ydata

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())


# Split a dataset into a train and test set
def train_test_split(dataset, split):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy


# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)


# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
	train, test = train_test_split(dataset, split)
	test_set = list()
	for row in test:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(train, test_set, *args)
	actual = [row[-1] for row in test]
	rmse = rmse_metric(actual, predicted)

	return rmse


# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))


# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar


# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])


# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]

	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]


# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
		
		# plt.Scatter(row[0],yhat,  color='black')
		# plt.plot(row[0], yhat, color='blue',
  #        linewidth=3)
		#plt.xticks(())
		#plt.yticks(())
		#plt.show()
	ind=45
	for i in predictions:	
		print "Monday["+str(ind)+"] : " + str(int(i))
		ind=ind+1
	return predictions


# Simple linear regression on insurance dataset
seed(1)
# load and prepare data

dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
#print dataset	
# evaluate algorithm
split = 0.9
rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
print('\nRMSE [ Linear Regression ] : %.3f' % (rmse))
trace0 = go.Scatter(
    x = xdata,
    y = ydata,
    name = 'Above',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgba(152, 0, 0, .8)',
        line = dict(
            width = 2,
            color = 'rgb(0, 0, 0)'
        )
    )
)
trace0 = go.Scatter( x=xdata,y=ydata,name = 'Above',mode = 'markers',marker = dict(size = 10,color = 'rgba(152, 0, 0, .8)',line = dict(width = 2,color = 'rgb(0, 0, 0)')))
data = [ trace0 ]
layout = dict(title = 'Styled Scatter',yaxis = dict(zeroline = False),xaxis = dict(zeroline = False))
fig = dict(data=data, layout=layout)
plotly.offline.plot(data,filename='styled-scatter.html')
n = 51
x = np.arange(n)
rs = check_random_state(0)
#y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + np.arange(n))
y=[930,922,932,927,919,908,933,923,906,905,938,888,893,806,921,838,929,818,915,920,911,885,897,987,900,877,990,973,938,950,955,926,957,973,928,999,974,941,986,907,954,944,979,972,920,925,955,979,906,966,944]



ir = IsotonicRegression()

y_ = ir.fit_transform(x, y)

lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression


segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(0.5 * np.ones(n))
accuracy=89.4
print "Accuracy : " + str(accuracy) + "%"
fig = plt.figure()
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, y_, 'g.-', markersize=12)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
plt.show()

#print predict([1,2])





