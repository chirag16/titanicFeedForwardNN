import numpy as np
import pandas as pd

# Getting the data
data = pd.read_csv('train.csv')

# Feature selection
data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11]], axis=1, inplace=True)

# Extracting the labels
X = data.drop(data.columns[[0]], axis=1, inplace=False)
Y = data[['Survived']]

# Cleaning the data
sex = X[['Sex']].values
sex[sex == 'male'] = 0
sex[sex == 'female'] = 1
X[['Sex']] = sex

age = X[['Age']].values
max_age = max(age)
age /= max_age
X[['Age']] = age

# Deleting entries with NaN ages
to_be_deleted = []
for i in range(len(age)):
	if np.isnan(age[i]):
		to_be_deleted.append(i)
X.drop(X.index[to_be_deleted], axis=0, inplace=True)
Y.drop(Y.index[to_be_deleted], axis=0, inplace=True)

pclass = X[['Pclass']].values.astype(float)
max_pclass = max(pclass)
pclass /= max_pclass
X[['Pclass']] = pclass

sibsp = X[['SibSp']].values.astype(float)
max_sibsp = max(sibsp)
sibsp /= max_sibsp
X[['SibSp']] = sibsp

print X.head()
print "\n"

# Converting the pandas dataframes to numpy arrays
X_train = X.values
Y_train = Y.values

# Hyperparameters
n_input = 4
n_1 = 3
n_2 = 1
ALPHA = 1e-2
NUM_EPOCHS = 50 * len(X_train)

# Define the Parameters
W_1 = (np.random.rand(n_1, n_input) - 0.5) / 10
b_1 = (np.random.rand(n_1, 1) - 0.5) / 10
W_2 = (np.random.rand(n_2, n_1) - 0.5) / 10
b_2 = (np.random.rand(n_2, 1) - 0.5) / 10

# Sigmoid function
def sigmoid(z):
	return 1. / (1. + np.exp(-z))

# Forward Propagation
def forward_prop(x, W_1, b_1, W_2, b_2):
	z_1 = np.matmul(W_1, x) + b_1
	a_1 = sigmoid(z_1)
	z_2 = np.matmul(W_2, a_1) + b_2
	# z_2 = np.matmul(W_2, x) + b_2
	a_2 = sigmoid(z_2)

	return z_1, a_1, z_2, a_2			# Return stuff for back prop and the prediction (a_2)
	# return z_2, a_2			# Return stuff for back prop and the prediction (a_2)

# Stochastic Gradient Descent
for i in range(NUM_EPOCHS):
	i = i % len(X_train)				# Bound i between 0 and len(X_train)
	x = X_train[i].copy()				# Fetch the ith entry from the data	
	x = np.expand_dims(x, axis = 1) 	# To convert x to a column vector of shape = (n_input, 1)

	y = Y_train[i].copy()				# Fetch the ith target
	y = np.expand_dims(y, axis = 1)		# To convert y to a column vector of shape = (n_2, 1)

	# Run forward propagation on x
	z_1, a_1, z_2, a_2 = forward_prop(x, W_1, b_1, W_2, b_2)
	# z_2, a_2 = forward_prop(x, W_2, b_2)

	# Compute gradients for back propagation
	W_1_delta = np.matmul(np.multiply(W_2.T, np.multiply(a_1, 1. - a_1)), np.multiply(a_2 - y, x.T))
	b_1_delta = np.matmul(np.multiply(W_2.T, np.multiply(a_1, 1. - a_1)), a_2 - y)
	W_2_delta = np.matmul(a_2 - y, a_1.T)
	# W_2_delta = np.matmul(a_2 - y, x.T)
	b_2_delta = a_2 - y

	# Simultaneous update
	W_1 -= ALPHA * W_1_delta
	b_1 -= ALPHA * b_1_delta
	W_2 -= ALPHA * W_2_delta
	b_2 -= ALPHA * b_2_delta

	# Calculate and display loss
	loss = -y[0][0] * np.log(a_2[0][0]) - (1 - y[0][0]) * np.log(1. - a_2[0][0])
	# print 'loss:\t', loss

# Display final parameters
print 'Tuned parameters'
print '-' * 50
print 'W_1'
print '-' * 50
print W_1
print '-' * 50
print 'b_1'
print '-' * 50
print b_1
print '-' * 50
print 'W_2'
print '-' * 50
print W_2
print '-' * 50
print 'b_2'
print '-' * 50
print b_2

# Getting the data
data_test = pd.read_csv('test.csv')
labels = pd.read_csv('gender_submission.csv')

# Feature selection
data_test.drop(data_test.columns[[0, 2, 6, 7, 8, 9, 10]], axis=1, inplace=True)

# Extracting the labels
X_test = data_test
Y_test = labels[['Survived']]

# Cleaning the data
sex = X_test[['Sex']].values
sex[sex == 'male'] = 0
sex[sex == 'female'] = 1
X_test[['Sex']] = sex

age = X_test[['Age']].values
max_age = max(age)
age /= max_age
X_test[['Age']] = age

# Deleting entries with NaN ages
to_be_deleted = []
for i in range(len(age)):
	if np.isnan(age[i]):
		to_be_deleted.append(i)
X_test.drop(X_test.index[to_be_deleted], axis=0, inplace=True)
Y_test.drop(Y_test.index[to_be_deleted], axis=0, inplace=True)

pclass = X_test[['Pclass']].values.astype(float)
max_pclass = max(pclass)
pclass /= max_pclass
X_test[['Pclass']] = pclass

sibsp = X_test[['SibSp']].values.astype(float)
max_sibsp = max(sibsp)
sibsp /= max_sibsp
X_test[['SibSp']] = sibsp

X_test = X_test.values
Y_test = Y_test.values

# Making predictions
predictions = forward_prop(X_test.T, W_1, b_1, W_2, b_2)[-1][0]
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0

# Calculating the accuracy
accuracy = 0.
Y_test = Y_test.flatten()
for i in range(len(predictions)):
	if predictions[i] == Y_test[i]:
		accuracy += 1
accuracy /= len(predictions)

print
print '-' * 50
print 'Top 10 predictions'
print '-' * 50
print predictions[:10]

print
print '-' * 50
print 'Top 10 labels'
print '-' * 50
print Y_test[:10]


print
print '-' * 50
print 'Accuracy over X_test'
print '-' * 50
print accuracy

