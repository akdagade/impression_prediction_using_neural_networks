import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import pandas as pd
#pd.set_option('display.max_columns', 500)
seed = 17

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def init_bias(shape):
    biases = tf.random_normal(shape)
    return tf.Variable(biases)

def forwardprop(x, w_1, w_2, w_3, b_1, b_2, b3):
    """ Forward-propagation """
    h1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)
    h2 = tf.nn.relu(tf.matmul(h1, w_2) + b_2)
    y_predict = (tf.matmul(h2, w_3) + b_3)
    return y_predict

######## Import data from tsv file
np.set_printoptions(suppress=True)
dataset = pd.read_csv('input.csv',delimiter=',')
#print(dataset[0:10])

######## Seperate data into input features X and output y
no = dataset.iloc[:, [0]]
X = dataset.iloc[:, [2,3,4,5]]
#Y = dataset.iloc[:, [6,7,8,9]].values
Y = dataset.iloc[:, [6]].values
#print(X[0:10])
#print("~~~~~~~~~~~~~~~~~~~~~~~~")
#print(Y[0:10])
#print("~~~~~~~~~~~~~~~~~~~~~~~~")

######## One hot encoding ########
onehot_cols = [ 'day', 'Week', 'Quartermonth' ]
X_onehot = pd.get_dummies(X, columns = onehot_cols ).values
#one =  np.ones((X_onehot.shape[0],1))
#X_onehot = np.concatenate((one,X_onehot),axis=1)
#print(X_onehot[0:10])
#print(Y[0:10])

######## Split data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_onehot, Y, test_size = 0.2, random_state = seed)

print("X_train : " + str(X_train.shape))
print("Y_train : " + str(Y_train.shape))
print("X_test : " + str(X_test.shape))
print("Y_test : " + str(Y_test.shape))

######## Initialize Parameters
import tensorflow as tf

x_size = 42
y_size = 1
h_1_size = 32
h_2_size = 32

# Prepare input data
x = tf.placeholder(tf.float32, shape = [None, x_size])
y_label = tf.placeholder(tf.float32, shape = [None, y_size])

# Weight initializations
w_1 = init_weights((x_size, h_1_size))
w_2 = init_weights((h_1_size, h_2_size))
w_3 = init_weights((h_2_size, y_size))
b_1 = init_bias((h_1_size,))
b_2 = init_bias((h_2_size,))
b_3 = init_bias((y_size,))

# Forward propagation
y_predict = forwardprop(x, w_1, w_2, w_3, b_1, b_2, b_3)

# Backward propagation
cost = tf.reduce_mean(tf.square(y_predict - y_label))

updates = tf.train.AdamOptimizer(0.05).minimize(cost)



# Run
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
cst = []

for i in range(25000):
   loss, _ = sess.run([cost,updates], feed_dict = {x: X_train, y_label: Y_train})
   if i%5000==0:
   	print("Itr : " + str(i) + "  >> Cost : " + str(loss))
   cst.append(loss)


pred = sess.run(y_predict, feed_dict={x: X_test})
pred1 = sess.run(y_predict, feed_dict={x: X_onehot})
pred2 = sess.run(y_predict, feed_dict={x: X_test})
plt.figure("1")
plt.plot(np.reshape(range(95),(95,1)),pred1,'b')
plt.plot(np.reshape(range(95),(95,1)),Y,'r')
print('Cost:\n', loss)
#print('pred:\n', pred)
#print("Training Set:")
#print('Comparision: \n', np.concatenate((pred1,Y_train),axis=1))
print("Test Set:")
print('Comparision: \n', np.concatenate((pred,Y_test),axis=1))
plt.figure("2")
plt.plot(range(len(cst)),cst,'b')
plt.figure("3")
plt.plot(np.reshape(range(19),(19,1)),pred2,'b')
plt.plot(np.reshape(range(19),(19,1)),Y_test,'r')

pr = (np.abs((Y_test-pred2))/Y_test)*100
pr = pr < 15
perc = (np.sum(pr)/len(pr))*100
print("\nAccuracy : " + str(perc) + " %")
plt.show()
sess.close()
