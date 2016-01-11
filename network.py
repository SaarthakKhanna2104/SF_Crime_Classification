import random
import numpy as np

class Network(object):
	def __init__(self,sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]] 
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

	def feedforward(self,a):
		for b,w in zip(self.biases,self.weights):
			a = sigmoid(np.dot(w,a)+b)
		return a

	def SGD(self,training_data,test_data,epochs=60,mini_batch_size=200,eta=3.0):
		if test_data:
			n_test = len(test_data)
		n = len(training_data)
		# print "length of training data: ",n
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)
			if test_data:
				print "Epoch {0}: {1}/{2}".format(j,self.evaluate(test_data),n_test)
			else:
				print "Epoch {0} complete".format(j)


	def update_mini_batch(self,mini_batch,eta):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x,y in mini_batch:
			delta_nabla_b,delta_nabla_w = self.backprop(x,y)
			nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]  ######## update the second part of SGD. Every neuron in the n/w is getting updated over every example in the mini-batch	
			nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)] #### SGD, update in W.
		self.biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]


	def backprop(self,x,y):
		x = np.reshape(x,(-1,1))
		y = np.reshape(y,(-1,1))
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		activation = x
		activations = [x]
		zs = []
		for b,w in zip(self.biases,self.weights):
			# print "w: ",np.shape(w)
			# print "b: ",np.shape(b)
			# print b
			# print "activation: ",np.shape(activation)
			z = np.dot(w,activation)+b
			zs.append(z)
			activation = sigmoid(z)
			# print "new activation: ",np.shape(activation)
			activations.append(activation)
		delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta,activations[-2].transpose())  ###change in code

		for l in xrange(2,self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
		return (nabla_b,nabla_w)

	def evaluate(self,test_data):
		# test_results = [(self.feedforward(np.reshape(x,(-1,1))), np.reshape(y,(-1,1))) for (x, y) in test_data]
		test_results = []
		for x,y in test_data:
			features = self.feedforward(np.reshape(x,(-1,1)))
			i = np.argmax(features)
			features[:] = 0
			features[i] = 1
			y = np.reshape(y,(-1,1))
			test_results.append([features,y])

		sum = 0
		for x,y in test_results:
			if (x == y).all():
				sum = sum + 1
		return sum

	def cost_derivative(self,output_activations,y):
		return (output_activations-y)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z) * (1-sigmoid(z))