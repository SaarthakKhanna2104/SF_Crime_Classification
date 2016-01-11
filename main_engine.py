from manipulate_data import get_numerical_data, split_data
import numpy as np
import network
import sys

if __name__ == "__main__":
	#### Create a parser for the project ######
	##### Pass train, test file using arguments #####
	data_x,data_y = get_numerical_data(sys.argv[1])
	train_x,test_x,train_y,test_y = split_data(data_x,data_y)
	train_data = zip(train_x,train_y)
	test_data = zip(test_x,test_y)
	n_hidden = 35 ##### value for size of hidden layer 
	nn = network.Network([len(train_x[0]),n_hidden,len(train_y[0])])
	nn.SGD(train_data,test_data) ##### Later add test data as well.
	print "DONE"
