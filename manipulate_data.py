import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import train_test_split

def split_data(data_x,data_y):
	##### In order to have consistent result, we use random number generator. Otherwise we'll be getting different
	##### results in every run.
	train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,test_size=0.50,random_state=42)
	return (train_x,test_x,train_y,test_y)


def get_numerical_data(data_file):
	sf_df_data = pd.read_csv(data_file)
	# sf_df_test = pd.read_csv(test_file)

	###### Change Date to Month and Year #########
	sf_df_data["Dates"] = pd.to_datetime(sf_df_data["Dates"])
	sf_df_data["Year"],sf_df_data["Month"] = sf_df_data['Dates'].apply(lambda x: str(x.year)), sf_df_data['Dates'].apply(lambda x: str(x.month))

	# sf_df_test["Dates"] = pd.to_datetime(sf_df_test["Dates"])
	# sf_df_test["Year"],sf_df_test["Month"] = sf_df_test['Dates'].apply(lambda x: str(x.year)), sf_df_test['Dates'].apply(lambda x: str(x.month))

	print len(pd.unique(sf_df_data['Category'].values.ravel()).tolist())
	print pd.unique(sf_df_data['Category'].values.ravel()).tolist()

	######## To deal with categorical variables, we can make use of Pandas and DictVectorizer ###########
	cat_cols = ['Year','DayOfWeek','PdDistrict']
	num_cols = ['X','Y']

	num_data_X = sf_df_data[num_cols].as_matrix()
	# num_test_X = sf_df_test[num_cols].as_matrix()

	max_data = np.amax(abs(num_data_X),0)
	# max_test = np.amax(abs(num_test_X),0)   ### Normalising data

	num_data_X = num_data_X/max_data
	# num_test_X = num_test_X/max_test	

	cat_df_data_X = sf_df_data[cat_cols]
	cat_df_data_Y = sf_df_data[['Category']]
	cat_dict_data_X = cat_df_data_X.T.to_dict().values() # A list of dictionaries.
	cat_dict_data_Y = cat_df_data_Y.T.to_dict().values()

	# cat_df_test_X = sf_df_test[cat_cols]
	#cat_df_test_Y = sf_df_test[['Category']]
	# cat_dict_test_X = cat_df_test_X.T.to_dict().values()
	#cat_dict_test_Y = cat_df_test_Y.T.to_dict().values()

	vectorizer = DV(sparse=False)
	vec_data_X = vectorizer.fit_transform(cat_dict_data_X)
	data_Y = vectorizer.fit_transform(cat_dict_data_Y)
	# vec_test_X = vectorizer.fit_transform(cat_dict_test_X)
	#vec_test_Y = vectorizer.fit_transform(cat_dict_test_Y)

	# data_X = np.hstack((vec_data_X,num_data_X))   ##### remove the lat. and long. from the input data.
	# test_X = np.hstack((vec_test_X,num_test_X))

	data_X = vec_data_X

	print 'Done converting categorical data'
	return (data_X,data_Y)


# print "Shape of train X: ",np.shape(train_X)
# print "TYPE of train X: ",type(train_X)
# print "Shape of train Y: ",np.shape(train_Y)
# print "TYPE of train X: ",type(train_Y)
# print "Shape of test X: ",np.shape(test_X)
# print "TYPE of train X: ",type(test_X)
# print clf.predict(num_test_X)


