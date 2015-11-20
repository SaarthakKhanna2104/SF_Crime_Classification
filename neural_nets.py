import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sknn.mlp import Classifier, Layer

sf_df_train = pd.read_csv('train.csv')
sf_df_test = pd.read_csv('test.csv')

###### Change Date to Month and Year #########
sf_df_train["Dates"] = pd.to_datetime(sf_df_train["Dates"])
sf_df_train["Year"],sf_df_train["Month"] = sf_df_train['Dates'].apply(lambda x: str(x.year)), sf_df_train['Dates'].apply(lambda x: str(x.month))

sf_df_test["Dates"] = pd.to_datetime(sf_df_test["Dates"])
sf_df_test["Year"],sf_df_test["Month"] = sf_df_test['Dates'].apply(lambda x: str(x.year)), sf_df_test['Dates'].apply(lambda x: str(x.month))


######## To deal with categorical variables, we can make use of Pandas and DictVectorizer ###########
cat_cols = ['Year','DayOfWeek','PdDistrict']
num_cols = ['X','Y']

num_train_X = sf_df_train[num_cols].as_matrix()
num_test_X = sf_df_test[num_cols].as_matrix()

max_train = np.amax(abs(num_train_X),0)
max_test = np.amax(abs(num_test_X),0)   ### Normalising data


num_train_X = num_train_X/max_train
num_test_X = num_test_X/max_test

cat_df_train_X = sf_df_train[cat_cols]
cat_df_train_Y = sf_df_train[['Category']]
cat_dict_train_X = cat_df_train_X.T.to_dict().values() # A list of dictionaries.
cat_dict_train_Y = cat_df_train_Y.T.to_dict().values()


cat_df_test_X = sf_df_test[cat_cols]
#cat_df_test_Y = sf_df_test[['Category']]
cat_dict_test_X = cat_df_test_X.T.to_dict().values()
#cat_dict_test_Y = cat_df_test_Y.T.to_dict().values()

vectorizer = DV(sparse=False)
vec_train_X = vectorizer.fit_transform(cat_dict_train_X)
train_Y = vectorizer.fit_transform(cat_dict_train_Y)
vec_test_X = vectorizer.fit_transform(cat_dict_test_X)
#vec_test_Y = vectorizer.fit_transform(cat_dict_test_Y)

train_X = np.hstack((vec_train_X,num_train_X))
test_X = np.hstack((vec_test_X,num_test_X))


clf = Classifier(layers=[Layer(type="Sigmoid",units=6),Layer(type="Sigmoid")],
	learning_rate=0.01,n_iter=50,batch_size=200)
clf.fit(train_X,train_Y)

# i = 0
# for l in test_X:
# 	i = i+1
# j = 0
# for l in train_Y:
# 	j = j + 1

# print i," ",j

print clf.predict(num_test_X)