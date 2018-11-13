import pandas as pd
import numpy as np
from os import path, getcwd
from sklearn.impute import SimpleImputer
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def preprocessor():
	""" Preprocess two datasets - Seattle and Boston. 
		Input: None
		Output: sea_df - Normalized Seattle dataframe 
				bos_df - Normalized Boston dataframe
				sea_df_nostd - Seattle dataframe
				bos_df_nostd - Boston dataframe
	"""

	## read files
	cities = ['seattle', 'boston']
	# retrieve current directory
	cwd = path.join(getcwd(), 'data')

	## Read files to data frames
	sea_df = pd.read_csv(path.join(cwd, cities[0], 'listings.csv'), parse_dates=['last_scraped', 'first_review', 'last_review'])
	bos_df = pd.read_csv(path.join(cwd, cities[1], 'listings.csv'), parse_dates=['last_scraped', 'first_review', 'last_review'])

	## Unify columns
	bos_df = bos_df[sea_df.columns]

	## prune off features
	cols = ['id', 'host_id', 'host_verifications', 'host_is_superhost', 'zipcode', 'bathrooms', 'bedrooms', 'beds', 
        'bed_type', 'amenities', 'square_feet', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 
        'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights','maximum_nights', 'calendar_updated', 
        'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 
        'number_of_reviews', 'first_review', 'last_review', 'review_scores_rating','review_scores_accuracy', 
        'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 
        'review_scores_value', 'instant_bookable', 'cancellation_policy', 'require_guest_profile_picture', 
        'require_guest_phone_verification','calculated_host_listings_count', 'reviews_per_month']
	sea_df = sea_df[cols]
	bos_df = bos_df[cols]

	## To check features with missings
	temp1 = sea_df.isnull().sum()/sea_df.shape[0]
	temp2 = bos_df.isnull().sum() / bos_df.shape[0]

	## Let us drop features with more than 20% missings
	sea_df = sea_df.drop(columns=temp2[temp2>.2].index.values)
	bos_df = bos_df.drop(columns=temp2[temp2>.2].index.values)

	## impute mode and mean to missings
	imp_mode = SimpleImputer(strategy='most_frequent')
	sea_df[['host_is_superhost', 'zipcode', 'host_since']] = imp_mode.fit_transform(sea_df[['host_is_superhost', 'zipcode', 'host_since']])
	bos_df[['host_is_superhost', 'zipcode', 'host_since']] = imp_mode.fit_transform(bos_df[['host_is_superhost', 'zipcode', 'host_since']])

	imp_mean = SimpleImputer(strategy='mean')
	sea_df[['bathrooms', 'bedrooms', 'beds']] = imp_mode.fit_transform(sea_df[['bathrooms', 'bedrooms', 'beds']])
	bos_df[['bathrooms', 'bedrooms', 'beds']] = imp_mode.fit_transform(bos_df[['bathrooms', 'bedrooms', 'beds']])

	# Transforming certain features

	## 1) Zipcode: remove unnecessary details or typos
	## extract numbers 
	sea_df.zipcode[sea_df.zipcode.str.contains('[^0-9]', regex=True)] = sea_df.zipcode.str.extract(r'(?:[^0-9])(\d+)')[0].value_counts().index[0]
	bos_df.zipcode = bos_df.zipcode.str.extract(r'(\d+)(?<![^0-9])')[0]

	## 2) host_since: change to days (difference of last day of dataframe and host_since)
	## Get last day as baseline point
	sea_now = sea_df.host_since.max()
	bos_now = bos_df.host_since.max()
	## retrieve days of difference from basline date
	sea_df['host_days'] = sea_df.host_since.map(lambda x: (sea_now - x).days)
	bos_df['host_days'] = bos_df.host_since.map(lambda x: (bos_now - x).days)
	## drop host_since column
	sea_df = sea_df.drop(columns=['host_since'])
	bos_df = bos_df.drop(columns=['host_since'])

	## 3) Price & extra_people: str -> float
	## drop '$' and ','
	sea_df.price = sea_df.price.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)
	bos_df.price = bos_df.price.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)
	sea_df.extra_people = sea_df.extra_people.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)
	bos_df.extra_people = bos_df.extra_people.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)

	## 4) host_verifications 
	### seattle
	sea_df.host_verifications = sea_df.host_verifications.replace(['[]', 'None'], "['none']")
	### define seattle categorical dummy dataframe
	sea_cat = pd.get_dummies(sea_df.host_verifications.map(literal_eval).apply(pd.Series).stack(), prefix='host_ver').sum(level=0)
	### boston
	bos_df.host_verifications = bos_df.host_verifications.replace(['[]', 'None'], "['none']")
	### define boston categorical dummy dataframe
	bos_cat = pd.get_dummies(bos_df.host_verifications.map(literal_eval).apply(pd.Series).stack(), prefix='host_ver').sum(level=0)
	### unify column width and order
	bos_cat['host_ver_photographer'] = 0
	bos_cat.columns = sea_cat.columns
	## drop host_verifications from sea_df, bos_df
	sea_df = sea_df.drop(columns=['host_verifications'])
	bos_df = bos_df.drop(columns=['host_verifications'])

	## 5) amenities
	### change format to string list same as host_verifications
	sea_df['amenities'] = sea_df['amenities'].map(lambda d: [amenity.replace('"', "").replace("{", "").replace("}", "") for amenity in d.split(",")]).astype(str)
	bos_df['amenities'] = bos_df['amenities'].map(lambda d: [amenity.replace('"', "").replace("{", "").replace("}", "") for amenity in d.split(",")]).astype(str)

	### seattle
	sea_df.amenities = sea_df.amenities.replace("['']", "['none']")
	### boston
	bos_df.amenities = bos_df.amenities.replace("['']", "['none']")
	## temporary dataframe before adding to sea_cat
	temp1 = pd.get_dummies(sea_df.amenities.map(literal_eval).apply(pd.Series).stack(), prefix='amenities').sum(level=0)
	## temporary dataframe before adding to bos_cat
	temp2 = pd.get_dummies(bos_df.amenities.map(literal_eval).apply(pd.Series).stack(), prefix='amenities').sum(level=0)
	### unify column width and order
	cols = ['amenities_Free Parking on Street',
	        'amenities_Paid Parking Off Premises',
	        'amenities_translation missing: en.hosting_amenity_49',
	        'amenities_translation missing: en.hosting_amenity_50']
	for col in cols:
	    temp1[col] = 0
	temp2.columns = temp1.columns
	## concatenate dummy variable dataframes
	sea_cat = pd.concat([sea_cat, temp1], axis=1)
	bos_cat = pd.concat([bos_cat, temp2], axis=1)
	## drop amenities from sea_df, bos_df
	sea_df = sea_df.drop(columns=['amenities'])
	bos_df = bos_df.drop(columns=['amenities'])

	# Processing numerical variables - Standardization
	## Retrieve numerical features
	sea_num_no_std = sea_df.select_dtypes(include=[int, float]).drop(columns=['id', 'host_id', 'price'])
	bos_num_no_std = bos_df.select_dtypes(include=[int, float]).drop(columns=['id', 'host_id', 'price'])

	## make copy
	sea_num = sea_num_no_std.copy()
	bos_num = bos_num_no_std.copy()

	## standardizing them
	scaler = StandardScaler()
	sea_num[sea_num.columns] = scaler.fit_transform(sea_num_no_std)
	bos_num[bos_num.columns] = scaler.fit_transform(bos_num_no_std)

	# Processing categorical variables
	sea_cat_2 = sea_df.select_dtypes(include=[object]).drop(columns=['zipcode'])
	bos_cat_2 = bos_df.select_dtypes(include=[object]).drop(columns=['zipcode'])

	## get dummy variables for categoricals
	bos_cat_2 = pd.get_dummies(bos_cat_2)
	sea_cat_2 = pd.get_dummies(sea_cat_2)

	## make column width same
	for col in set(bos_cat_2.columns) - set(sea_cat_2.columns):
	    sea_cat_2[col] = 0
	    
	## make column order same
	sea_cat_2.columns = bos_cat_2.columns

	## merge all categorical dataframes
	sea_cat = pd.concat([sea_cat, sea_cat_2], axis=1)
	bos_cat = pd.concat([bos_cat, bos_cat_2], axis=1)

	#############
	## putting it all together (df, num, cat)
	sea_df = pd.concat([sea_df[['id', 'host_id', 'zipcode', 'price']], sea_num, sea_cat], axis=1)
	bos_df = pd.concat([bos_df[['id', 'host_id', 'zipcode', 'price']], bos_num, bos_cat], axis=1)

	## for EDA, getting not standardized numericals
	sea_df_nostd = pd.concat([sea_df[['id', 'host_id', 'zipcode', 'price']], sea_num_no_std, sea_cat], axis=1)
	bos_df_nostd = pd.concat([bos_df[['id', 'host_id', 'zipcode', 'price']], bos_num_no_std, bos_cat], axis=1)

	return sea_df, bos_df, sea_df_nostd, bos_df_nostd




def model_data(sea_df, bos_df, random_state=777):
	""" This function will train the model with XGBregressor. 
		Input: sea_df - Normalized Seattle dataframe
			   bos_df - Normalized Boston dataframe
			   random_state - default is 777
		Output: xgb_reg - XGBregressor model trained
				X_train, X_test, y_train, y_test - train, test datasets

	"""

	## add host property number info feature 
	sea_df['host_has'] = sea_df.groupby('host_id').transform('count')['id'].values
	bos_df['host_has'] = bos_df.groupby('host_id').transform('count')['id'].values

	## standardize it
	sea_df['host_has'] = scaler.fit_transform(sea_df[['host_has']])
	bos_df['host_has'] = scaler.fit_transform(bos_df[['host_has']])

	## add location frequency info feature  
	sea_df['zip_has'] = sea_df.groupby('zipcode').transform('count')['id'].values
	bos_df['zip_has'] = bos_df.groupby('zipcode').transform('count')['id'].values
	# sea_df = sea_df.drop(columns=['zipcode'])

	## standardize it
	sea_df['zip_has'] = scaler.fit_transform(sea_df[['zip_has']])
	bos_df['zip_has'] = scaler.fit_transform(bos_df[['zip_has']])

	## Add indicator 
	sea_df['Seattle'] = 1
	sea_df['Boston'] = 0
	bos_df['Seattle'] = 0
	bos_df['Boston'] = 1

	## concatenate data frames
	df = pd.concat([sea_df, bos_df], axis=0)
	df = df.reset_index(drop=True)

	## split into response and predictors, train and test data
	X = df.drop(columns=['id', 'host_id', 'zipcode', 'price'])
	y = df['price']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)


	## convert the dataset into an optimized data structure called Dmatrix that XGBoost supports
	data_dmatrix = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)

	## parameter dictionary holder
	params = []

	## candidate parameters
	max_depths = [3, 4, 5]
	gammas = [0, 0.1, 0.5, 1, 5]
	learning_rates = [0.05, 0.1, 0.5]
	subsamples = [0.7, 0.8, 1]
	reg_lambdas = [0.5, 1, 5, 10]
	colsample_bytrees = [0.7, 0.8, 1]

	## manually grid search cv
	for max_depth in max_depths:
	    for gamma in gammas:
	        for learning_rate in learning_rates:
	            for subsample in subsamples:
	                for reg_lambda in reg_lambdas:
	                    for colsample_bytree in colsample_bytrees:
	                    	## define hyper parameters for xgbregressor
	                        param = dict(max_depth=max_depth, 
	                                           gamma=gamma, 
	                                           learning_rate=learning_rate, 
	                                           subsample = subsample, 
	                                           reg_lambda=reg_lambda, 
	                                           colsample_bytree=colsample_bytree)
	                        ## retrieve cv results dataframe
	                        cv_results = xgb.cv(dtrain=data_dmatrix, params=param, nfold=5,
	                                            num_boost_round=50, early_stopping_rounds=20, 
	                                            metrics="rmse", as_pandas=True, seed=(random_state*29)%10000)
	                        cv_score = cv_results["test-rmse-mean"].iloc[-1]
	                        ## append the score and parameters to params
	                        params.append((cv_score, param))


	## get best parameters that gives lowest validation rmse score
	params = sorted(params, key=lambda x: x[0])
	best_params = params[0][1]
	best_valid_score = params[0][0]

	## train the model with best parameters
	xgb_reg = xgb.XGBRegressor(max_depth=best_params['max_depth'], gamma=best_params['gamma'], 
	                           learning_rate=best_params['learning_rate'], subsample=best_params['subsample'], 
	                           reg_lambda=best_params['reg_lambda'], colsample_bytree=best_params['colsample_bytree'], 
	                           n_estimators = 100)
	xgb_reg.fit(X_train, y_train)

	## print out result 
	print("======= CV Result =======")
	print("> best validation RMSE: {:.4f}".format(best_valid_score))
	print("> Best hyper parameters:")
	print("  ", best_params)
	
	return xgb_reg, X_train, X_test, y_train, y_test


def predict_model(xgb_reg, X_test, y_test, show=False, number=10):
	""" This function will predict values based on input data
		Input: xgb_reg - XGBregressor model trained 
			   X_test, y_test - test dataset (features and label)
		Output: preds - predicted values
				rmse - root mean square error 

	"""

	## predict
	y_hat = xgb_reg.predict(X_test)

	## evaluate on test data
	r2 = r2_score(y_test, y_hat)
	rmse = np.sqrt(mean_squared_error(y_test, y_hat))

	print("======= Result =======")
	print("> test r-squared: {:.4f}".format(r2))
	print("> test RMSE: {:.4f}".format(rmse))

	## visualize if show == True
	if show:
		features = sorted(zip(xgb_reg.feature_importances_, X_train.columns), reverse=True)
		attr = []
		coef = []
		number = 10

		for feature in features:
		    attr.append(feature[1]) 
		    coef.append(feature[0])

		plt.figure(figsize=(12,4))
		ax = plt.subplot()
		ax.bar(attr[:number], height=coef[:number], color='green', alpha=0.5)
		sns.despine(top=True, right=True, left=True)
		ax.xaxis.grid(False)
		plt.xticks(rotation=90)
		plt.xlabel('Features')
		plt.ylabel('Feature importance')
		plt.title('Feature importance for the Top {} features'.format(number), fontsize=16)
		plt.show()

	## print features
	print("Top {} important features:\n {}".format(number, attr[:number]))

	return preds, rmse

def feval_score(preds, dtrain):
	""" Customized r-squared evaluation metric function for cross validation for xgb.cv
	"""
    responses = dtrain.get_label()
    r2 = r2_score(responses, preds)
    return 'r2', r2


