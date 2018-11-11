import pandas as pd
import numpy as np
from os import path, getcwd
from sklearn.impute import SimpleImputer
from ast import literal_eval
from sklearn.preprocessing import StandardScaler

def preprocessor():
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
	sea_df[['host_is_superhost', 'zipcode']] = imp_mode.fit_transform(sea_df[['host_is_superhost', 'zipcode']])
	bos_df[['host_is_superhost', 'zipcode']] = imp_mode.fit_transform(bos_df[['host_is_superhost', 'zipcode']])

	imp_mean = SimpleImputer(strategy='mean')
	sea_df[['bathrooms', 'bedrooms', 'beds']] = imp_mode.fit_transform(sea_df[['bathrooms', 'bedrooms', 'beds']])
	bos_df[['bathrooms', 'bedrooms', 'beds']] = imp_mode.fit_transform(bos_df[['bathrooms', 'bedrooms', 'beds']])

	# Transforming certain features

	## 1) Zipcode: remove unnecessary details or typos 
	## extract numbers 
	sea_df.zipcode[sea_df.zipcode.str.contains('[^0-9]', regex=True)] = sea_df.zipcode.str.extract(r'(?:[^0-9])(\d+)')[0].value_counts().index[0]
	bos_df.zipcode = bos_df.zipcode.str.extract(r'(\d+)(?<![^0-9])')[0]


	## 2) Price & extra_people: str -> float
	## drop '$' and ','
	sea_df.price = sea_df.price.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)
	bos_df.price = bos_df.price.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)
	sea_df.extra_people = sea_df.extra_people.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)
	bos_df.extra_people = bos_df.extra_people.map(lambda str_price: str_price[1:]).str.replace(',', '').astype(float)

	## 3) host_verifications 
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

	## 4) amenities
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


