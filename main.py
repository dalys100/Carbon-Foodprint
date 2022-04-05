# import packages to start data cleaning
import numpy as np
import pandas as pd
from geopy.distance import distance
import matplotlib.pyplot as plt
import seaborn as sns



## DATA CLEANING
# CALCULATING DATA FOR TRANSPORT BY SHIPS
# Read file and delete all rows without values for column 'country'
harbor = pd.read_excel("worldwide_port_data.xlsx")
harbor = pd.DataFrame(harbor)
harbor = harbor.replace(r'^\s*$', np.NaN, regex=True)
harbor = harbor.dropna(subset=['country'])
print('We will use these two data frames to calculate the co2-values associated with transport.')
print('overview of harbor data:')
print(harbor.head())
# Multiple ports reported per country. We will drop every port except one per country to simplify calculations.

# get unique country names (n=179)
uniqueCountries = harbor['country'].unique()

# dropping ALL duplicate values in column 'country' except for one
harbor.sort_values("country", inplace=True)
harbor.drop_duplicates(subset="country",
                       keep='first', inplace=True)

# create dictionary with country name as key and harbor coordinates as values
d = dict([(country, [latitude, longitude]) for country, latitude, longitude in
          zip(harbor.country, harbor.latitude, harbor.longitude)])

# norming Hamburg as the standard German harbor
germany_coord = 53.53577, 9.98743
countries = d

# calculate difference between Hamburg and each other country (in metres)
# and store country name and associated distance into a dictionary (in kilometres)
keys = []
values = []
for countries, coord in countries.items():
    dist = distance(germany_coord, coord).m
    # print(countries, dist)
    keys.append(countries)
    values.append(dist / 1000)

result = dict(zip(keys, values))

# include calculated distance as a new column in the 'harbor' data frame
harbor['distance_km'] = harbor['country'].apply(lambda x: result.get(x)).fillna('')

# calculate ship weight
# 1 filled container = 22 tonnes fruit + 3 tonne container --> 5000 containers per ship
# empty ship weighs 100000 tonnes
weight_ship = (25 * 5000) + 100000

# calculate CO2-emissions (in kg per 1kg produce) [ CO2-emission per ton per km = 0.015kg]
harbor['ship_co2_per_kg'] = ((harbor['distance_km'] * 0.015 * weight_ship) / (5000 * 22)) / 1000

# CALCULATING DATA FOR TRANSPORT BY PLANE
# Read file and delete all rows without values for column 'country'
airport = pd.read_excel("airport_data.xlsx")
airport = pd.DataFrame(airport)
airport = airport.replace(r'^\s*$', np.NaN, regex=True)
airport = airport.dropna(subset=['Airport Country'])
print('overview of airport data:')
print(airport.head())
# Multiple airports reported per country. We drop every airport except one per country to simplify calculations.

# get unique country names (n=237)
uniqueCountries2 = airport['Airport Country'].unique()

# dropping ALL duplicate values in column country except for one
airport.sort_values("Airport Country", inplace=True)
airport.drop_duplicates(subset="Airport Country",
                        keep='first', inplace=True)
airport['Country'] = airport['Airport Country']

# create dictionary with country name as key and airport coordinates as values
d2 = dict([(Country, [Latitude, Longitude]) for Country, Latitude, Longitude in
           zip(airport.Country, airport.Latitude, airport.Longitude)])

# norming Frankfurt am Main as the standard German airport
germany_coord2 = 50.033333, 8.570556
countries2 = d2

# calculate difference between Frankfurt and each other country (in metres)
# and store country name and associated distance into a dictionary (in kilometres)
keys2 = []
values2 = []
for countries2, coord in countries2.items():
    dist2 = distance(germany_coord2, coord).m
    # print(countries, dist)
    keys2.append(countries2)
    values2.append(dist2 / 1000)

result2 = dict(zip(keys2, values2))

# include calculated distance as new column in dataframe
airport['distance_km'] = airport['Country'].apply(lambda x: result2.get(x)).fillna('')

# an average air-freight plane can carry 70 tonnes of produce
# calculate CO2-emissions (in kg per 1kg produce) [1 flown kilometer = 3.65 kg co2]
airport['plane_co2_per_kg'] = (airport['distance_km'] * 3.65) / 70000
# print data frames
print('Show the calculated co2 values for both sea and air transport:')
print(harbor['ship_co2_per_kg'])
print(airport['plane_co2_per_kg'])



## GENERATE DATAFRAME FOR REGIONAL PRODUCE - GERMANY VERSION
# import base value data
base_data = pd.read_excel("fruit_veggies_agriculture_base_CO2.xlsx")
base_data = pd.DataFrame(base_data)

# generate 'produce' and 'base co2' columns
# 'produce' indicates the name of the fruit or vegetable
# 'base co2' indicates the co2 emissions associated with growing that fruit or vegetable
fruit_veggie = base_data['produce']
base_value = base_data['CO2_base']
# each fruit/veggie gets 4 rows so both naturally-grown, greenhouse-grown, organic and non-organic produce
# is represented in the dataframe
produce = [fruit_veggie[i // 4] for i in range(len(fruit_veggie) * 4)]
base_co2 = [base_value[i // 4] for i in range(len(base_value) * 4)]
produce = pd.DataFrame(produce)
base_co2 = pd.DataFrame(base_co2)
produce.columns = ['produce']
base_co2.columns = ['base_co2']

# generate origin country column (only Germany due to regional produce)
origin_country_regional = pd.DataFrame(['Germany'] * len(fruit_veggie) * 4)
origin_country_regional.columns = ['origin_country']

# generate 'organic' columns
# 'organic_produce' indicates whether the produce is organic or not
# 'organic_value' indicates how much less co2 emissions (in %) get generated by organic agriculture
# organic produce generates 15% less emissions than traditionally grown produce
org_name = pd.DataFrame(['organic', 'not organic'] * 74)
org_value = pd.DataFrame([15, 0] * 74)
organic_data = pd.concat([org_name, org_value], axis=1).reindex(org_name.index)
organic_data.columns = ['organic_produce', 'organic_value']

# generate 'greenhouse' columns
# 'greenhouse_produce' indicates whether the produce was grown in a greenhouse or not
# 'greenhouse_value' indicates how much co2 emissions (in kg) get generated through greenhouse agriculture
# greenhouse-grown produce generates an extra 2.5 kg of co2 emissions
gh_name = pd.DataFrame(['greenhouse', 'greenhouse', 'no greenhouse', 'no greenhouse'] * 37)
gh_value = pd.DataFrame([2.5, 2.5, 0, 0] * 37)
gh_name.columns = ['greenhouse_produce']
gh_value.columns = ['greenhouse_value']
greenhouse_data = pd.concat([gh_name, gh_value], axis=1).reindex(gh_name.index)

# generate main dataframe for regional produce and calculate final co2 values
data_reg = pd.concat([produce, base_co2, origin_country_regional, greenhouse_data, organic_data], axis=1)
data_reg['final_co2'] = (data_reg['base_co2'] + data_reg['greenhouse_value']) * (1 - (data_reg['organic_value'] / 100))
print('The following three temporary data frames show how the final data frame will be structured.')
print('--> data structure of data frame for regional produce:')
print(data_reg.head())



## GENERATE DATAFRAME FOR OVERSEAS PRODUCE - AIR FREIGHT VERSION
# generate produce and base co2 value columns
produce_air = [fruit_veggie[i // (len(uniqueCountries2) * 2)] for i in
               range((len(uniqueCountries2) * len(fruit_veggie) * 2))]
base_co2_air = [base_value[i // (len(uniqueCountries2) * 2)] for i in
                range((len(uniqueCountries2) * len(fruit_veggie) * 2))]
produce_air = pd.DataFrame(produce_air)
base_co2_air = pd.DataFrame(base_co2_air)
produce_air.columns = ['produce']
base_co2_air.columns = ['base_co2']

# generate organic data columns (same logic like before)
org_name_air = pd.DataFrame(['organic', 'not organic'] * (len(uniqueCountries2) * len(fruit_veggie)))
org_value_air = pd.DataFrame([15, 0] * (len(uniqueCountries2) * len(fruit_veggie)))
organic_data_air = pd.concat([org_name_air, org_value_air], axis=1).reindex(org_name_air.index)
organic_data_air.columns = ['organic_produce', 'organic_value']

# generate origin country column (same logic like before)
origin_country_air = [uniqueCountries2[i // 2] for i in range(len(uniqueCountries2) * 2)]
origin_country_air = pd.DataFrame(origin_country_air)
origin_country_air.columns = ['origin_country']
origin_country_air = pd.concat([origin_country_air] * len(fruit_veggie), ignore_index=True)

# generate transport data columns
# 'transport_type' indicates which transport vehicle is used
# 'plane_co2_per_kg' indicates the associated co2 value of using a plane to transport produce
transport_plane = pd.DataFrame(['airplane'] * (len(uniqueCountries2) * len(fruit_veggie) * 2))
transport_plane = pd.DataFrame(transport_plane)
transport_plane.columns = ['transport_type']

transport_plane_value1 = np.array(airport['plane_co2_per_kg'])
transport_plane_value = [transport_plane_value1[i // 2] for i in range(len(transport_plane_value1) * 2)]
transport_plane_value = pd.DataFrame(transport_plane_value)
transport_plane_value.columns = ['plane_co2_per_kg']
transport_plane_value = pd.concat([transport_plane_value] * len(fruit_veggie), ignore_index=True)

# generate main dataframe for air freight and calculate final co2 values
data_air = pd.concat([produce_air, base_co2_air, origin_country_air, transport_plane, transport_plane_value,
                      organic_data_air], axis=1)
data_air['final_co2'] = (data_air['base_co2'] + data_air['plane_co2_per_kg']) * (1 - (data_air['organic_value'] / 100))
print('data structure of air freight data frame:')
print(data_air.head())



## GENERATE DATAFRAME FOR OVERSEAS PRODUCE - SEA FREIGHT VERSION
# generate produce and base co2 value columns
produce_sea = [fruit_veggie[i // (len(uniqueCountries) * 2)] for i in
               range((len(uniqueCountries) * len(fruit_veggie) * 2))]
base_co2_sea = [base_value[i // (len(uniqueCountries) * 2)] for i in
                range((len(uniqueCountries) * len(fruit_veggie) * 2))]
produce_sea = pd.DataFrame(produce_sea)
base_co2_sea = pd.DataFrame(base_co2_sea)
produce_sea.columns = ['produce']
base_co2_sea.columns = ['base_co2']

# generate organic data columns (same logic like before)
org_name_sea = pd.DataFrame(['organic', 'not organic'] * (len(fruit_veggie) * len(uniqueCountries)))
org_value_sea = pd.DataFrame([15, 0] * (len(uniqueCountries) * len(fruit_veggie)))
organic_data_sea = pd.concat([org_name_sea, org_value_sea], axis=1).reindex(org_name_sea.index)
organic_data_sea.columns = ['organic_produce', 'organic_value']

# generate origin country column
origin_country_sea = [uniqueCountries[i // 2] for i in range((len(uniqueCountries) * 2))]
origin_country_sea = pd.DataFrame(origin_country_sea)
origin_country_sea.columns = ['origin_country']
origin_country_sea = pd.concat([origin_country_sea] * len(fruit_veggie), ignore_index=True)

# generate transport data columns
transport_ship = pd.DataFrame(['ship'] * (len(uniqueCountries) * len(fruit_veggie) * 2))
transport_ship = pd.DataFrame(transport_ship)
transport_ship.columns = ['transport_type']

transport_ship_value1 = np.array(harbor['ship_co2_per_kg'])
transport_ship_value = [transport_ship_value1[i // 2] for i in range((len(transport_ship_value1) * 2))]
transport_ship_value = pd.DataFrame(transport_ship_value)
transport_ship_value.columns = ['ship_co2_per_kg']
transport_ship_value = pd.concat([transport_ship_value] * len(fruit_veggie), ignore_index=True)

# generate main dataframe and calculate final co2 values
data_ship = pd.concat([produce_sea, base_co2_sea, origin_country_sea, transport_ship, transport_ship_value,
                       organic_data_sea], axis=1)
data_ship['final_co2'] = (data_ship['base_co2'] + data_ship['ship_co2_per_kg']) * \
                         (1 - (data_ship['organic_value'] / 100))
print('data structure of sea freight data frame:')
print(data_ship.head())



# CREATE FINAL DATA FRAME
# merge data_reg, data_ship and data_air together into a single data frame
data_fin = pd.merge(data_air, data_ship, how='outer')
data_final = pd.merge(data_fin, data_reg, how='outer')

# compute final column 'CO2 Score' and integrate into final dataframe
# determine cutoff-values for class A-E according to quantiles
print(np.quantile(data_final['final_co2'], 0.2))
print(np.quantile(data_final['final_co2'], 0.4))
print(np.quantile(data_final['final_co2'], 0.6))
print(np.quantile(data_final['final_co2'], 0.8))

# round up the calculated values and set them as intervals for each climate score category
conditions = [
    (data_final['final_co2'] < 0.35),
    (data_final['final_co2'] >= 0.35) & (data_final['final_co2'] < 0.45),
    (data_final['final_co2'] >= 0.45) & (data_final['final_co2'] < 0.6),
    (data_final['final_co2'] >= 0.6) & (data_final['final_co2'] < 0.8),
    (data_final['final_co2'] >= 0.8)
]
values = ['A', 'B', 'C', 'D', 'E']
data_final['co2_score'] = np.select(conditions, values)

# sort columns of the data frame into this specific order
data_final = data_final[['produce', 'base_co2', 'origin_country', 'transport_type', 'plane_co2_per_kg',
                         'ship_co2_per_kg', 'greenhouse_produce', 'greenhouse_value', 'organic_produce',
                         'organic_value', 'final_co2', 'co2_score']]
print('Data structure of the final data frame:')
print(data_final.head())



## GETTING AN OVERVIEW OF OUR FINAL DATA FRAME
# create descriptive overview over our final data set
print('Creating an overview on our data.')
print(data_final.describe().T)
print('Number of unique values in each column:')
print(data_final.nunique())

# check for duplicate rows
print('Number of duplicate rows:')
print(data_final.duplicated().sum())

# check distribution of numeric variables
fig, axs = plt.subplots(2, 2, figsize=(7, 7))
sns.histplot(data_final['base_co2'], ax=axs[0, 0])
sns.histplot(data_final['plane_co2_per_kg'], ax=axs[0, 1])
sns.histplot(data_final['ship_co2_per_kg'], ax=axs[1, 0])
sns.histplot(data_final['final_co2'], ax=axs[1, 1])
plt.show()
print('The variables base_co2 and final_co2 are right-skewed.')

# check distribution of co2 scores
sns.set(style="whitegrid")
fig = plt.figure(figsize=(10, 6))
plt.title("Distribution of the climate scores", size=20, pad=26)
sns.countplot('co2_score', data=data_final, palette='pastel')
plt.show()

# generate correlation matrix
plt.figure(figsize=(15, 15))
sns.heatmap(data_final.corr(), annot=True, cmap='PuBuGn')
plt.title("Correlation matrix", size=20, pad=26)
plt.show()



## START DATA MODELING
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


# Preprocessing data for performing machine learning algorithms
# fill all NA's in the data frame
data_final = data_final.fillna(0)
mydf = data_final

# check variable types
print(mydf.info)
# all 'object' type variables need to be encoded

# encode all categorical variables
mydf_replace = mydf.copy()
labels_produce = mydf_replace['produce'].astype('category').cat.categories.tolist()
replace_produce = {'produce': {k: v for k, v in zip(labels_produce, list(range(1, len(labels_produce) + 1)))}}
labels_country = mydf_replace['origin_country'].astype('category').cat.categories.tolist()
replace_country = {'origin_country': {k: v for k, v in zip(labels_country, list(range(1, len(labels_country) + 1)))}}
labels_transport = mydf_replace['transport_type'].astype('category').cat.categories.tolist()
replace_transport = {
    'transport_type': {k: v for k, v in zip(labels_transport, list(range(1, len(labels_transport) + 1)))}}
labels_greenhouse = mydf_replace['greenhouse_produce'].astype('category').cat.categories.tolist()
replace_greenhouse = {
    'greenhouse_produce': {k: v for k, v in zip(labels_greenhouse, list(range(1, len(labels_greenhouse) + 1)))}}
labels_organic = mydf_replace['organic_produce'].astype('category').cat.categories.tolist()
replace_organic = {'organic_produce': {k: v for k, v in zip(labels_organic, list(range(1, len(labels_organic) + 1)))}}
labels_co2score = mydf_replace['co2_score'].astype('category').cat.categories.tolist()
replace_co2score = {'co2_score': {k: v for k, v in zip(labels_co2score, list(range(1, len(labels_co2score) + 1)))}}

mydf_replace.replace(replace_produce, inplace=True)
mydf_replace.replace(replace_country, inplace=True)
mydf_replace.replace(replace_transport, inplace=True)
mydf_replace.replace(replace_greenhouse, inplace=True)
mydf_replace.replace(replace_organic, inplace=True)
mydf_replace.replace(replace_co2score, inplace=True)

# determining training and test data + normalizing data
y = mydf_replace.co2_score
mydf_features = ['produce', 'origin_country', 'transport_type', 'plane_co2_per_kg', 'ship_co2_per_kg',
                 'greenhouse_produce', 'organic_produce']
X = mydf_replace[mydf_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



# START ALGORITHMS
print('Machine Learning models are applied. Results:')
# Decision Tree Regressor model
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train, y_train)
prediction_dtr = dtr.predict(X_test)
print('Report for Decision Tree:')
print(classification_report(y_test, prediction_dtr))
print(confusion_matrix(y_test, prediction_dtr))

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
prediction_rfc = rfc.predict(X_test)
print('Report for Random Forest Classifier:')
print(classification_report(y_test, prediction_rfc))
print(confusion_matrix(y_test, prediction_rfc))

# Support Vector Classifier
svc = SVC()
svc.fit(X_train, y_train)
prediction_svc = svc.predict(X_test)
print('Report for Support Vector Classifier:')
print(classification_report(y_test, prediction_svc))
print(confusion_matrix(y_test, prediction_svc))

# Stochastic Gradient Descent Classifier
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
prediction_sgd = sgd.predict(X_test)
print('Report for Stochastic Gradient Descent Classifier:')
print(classification_report(y_test, prediction_sgd))
print(confusion_matrix(y_test, prediction_sgd))

# Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
prediction_gnb = gnb.predict(X_test)
print('Report for Naive Bayes:')
print(classification_report(y_test, prediction_gnb))
print('####')
print('Random Forest Classifier is chosen as the best model. It has the highest accuracy right after the Decision Tree model.')
print('The Random Forest prevents overfitting by using multiple trees which creates a more accurate result for our use case.')
print('We use accuracy as the main performance evaluator as falsely predicted values to do not cause significant harm for the intended user.')



## FINE-TUNING THE MODEL
# randomized parameter optimization
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
# create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, search across 100 different combinations,
# and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# fit the random search model
rf_random.fit(X_train, y_train)
print('These are the best parameters determined by Randomized Parameter Optimization:')
print(rf_random.best_params_)


# grid search optimization
# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [2, 3, 4],
    'min_samples_split': [3, 5, 7],
    'n_estimators': [100, 200, 300, 1600]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print('These are the best parameters determined by Grid Search Optimization:')
print(grid_search.best_params_)


# compare model performance
# base model
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
prediction_rfc = rfc.predict(X_test)
print('Fine Tuning - Comparing model performance')
print('base model (Random Forest):')
print(classification_report(y_test, prediction_rfc))

# improved model - randomized search
rfc_opt = RandomForestClassifier(n_estimators=800,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 max_features='auto',
                                 max_depth=100,
                                 bootstrap=True)
rfc_opt.fit(X_train, y_train)
prediction_rfc_opt = rfc_opt.predict(X_test)
print('Randomized Search Cross-Validation (Random Forest):')
print(classification_report(y_test, prediction_rfc_opt))

# improved model - grid search
rfc_opt2 = RandomForestClassifier(n_estimators=300,
                                  min_samples_split=5,
                                  min_samples_leaf=2,
                                  max_features=3,
                                  max_depth=90,
                                  bootstrap=True)
rfc_opt2.fit(X_train, y_train)
prediction_rfc_opt2 = rfc_opt2.predict(X_test)
print('Grid Search Cross-Validation (Random Forest):')
print(classification_report(y_test, prediction_rfc_opt2))
print(confusion_matrix(y_test, prediction_rfc_opt2))
print('Fine-Tuning leads to a final accuracy of 92%.')



## PERFORMANCE EVALUATION
# summarize feature importance
feat_importances = pd.Series(rfc_opt2.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()
print('The most important features are: produce, plane_co2_per_kg and ship_co2_per_kg.')
print('This makes sense as the CO2 emissions from transportation have the highest variance in values among our variables.')
