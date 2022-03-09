import numpy as np
import pandas as pd
from geopy.distance import distance
import matplotlib.pyplot as plt

# CALCULATING TRANSPORT DATA FOR SHIPS (DONE)
# Read file and delete all rows without values for column 'country'
harbor = pd.read_excel("C:/Users/Daria/Documents/datasets_techlabs/worldwide_port_data.xlsx")
harbor = pd.DataFrame(harbor)
harbor = harbor.replace(r'^\s*$', np.NaN, regex=True)
harbor = harbor.dropna(subset=['country'])

# get unique country names (n=179)
uniqueCountries = harbor['country'].unique()

# dropping ALL duplicate values in column country except for one
harbor.sort_values("country", inplace=True)
harbor.drop_duplicates(subset="country",
                       keep='first', inplace=True)
# making sure that only one coordinate per country remains (rows=179)
# print(harbor)

# create dictionary with country name as key and coordinates as values
d = dict([(country, [latitude, longitude]) for country, latitude, longitude in
          zip(harbor.country, harbor.latitude, harbor.longitude)])

# norming Hamburg as the standard German harbor
germany_coord = 53.53577, 9.98743
countries = d

# calculate difference between Hamburg and each other country (in metres)
# and store country name and associated distance into dictionary (in kilometres)
keys = []
values = []
for countries, coord in countries.items():
    dist = distance(germany_coord, coord).m
    # print(countries, dist)
    keys.append(countries)
    values.append(dist / 1000)

myresult = dict(zip(keys, values))
# print(myresult)

# include calculated distance as new column in dataframe
harbor['distance_km'] = harbor['country'].apply(lambda x: myresult.get(x)).fillna('')

# 1 filled container = 22 tonnes fruit + 3 tonne container --> 5000 containers per ship
weight_ship = (25 * 5000) + 100000
# calculate CO2-emissions (in kg per 1kg produce) [ CO2-emission per ton per km = 0.015kg]
harbor['ship_co2_per_kg'] = ((harbor['distance_km'] * 0.015 * weight_ship) / (5000 * 22)) / 1000
# print(harbor) #!!!


# CALCULATING TRANSPORT DATA FOR PLANES (DONE)
# Read file and delete all rows without values for column 'country'
airport = pd.read_excel("C:/Users/Daria/Documents/datasets_techlabs/airport_data.xlsx")
airport = pd.DataFrame(airport)
airport = airport.replace(r'^\s*$', np.NaN, regex=True)
airport = airport.dropna(subset=['Airport Country'])

# get unique country names (n=179)
uniqueCountries2 = airport['Airport Country'].unique()

# dropping ALL duplicate values in column country except for one
airport.sort_values("Airport Country", inplace=True)
airport.drop_duplicates(subset="Airport Country",
                        keep='first', inplace=True)
airport['Country'] = airport['Airport Country']

# create dictionary with country name as key and coordinates as values
d2 = dict([(Country, [Latitude, Longitude]) for Country, Latitude, Longitude in
           zip(airport.Country, airport.Latitude, airport.Longitude)])

# norming Frankfurt am Main as the standard German airport
germany_coord2 = 50.033333, 8.570556
countries2 = d2

# calculate difference between Frankfurt and each other country (in metres)
# and store country name and associated distance into dictionary (in kilometres)
keys2 = []
values2 = []
for countries2, coord in countries2.items():
    dist2 = distance(germany_coord2, coord).m
    # print(countries, dist)
    keys2.append(countries2)
    values2.append(dist2 / 1000)

myresult2 = dict(zip(keys2, values2))
# print(myresult2)

# include calculated distance as new column in dataframe
airport['distance_km'] = airport['Country'].apply(lambda x: myresult2.get(x)).fillna('')

# calculate CO2-emissions (in kg per 1kg produce) [1 flown kilometer = 3.65 kg CO2-e]
airport['plane_co2_per_kg'] = (airport['distance_km'] * 3.65) / 70000
# print(airport) #!!!


## GENERATE DATAFRAME FOR REGIONAL PRODUCE - GERMANY VERSION
# import base value data
base_data = pd.read_excel("C:/Users/Daria/Documents/datasets_techlabs/fruit_veggies_agriculture_base_CO2.xlsx")
base_data = pd.DataFrame(base_data)

# import origin countries of different fruits
fruit_origin = pd.read_excel("C:/Users/Daria/Documents/datasets_techlabs/fruit_origin_country.xlsx")
fruit_origin = pd.DataFrame(fruit_origin)

# import origin countries of different vegetables
veggie_origin = pd.read_excel("C:/Users/Daria/Documents/datasets_techlabs/veggie_country_data.xlsx")
veggie_origin = pd.DataFrame(veggie_origin)

# generate 'organic' columns
org_name = pd.DataFrame(['organic', 'not organic'] * 74)
org_value = pd.DataFrame([15, 0] * 74)
organic_data = pd.concat([org_name, org_value], axis=1).reindex(org_name.index)
organic_data.columns = ['organic_produce', 'organic_value']
# print(organic_data)

# generate 'greenhouse/storage' columns
gh_name = pd.DataFrame(['greenhouse', 'greenhouse', 'no greenhouse', 'no greenhouse'] * 37)
gh_value = pd.DataFrame([2.5, 2.5, 0, 0] * 37)
gh_name.columns = ['greenhouse_produce']
gh_value.columns = ['greenhouse_value']
# print(gh_name)
# print(gh_value)
greenhouse_data = pd.concat([gh_name, gh_value], axis=1).reindex(gh_name.index)
# print(greenhouse_data)

# generate 'produce' and 'base' columns (duplicate the values)
fruit_veggie = base_data['produce']
base_value = base_data['CO2_base']
produce = [fruit_veggie[i // 4] for i in range(len(fruit_veggie) * 4)]
base_co2 = [base_value[i // 4] for i in range(len(base_value) * 4)]
produce = pd.DataFrame(produce)
base_co2 = pd.DataFrame(base_co2)
produce.columns = ['produce']
base_co2.columns = ['base_co2']
# print(produce)

origin_country_regional = pd.DataFrame(['Germany'] * 148)
origin_country_regional.columns = ['origin_country']

# generate main dataframe and calculate final co2 values
data = pd.concat([produce, base_co2, origin_country_regional, greenhouse_data, organic_data], axis=1)
data['final_co2'] = (data['base_co2'] + data['greenhouse_value']) * (1 - (data['organic_value'] / 100))
# print(data)
#data.to_excel("calc_co2_data_regional.xlsx",
            #  sheet_name='Germany')

## GENERATE DATAFRAME FOR OVERSEAS PRODUCE - AIR FREIGHT VERSION
#
produce_air = [fruit_veggie[i // 474] for i in range(len(fruit_veggie) * 474)]
base_co2_air = [base_value[i // 474] for i in range(len(base_value) * 474)]
produce_air = pd.DataFrame(produce_air)
base_co2_air = pd.DataFrame(base_co2_air)
produce_air.columns = ['produce']
base_co2_air.columns = ['base_co2']
# print(produce_air)

#
org_name_air = pd.DataFrame(['organic', 'not organic'] * 8769)
org_value_air = pd.DataFrame([15, 0] * 8769)
organic_data_air = pd.concat([org_name_air, org_value_air], axis=1).reindex(org_name_air.index)
organic_data_air.columns = ['organic_produce', 'organic_value']
# print(organic_data_air)

# generate origin country and transport value columns -
origin_country_air = [uniqueCountries2[i // 2] for i in range(len(uniqueCountries2) * 2)]
origin_country_air = pd.DataFrame(origin_country_air)
origin_country_air.columns = ['origin_country']
origin_country_air = pd.concat([origin_country_air] * 37, ignore_index=True)
# print(origin_country_air)

#
transport_plane = pd.DataFrame(['airplane'] * 17538)
transport_plane = pd.DataFrame(transport_plane)
transport_plane.columns = ['transport_type']
# print(transport_plane)

# correct
transport_plane_value1 = np.array(airport['plane_co2_per_kg'])
transport_plane_value = [transport_plane_value1[i // 2] for i in range(len(transport_plane_value1) * 2)]
transport_plane_value = pd.DataFrame(transport_plane_value)
transport_plane_value.columns = ['plane_co2_per_kg']
transport_plane_value = pd.concat([transport_plane_value] * 37, ignore_index=True)
# print(transport_plane_value)


# generate main dataframe and calculate final co2 values
data_air = pd.concat([produce_air, base_co2_air, origin_country_air, transport_plane, transport_plane_value,
                      organic_data_air], axis=1)
data_air['final_co2'] = (data_air['base_co2'] + data_air['plane_co2_per_kg']) * (1 - (data_air['organic_value'] / 100))
# print(data_air)
#data_air.to_excel("calc_co2_plane_data_final.xlsx",
                #  sheet_name='plane')

## GENERATE DATAFRAME FOR OVERSEAS PRODUCE - SEA FREIGHT VERSION
#
produce_sea = [fruit_veggie[i // 358] for i in range(len(fruit_veggie) * 358)]
base_co2_sea = [base_value[i // 358] for i in range(len(base_value) * 358)]
produce_sea = pd.DataFrame(produce_sea)
base_co2_sea = pd.DataFrame(base_co2_sea)
produce_sea.columns = ['produce']
base_co2_sea.columns = ['base_co2']
# print(produce_sea)

#
org_name_sea = pd.DataFrame(['organic', 'not organic'] * 6623)
org_value_sea = pd.DataFrame([15, 0] * 6623)
organic_data_sea = pd.concat([org_name_sea, org_value_sea], axis=1).reindex(org_name_sea.index)
organic_data_sea.columns = ['organic_produce', 'organic_value']
# print(organic_data_sea)

# generate origin country and transport value columns
origin_country_sea = [uniqueCountries[i // 2] for i in range(len(uniqueCountries) * 2)]
origin_country_sea = pd.DataFrame(origin_country_sea)
origin_country_sea.columns = ['origin_country']
origin_country_sea = pd.concat([origin_country_sea] * 37, ignore_index=True)
# print(origin_country_sea)

#
transport_ship = pd.DataFrame(['ship'] * 13246)
transport_ship = pd.DataFrame(transport_ship)
transport_ship.columns = ['transport_type']
# print(transport_ship)

#
transport_ship_value1 = np.array(harbor['ship_co2_per_kg'])
transport_ship_value = [transport_ship_value1[i // 2] for i in range(len(transport_ship_value1) * 2)]
transport_ship_value = pd.DataFrame(transport_ship_value)
transport_ship_value.columns = ['ship_co2_per_kg']
transport_ship_value = pd.concat([transport_ship_value] * 37, ignore_index=True)
# print(transport_ship_value)

# generate main dataframe and calculate final co2 values
data_ship = pd.concat([produce_sea, base_co2_sea, origin_country_sea, transport_ship, transport_ship_value,
                       organic_data_sea], axis=1)
data_ship['final_co2'] = (data_ship['base_co2'] + data_ship['ship_co2_per_kg']) * (
            1 - (data_ship['organic_value'] / 100))
# print(data_ship)
data_ship.to_excel("calc_co2_ship_data_final.xlsx",
                   sheet_name='ship')

# merge 3 separate dataframes into 1
data_fin = pd.merge(data_air, data_ship, how='outer')
# print(data_final)
data_final = pd.merge(data_fin, data, how='outer')


# histogram shows distribution of the final co2-values
#print(min(data_final['final_co2']))
#print(max(data_final['final_co2']))
#data_final['final_co2'].hist()
#plt.show()


# compute final column 'CO2 Score' and integrate into final dataframe
conditions = [
    (data_final['final_co2'] < 0.25),
    (data_final['final_co2'] >= 0.25) & (data_final['final_co2'] < 0.5),
    (data_final['final_co2'] >= 0.5) & (data_final['final_co2'] < 1),
    (data_final['final_co2'] >= 1) & (data_final['final_co2'] < 2),
    (data_final['final_co2'] >= 2)
]
values = ['A', 'B', 'C', 'D', 'E']
data_final['co2_score'] = np.select(conditions, values)

data_final = data_final[['produce', 'base_co2', 'origin_country', 'transport_type', 'plane_co2_per_kg',
                         'ship_co2_per_kg', 'greenhouse_produce', 'greenhouse_value', 'organic_produce',
                         'organic_value', 'final_co2','co2_score']]
#print(data_final)
#data_final.to_excel("calc_data_final.xlsx",
                #    sheet_name='all countries')


# check distribution of the CO2-scores
#data_final['co2_score'].hist()
#plt.show()



## START DATA MODELING
from sklearn import neighbors, datasets, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
#%matplotlib inline

# Preprocessing Data for performing Machine learning algorithms
data_final = data_final.fillna(0)
mydf = data_final

mydf_replace = mydf.copy()
labels_produce = mydf_replace['produce'].astype('category').cat.categories.tolist()
replace_produce = {'produce' : {k: v for k,v in zip(labels_produce,list(range(1,len(labels_produce)+1)))}}
labels_country = mydf_replace['origin_country'].astype('category').cat.categories.tolist()
replace_country = {'origin_country' : {k: v for k,v in zip(labels_country,list(range(1,len(labels_country)+1)))}}
labels_transport = mydf_replace['transport_type'].astype('category').cat.categories.tolist()
replace_transport = {'transport_type' : {k: v for k,v in zip(labels_transport,list(range(1,len(labels_transport)+1)))}}
labels_greenhouse = mydf_replace['greenhouse_produce'].astype('category').cat.categories.tolist()
replace_greenhouse = {'greenhouse_produce' : {k: v for k,v in zip(labels_greenhouse,list(range(1,len(labels_greenhouse)+1)))}}
labels_organic = mydf_replace['organic_produce'].astype('category').cat.categories.tolist()
replace_organic = {'organic_produce' : {k: v for k,v in zip(labels_organic,list(range(1,len(labels_organic)+1)))}}
labels_co2score = mydf_replace['co2_score'].astype('category').cat.categories.tolist()
replace_co2score = {'co2_score' : {k: v for k,v in zip(labels_co2score,list(range(1,len(labels_co2score)+1)))}}

mydf_replace.replace(replace_produce, inplace=True)
mydf_replace.replace(replace_country, inplace=True)
mydf_replace.replace(replace_transport, inplace=True)
mydf_replace.replace(replace_greenhouse, inplace=True)
mydf_replace.replace(replace_organic, inplace=True)
mydf_replace.replace(replace_co2score, inplace=True)


# original
y = mydf_replace.co2_score
#mydf_features = ['produce', 'origin_country','transport_type','greenhouse_produce','organic_produce']
#X = mydf_replace[mydf_features]

# adjusted
mydf2 = ['produce', 'origin_country','transport_type', 'base_co2',
                 'greenhouse_produce','organic_produce']
X = mydf_replace[mydf2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Decision Tree Regressor model
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train, y_train)
prediction_dtr = dtr.predict(X_test)
#print(classification_report(y_test, prediction_dtr))
# accuracy of 57%
# with 'base_co2' --> accuracy: 62%
# with all transport co2 values --> accuracy: 99%

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
prediction_rfc = rfc.predict(X_test)
# print(classification_report(y_test, prediction_rfc))
# accuracy of 54%
# with 'base_co2' --> accuracy: 55%

# Support Vector Classifier
svc = SVC()
svc.fit(X_train, y_train)
prediction_svc = svc.predict(X_test)
#print(classification_report(y_test, prediction_svc))
# accuracy of 53%
# with 'base_co2' --> accuracy: 64%

# Stochastic Gradient Decent Classifier
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
prediction_sgd = sgd.predict(X_test)
#print(classification_report(y_test, prediction_sgd))
# accuracy of 52%
# with 'base_co2' --> accuracy: 58%

# Linear Regression - 'parameters can't handle mix of multiclass and continuous targets'
from sklearn.linear_model import LinearRegression
#lr = LinearRegression(normalize=True)
#lr.fit(X, y)
#prediction_lr = lr.predict(X_test)
#print(classification_report(y_test, prediction_lr))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
prediction_gnb = gnb.predict(X_test)
#print(classification_report(y_test, prediction_gnb))
# accuracy of 50%
# with 'base_co2' --> accuracy: 60%

# KNN - doesn't work!!
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
prediction_knn = knn.predict_proba(X_test)
#print(classification_report(y_test, prediction_knn))
# accuracy of ...


## fine-tuning the model
# grid search
from sklearn.model_selection import GridSearchCV
#params = {"n_neighbors": np.arange(1,5), "metric": ["euclidean", "cityblock"]}
#grid = GridSearchCV(estimator=knn, param_grid=params)
#grid.fit(X_train, y_train)
#print(grid.best_score_)
#print(grid.best_estimator_.n_neighbors)

# randomized parameter optimization
from sklearn.model_selection import RandomizedSearchCV
params = {"n_neighbors": range(1,5), "weights": ["uniform", "distance"]}
rsearch = RandomizedSearchCV(estimator=knn,
                             param_distributions=params,
                             cv=4,
                             n_iter=8,
                             random_state=5)
rsearch.fit(X_train, y_train)
#print(rsearch.best_score_)

