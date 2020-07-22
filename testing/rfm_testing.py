"""


"""
import numpy as np
import pandas as pd
import requests
import json
import zstandard as zstd
from bson import json_util
from scipy import log, exp, mean, stats, special
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
# %matplotlib inline

#####################################
### Parameters

base_url = 'http://tethys-ts.duckdns.org/tethys/data/'

precip_sites = ['217810', '218810', '219510', '219910', '228213', '310510', '311810', '311910', '320010', '321710']
flow_sites = ['66442', '66401']

long_precip = ['219510', '219910', '320010']

n_days_shift = 3

####################################
### Get data

## Datasets
datasets = requests.get(base_url + 'datasets').json()

p_dataset = datasets[0]['dataset_id']
f_dataset = datasets[1]['dataset_id']

## Sites
p_sites1 = requests.post(base_url + 'sampling_sites', params={'dataset_id': p_dataset}).json()
p_sites = [p for p in p_sites1 if p['ref'] in precip_sites]

f_sites1 = requests.post(base_url + 'sampling_sites', params={'dataset_id': f_dataset}).json()
f_sites = [f for f in f_sites1 if f['ref'] in flow_sites]

## TS Data

precip_r_dict = {}
for p in p_sites:
    if p['ref'] in long_precip:
        r = requests.get(base_url + 'time_series_result', params={'dataset_id': p_dataset, 'site_id': p['site_id'], 'compression': 'zstd', 'to_date': '2019-07-1T00:00'})
        dc = zstd.ZstdDecompressor()
        df1 = pd.DataFrame(json.loads(dc.decompress(r.content)))
        df1['from_date'] = pd.to_datetime(df1['from_date']) + pd.DateOffset(hours=12)
        df1.set_index('from_date', inplace=True)
        precip_r_dict.update({p['ref']: df1.copy()})

flow_r_dict = {}
for f in f_sites:
    r = requests.get(base_url + 'time_series_result', params={'dataset_id': f_dataset, 'site_id': f['site_id'], 'compression': 'zstd', 'to_date': '2019-07-1T00:00'})
    dc = zstd.ZstdDecompressor()
    df1 = pd.DataFrame(json.loads(dc.decompress(r.content)))
    df1['from_date'] = pd.to_datetime(df1['from_date']) + pd.DateOffset(hours=12)
    df1.set_index('from_date', inplace=True)
    flow_r_dict.update({f['ref']: df1.copy()})



p_lambda = 0
p_list = []
for s, df1 in precip_r_dict.items():
    # dc = zstd.ZstdDecompressor()
    # df1 = pd.DataFrame(json.loads(dc.decompress(r.content)))
    # df1['from_date'] = pd.to_datetime(df1['from_date'])
    # df1.set_index('from_date', inplace=True)
    # df2 = df1.resample('D').sum().iloc[1:-1].fillna(0).result
    df2 = df1.resample('H').mean().interpolate().resample('D').result.idxmax().iloc[1:-1].dt.hour
    # if p_lambda == 0:
    #     arr, p_lambda = stats.boxcox(df2 + 1)
    # else:
    #     arr = stats.boxcox(df2 + 1, p_lambda)
    # df2 = pd.Series(arr, index=df2.index)
    df2 = np.log(df2 + 1)
    # df3 = df2.result
    # df2['site'] = p['ref']
    site_name = s
    df_list = []
    for d in range(1, n_days_shift+1):
        n1 = df2.shift(d, 'D')
        n1.name = site_name + '_' + str(d)
        df_list.append(n1)
    df4 = pd.concat(df_list, axis=1).dropna()

    p_list.append(df4)

p_data = pd.concat(p_list, axis=1).dropna()

f_lambda = 0
f_list = []
for s, df1 in flow_r_dict.items():
    if s == '66401':
        # dc = zstd.ZstdDecompressor()
        # df1 = pd.DataFrame(json.loads(dc.decompress(r.content)))
        # df1['from_date'] = pd.to_datetime(df1['from_date'])
        # df1.set_index('from_date', inplace=True)
        # df2 = df1.resample('D').max().iloc[1:-1].interpolate().result
        df2 = df1.resample('H').mean().interpolate().resample('D').max().iloc[1:-1].interpolate().result
        cut_off_index = df2 >= df2.quantile(0.95)

        df2 = df1.resample('H').mean().interpolate().resample('D').result.idxmax().iloc[1:-1].dt.hour + 1
        df2 = df2[cut_off_index].copy()
        # cut_off = df2.quantile(0.9)
        # df2 = df2[df2 >= cut_off]

        # if f_lambda == 0:
        #     arr, f_lambda = stats.boxcox(df2)
        # else:
        #     arr = stats.boxcox(df2, f_lambda)
        # df2 = pd.Series(arr, index=df2.index)
        df2 = np.log(df2)
        # df3 = df2.result
        # df2['site'] = p['ref']
        site_name = s
        df2.name = site_name + '_0'

        # df_list = []
        # for d in range(0, n_days_shift+1):
        #     n1 = df2.shift(d, 'D')
        #     n1.name = site_name + '_' + str(d)
        #     df_list.append(n1)
        # df4 = pd.concat(df_list, axis=1).dropna()
        #
        # f_list.append(df4)

# f_data = pd.concat(f_list, axis=1).dropna()
f_data = df2.dropna()

# cut_off = f_data['66401_0'].quantile(0.95)
#
# f_data = f_data[f_data['66401_0'] >= cut_off]

features1 = pd.concat([f_data, p_data], axis=1).dropna()

# features1 = features1[['66401_0', '66401_1', '219510_1', '219910_1', '219510_2', '219910_2', '66401_2']]
# features1 = features1[['66401_0', '219510_1', '219910_1', '219510_2', '219910_2', '320010_1', '219510_3', '219910_3']]
# features1 = features1.drop(['66401_1', '66401_2', '66401_3'], axis=1)

# Labels are the values we want to predict
labels = np.array(features1['66401_0'])

# Remove the labels from the features
# axis 1 refers to the columns
features = features1.drop('66401_0', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 42)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 200, random_state = 42, n_jobs=2)
# rf = ExtraTreesRegressor(n_estimators = 1000, random_state = 42, n_jobs=2)
# rf = GradientBoostingRegressor(loss='ls', n_estimators = 200, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)
# rf.fit(features, labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# predictions = rf.predict(features)

# Calculate the absolute errors
errors = abs(np.exp(predictions) - np.exp(test_labels))
bias_errors = (np.exp(predictions) - np.exp(test_labels))
# errors = abs(predictions - test_labels)
# bias_errors = (predictions - test_labels)
# errors = abs(np.exp(predictions) - np.exp(labels))
# bias_errors = (np.exp(predictions) - np.exp(labels))

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'm3/s.')
print('Mean Error (Bias):', round(np.mean(bias_errors), 2), 'm3/s.')

# Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / test_labels)
mape = 100 * (errors / np.exp(test_labels))
# mape = 100 * (errors / np.exp(labels))
#
# Calculate and display accuracy
accuracy = np.mean(mape)
print('MANE:', round(accuracy, 2), '%.')

# bias1 = np.mean(100 * (bias_errors / test_labels))
bias1 = np.mean(100 * (bias_errors / np.exp(test_labels)))
# bias1 = np.mean(100 * (bias_errors / np.exp(labels)))
print('MNE:', round(bias1, 2), '%.')

bias2 = 100 * np.mean(bias_errors)/np.mean(np.exp(test_labels))
# bias2 = 100 * np.mean(bias_errors)/np.mean(np.exp(labels))
print('NME:', round(bias2, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


act1 = pd.Series(np.exp(test_labels), name='actuals')
# act1 = pd.Series(np.exp(labels), name='actuals')
# act1 = pd.Series(test_labels, name='actuals')
predict1 = pd.Series(np.exp(predictions), name='predictions')
# predict1 = pd.Series(predictions, name='predictions')

# residuals = pd.Series(np.exp(test_labels) - np.exp(predictions), name='residuals').sort_values()
residuals = pd.Series(np.exp(labels) - np.exp(predictions), name='residuals').sort_values()

# act1 = pd.Series(np.exp(test_labels), index=features1.index[len(train_labels):], name='actuals')
# predict1 = pd.Series(np.exp(predictions), index=features1.index[len(train_labels):], name='predictions')
# act1 = pd.Series(test_labels, index=features1.index[len(train_labels):], name='actuals')
# predict1 = pd.Series(predictions, index=features1.index[len(train_labels):], name='predictions')

combo1 = pd.concat([act1, predict1], axis=1).sort_values('predictions').reset_index(drop=True)
print(combo1.describe())

# combo1.index = features1.index
ax = combo1.reset_index().plot.scatter(x=0, y='actuals', legend=True)
combo1.plot(y='predictions', ax=ax, color='red')
plt.show()

# combo2 = combo1[combo1.actuals > 200]

# mane1 = np.mean(np.abs(combo2['actuals'] - combo2['predictions'])/combo2['actuals'])

# mane2 = np.mean(np.abs(combo2['actuals'] - combo2['predictions'])/combo2['actuals'])
# mne2 = np.mean((combo2['actuals'] - combo2['predictions'])/combo2['actuals'])
# print('MANE ' + str(round(mane2, 3)))
# print('MNE ' + str(round(mne2, 3)))







# y_bc1 = stats.boxcox(df2.result)
# y_df_trans = pd.Series(y_bc1[0])
# y_df_trans.index = y_df.index
# y_lambda = y_bc1[1]
# boxcox_y_dict.update({xi: y_lambda})
#
# y_lambda = boxcox_y_dict[best_x]
# predict1 = special.inv_boxcox(predict1, y_lambda)
#
#
#
#
#
# y_bc1 = stats.boxcox(p_data)
























