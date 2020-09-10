"""


"""
import os
import numpy as np
import pandas as pd
import requests
import json
import zstandard as zstd
import xarray as xr
import pickle
from scipy import log, exp, mean, stats, special
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from scipy.signal import argrelextrema
# %matplotlib inline

#####################################
### Parameters

base_url = 'http://tethys-ts.xyz/tethys/data/'

precip_sites = ['217810', '218810', '219510', '219910', '228213', '310510', '311810', '311910', '320010', '321710']
flow_sites = ['66442', '66401', '66403']

long_precip = ['219510', '219910', '320010']

to_date = '2019-07-1T00:00'

n_hours_shift = 36

ms_base_path = r'N:\met_service\forecasts'

ms_nc_list = ['wrf_hourly_precip_nz8kmN-ECMWF_2020083112.nc', 'wrf_hourly_precip_nz4kmN-NCEP_2020083106.nc', 'wrf_hourly_precip_nz8kmN-UKMO_2020083112.nc', 'wrf_hourly_precip_nz4kmN-NCEP_2020083100.nc', 'wrf_hourly_precip_nz8kmN-ECMWF_2020083100.nc', 'wrf_hourly_precip_nz4kmN-NCEP_2020083018.nc', 'wrf_hourly_precip_nz8kmN-UKMO_2020083100.nc', 'wrf_hourly_precip_nz8kmN-NCEP_2020083100.nc', 'wrf_hourly_precip_nz4kmN-NCEP_2020083012.nc', 'wrf_hourly_precip_nz8kmN-NCEP_2020083018.nc', 'wrf_hourly_precip_nz8kmN-ECMWF_2020083012.nc']

####################################
### Get data

## Datasets
datasets = requests.get(base_url + 'datasets').json()

p_dataset = [d for d in datasets if (d['feature'] == 'atmosphere') and (d['parameter'] == 'precipitation') and (d['processing_code'] == 'quality_controlled_data')][0]['dataset_id']
f_dataset = [d for d in datasets if (d['feature'] == 'waterway') and (d['parameter'] == 'streamflow') and (d['processing_code'] == 'quality_controlled_data')][0]['dataset_id']

## Sites
p_sites1 = requests.post(base_url + 'sampling_sites', params={'dataset_id': p_dataset}).json()
p_sites = [p for p in p_sites1 if p['ref'] in precip_sites]

f_sites1 = requests.post(base_url + 'sampling_sites', params={'dataset_id': f_dataset}).json()
f_sites = [f for f in f_sites1 if f['ref'] in flow_sites]

## TS Data

precip_r_dict = {}
for p in p_sites:
    if p['ref'] in long_precip:
        print(p['ref'])
        r = requests.get(base_url + 'time_series_results', params={'dataset_id': p_dataset, 'site_id': p['site_id'], 'compression': 'zstd', 'to_date': to_date})
        dc = zstd.ZstdDecompressor()
        df1 = pd.DataFrame(json.loads(dc.decompress(r.content)))
        df1['from_date'] = pd.to_datetime(df1['from_date']) + pd.DateOffset(hours=12)
        df1.set_index('from_date', inplace=True)
        precip_r_dict.update({p['ref']: df1.copy()})

flow_r_dict = {}
for f in f_sites:
    print(f['ref'])
    r = requests.get(base_url + 'time_series_results', params={'dataset_id': f_dataset, 'site_id': f['site_id'], 'compression': 'zstd', 'to_date': to_date})
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
    df2 = df1.resample('H').sum().iloc[1:-1].fillna(0).result
    # df2 = df1.resample('H').mean().interpolate().resample('D').result.idxmax().iloc[1:-1].dt.hour
    # if p_lambda == 0:
    #     arr, p_lambda = stats.boxcox(df2 + 1)
    # else:
    #     arr = stats.boxcox(df2 + 1, p_lambda)
    # df2 = pd.Series(arr, index=df2.index)
    # df2 = np.log(df2 + 1)
    # df3 = df2.result
    # df2['site'] = p['ref']
    site_name = s
    df_list = []
    for d in range(14, n_hours_shift+1):
        n1 = df2.shift(d, 'H')
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
        # df2 = df1.resample('H').mean().iloc[1:-1].interpolate().result
        df2 = df1.result.copy()
        # df3 = df1.resample('H').mean().interpolate().rolling(24).mean().result
        # cut_off_index = df2 >= df2.quantile(0.95)

        # df2 = df1.resample('H').mean().interpolate().resample('D').result.idxmax().iloc[1:-1].dt.hour + 1
        # df2 = df2[cut_off_index].copy()
        # cut_off = df2.quantile(0.9)
        # df2 = df2[df2 >= cut_off]

        # if f_lambda == 0:
        #     arr, f_lambda = stats.boxcox(df2)
        # else:
        #     arr = stats.boxcox(df2, f_lambda)
        # df2 = pd.Series(arr, index=df2.index)
        # df3 = df2.result
        # df2['site'] = p['ref']
        site_name = s
        df2.name = site_name + '_0'

        df_list = []
        for d in [0, 48]:
            n1 = df2.shift(d, 'H')
            n1.name = site_name + '_' + str(d)
            df_list.append(n1)
        # f_data = np.log(pd.concat(df_list, axis=1)).dropna()
f_data = pd.concat(df_list, axis=1).dropna()

        # f_list.append(df4)

        # f_data = pd.concat(f_list, axis=1).dropna()
        # f_data = df2.dropna()

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


# train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 42)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = None)

# Instantiate model with 1000 decision trees
# rf = RandomForestRegressor(n_estimators = 200, random_state = 42, n_jobs=2)
rf = RandomForestRegressor(n_estimators = 200, random_state = None, n_jobs=3)

# rf = ExtraTreesRegressor(n_estimators = 1000, random_state = 42, n_jobs=2)
# rf = GradientBoostingRegressor(loss='ls', n_estimators = 1000, random_state = None)

# Train the model on training data
rf.fit(train_features, train_labels)
# rf.fit(features, labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features1)
# predictions = rf.predict(features)

max_index = argrelextrema(test_labels, np.greater_equal, order=12)[0]

upper_index = np.where(test_labels > np.percentile(test_labels, 90))[0]

test_labels_index = max_index[np.in1d(max_index, upper_index)]

test_labels6 = test_labels[test_labels_index]
predictions6 = predictions[test_labels_index]
# Calculate the absolute errors
# errors = abs(np.exp(predictions) - np.exp(test_labels))
# bias_errors = (np.exp(predictions) - np.exp(test_labels))
errors = abs(predictions6 - test_labels6)
bias_errors = (predictions6 - test_labels6)
# errors = abs(np.exp(predictions) - np.exp(labels))
# bias_errors = (np.exp(predictions) - np.exp(labels))

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'm3/s.')
print('Mean Error (Bias):', round(np.mean(bias_errors), 2), 'm3/s.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels6)
# mape = 100 * (errors / np.exp(test_labels))
# mape = 100 * (errors / np.exp(labels))
#
# Calculate and display accuracy
accuracy = np.mean(mape)
print('MANE:', round(accuracy, 2), '%.')

bias1 = np.mean(100 * (bias_errors / test_labels6))
# bias1 = np.mean(100 * (bias_errors / np.exp(test_labels)))
# bias1 = np.mean(100 * (bias_errors / np.exp(labels)))
print('MNE:', round(bias1, 2), '%.')

bias2 = 100 * np.mean(bias_errors)/np.mean(test_labels6)
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
# residuals = pd.Series(np.exp(labels) - np.exp(predictions), name='residuals').sort_values()

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



### Plotting

date_cut_off = '2013-07-01'

actual1 = flow_r_dict['66401'].result.loc[date_cut_off:]
actual1.name = 'Actual Flow'

test_labels = np.array(actual1)

data1 = pd.concat([f_data, p_data], axis=1).dropna()

features = data1.loc[:date_cut_off]
train_labels = np.array(features['66401_0'])
features1 = features.drop(['66401_0', '66401_48'], axis = 1)
train_features1 = np.array(features1)

features2 = features.drop('66401_0', axis = 1)
train_features2 = np.array(features2)

# top_5 = features1[features1['66401_0'] > features1['66401_0'].quantile(0.80)].copy()

# top_5_precip = pd.concat([top_5_flow, p_data], axis=1, join='inner').drop('66401_0', axis = 1).dropna()

# top_5_train_labels = np.array(top_5['66401_0'])
# top_5_train_features = np.array(top_5.drop('66401_0', axis = 1))

# top_5_test_labels = np.array(top_5_flow[:date_cut_off])
# top_5_test_features = np.array(top_5_precip.loc[:date_cut_off])
# top_5_test_features_index = top_5_precip.loc[:date_cut_off].index

test_features = data1.loc[date_cut_off:]

test_features1 = np.array(test_features.drop(['66401_0', '66401_48'], axis = 1))
test_features2 = np.array(test_features.drop('66401_0', axis = 1))

rf = HistGradientBoostingRegressor(max_iter = 100, random_state = 42)
rf.fit(train_features1, train_labels)
predictions1 = rf.predict(test_features1)
predict1 = pd.Series(predictions1, index=test_features.index, name='100 HistGB Predicted Flow')
# median1 = actual1[actual1 > actual1.quantile(0.95)].median()
# predict3 = predict2[predict2 > median1]

# rf = HistGradientBoostingRegressor(max_iter = 200, random_state = 42)
# rf.fit(train_features1, train_labels)
# predictions2 = rf.predict(test_features1)
# predict2 = pd.Series(predictions2, index=test_features.index, name='200 HistGB Predicted Flow')

# rf = RandomForestRegressor(n_estimators = 200, random_state = None, n_jobs=3)
# rf.fit(top_5_train_features, top_5_train_labels)
# predictions1 = rf.predict(test_features)
# predict2 = pd.Series(np.exp(predictions1), index=test_features1.index, name='RF top 5% Predicted Flow')
# predict2 = pd.Series(predictions1, index=test_features1.index, name='RF top 5% Predicted Flow')
# median1 = actual1[actual1 > actual1.quantile(0.95)].median()
# predict4 = predict2[predict2 > median1]

# rf = RandomForestRegressor(n_estimators = 200, random_state = None, n_jobs=3)
# rf.fit(train_features, train_labels)
# predictions2 = rf.predict(test_features)
# predict3 = pd.Series(np.exp(predictions1), index=test_features1.index, name='RF all Predicted Flow')
# predict3 = pd.Series(predictions2, index=test_features1.index, name='RF all Predicted Flow')
# median1 = actual1[actual1 > actual1.quantile(0.95)].median()
# predict4 = predict3[predict3 > median1]

combo1 = pd.merge(actual1.reset_index(), predict1.reset_index(), how='left').set_index('from_date')
# combo1 = pd.merge(actual1.reset_index(), predict1.reset_index(), how='left')
# combo2 = pd.merge(combo1, predict2.reset_index(), how='left').set_index('from_date')

# ax = predict2.plot()
# combo1.plot(y='predictions', ax=ax, color='red')
# predict2.plot()
# plt.show()

max_index = argrelextrema(test_labels, np.greater, order=12)[0]

upper_index = np.where(test_labels > np.percentile(test_labels, 80))[0]

test_labels_index = max_index[np.in1d(max_index, upper_index)]

max_data = combo1.iloc[test_labels_index]


ax = combo1.plot()
max_data.reset_index().plot.scatter('from_date', 'Actual Flow', ax=ax)
plt.show()


p1 = max_data.iloc[:, 1]
a1 = max_data.iloc[:, 0]

errors = abs(p1 - a1)
bias_errors = (p1 - a1)
# errors = abs(np.exp(predictions) - np.exp(labels))
# bias_errors = (np.exp(predictions) - np.exp(labels))

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'm3/s.')
print('Mean Error (Bias):', round(np.mean(bias_errors), 2), 'm3/s.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / a1)
#
# Calculate and display accuracy
accuracy = np.mean(mape)
print('MANE:', round(accuracy, 2), '%.')

bias1 = np.mean(100 * (bias_errors / a1))
print('MNE:', round(bias1, 2), '%.')

bias2 = 100 * np.mean(bias_errors)/np.mean(a1)
print('NME:', round(bias2, 2), '%.')



# r = permutation_importance(rf, test_features1, np.array(actual1), n_repeats=5, random_state=42)


# Get numerical feature importances
importances = list(rf.feature_importances_)
# importances = list(r['importances_mean'])

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features1.columns, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


### MetService forecast tests

ms1 = xr.open_dataset(ms_nc1)
t1 = pd.to_datetime(ms1.time.values) + pd.DateOffset(hours=12)
print(t1[-1] - pd.Timestamp('2020-09-01 09:30'))
ms1.close()

ms2 = xr.open_dataset(ms_nc2)
t2 = pd.to_datetime(ms2.time.values) + pd.DateOffset(hours=12)
print(t2[-1] - pd.Timestamp('2020-09-01 06:30'))
ms2.close()

ms3 = xr.open_dataset(ms_nc3)
t3 = pd.to_datetime(ms3.time.values) + pd.DateOffset(hours=12)
print(t3[-1] - pd.Timestamp('2020-09-01 08:30'))
ms3.close()

ms4 = xr.open_dataset(ms_nc4)
t4 = pd.to_datetime(ms4.time.values) + pd.DateOffset(hours=12)
print(t4[-1] - pd.Timestamp('2020-08-31 22:30'))
ms4.close()



for nc in ms_nc_list:
    print(nc)
    path1 = os.path.join(ms_base_path, nc)
    mod1 = pd.Timestamp(os.path.getmtime(path1), unit='s').round('s') + pd.DateOffset(hours=12)
    print(mod1)
    ms1 = xr.open_dataset(path1)
    t1 = pd.to_datetime(ms1.time.values) + pd.DateOffset(hours=12)
    print(t1[-1] - mod1)
    ms1.close()
























