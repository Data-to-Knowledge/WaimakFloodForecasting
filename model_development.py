"""


"""
import os
import numpy as np
import pandas as pd
import requests
import json
import zstandard as zstd
import pickle
from scipy import log, exp, mean, stats, special
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt
# from sklearn.inspection import permutation_importance
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

model_file = 'waimak_flood_model_v02.skl.pkl'

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
    df2 = df1.resample('H').sum().iloc[1:-1].fillna(0).result
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
        df2 = df1.result.copy()
        site_name = s
        df2.name = site_name + '_0'

        df_list = []
        for d in [0, 48]:
            n1 = df2.shift(d, 'H')
            n1.name = site_name + '_' + str(d)
            df_list.append(n1)
        # f_data = np.log(pd.concat(df_list, axis=1)).dropna()
f_data = pd.concat(df_list, axis=1).dropna()

#####################################################
### Run model training and testing

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

test_features = data1.loc[date_cut_off:]

test_features1 = np.array(test_features.drop(['66401_0', '66401_48'], axis = 1))
test_features2 = np.array(test_features.drop('66401_0', axis = 1))

rf = HistGradientBoostingRegressor(max_iter = 100, random_state = 42)
rf.fit(train_features1, train_labels)
predictions1 = rf.predict(test_features1)
predict1 = pd.Series(predictions1, index=test_features.index, name='100 HistGB Predicted Flow')

# rf = HistGradientBoostingRegressor(max_iter = 200, random_state = 42)
# rf.fit(train_features1, train_labels)
# predictions2 = rf.predict(test_features1)
# predict2 = pd.Series(predictions2, index=test_features.index, name='200 HistGB Predicted Flow')

combo1 = pd.merge(actual1.reset_index(), predict1.reset_index(), how='left').set_index('from_date')
# combo1 = pd.merge(actual1.reset_index(), predict1.reset_index(), how='left')
# combo2 = pd.merge(combo1, predict2.reset_index(), how='left').set_index('from_date')

### Process results
max_index = argrelextrema(test_labels, np.greater, order=12)[0]

upper_index = np.where(test_labels > np.percentile(test_labels, 80))[0]

test_labels_index = max_index[np.in1d(max_index, upper_index)]

max_data = combo1.iloc[test_labels_index]

## Estimate accuracy/errors
p1 = max_data.iloc[:, 1]
a1 = max_data.iloc[:, 0]

errors = abs(p1 - a1)
bias_errors = (p1 - a1)

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

# Get numerical feature importances -- Must be run without the Hist
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features1.columns, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


## Plotting

ax = combo1.plot()
max_data.reset_index().plot.scatter('from_date', 'Actual Flow', ax=ax)
plt.show()

##################################
### Save the model


base_dir = os.path.realpath(os.path.dirname(__file__))

# pkl1 = pickle.dumps(rf)

with open(os.path.join(base_dir, model_file), 'wb') as f:
    pickle.dump(rf, f)


# with open(os.path.join(base_dir, model_file), 'rb') as f:
#     rff = pickle.load(f)





















