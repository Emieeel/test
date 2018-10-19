import numpy as np 
import pandas as pd
import scipy as sp
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.utils import np_utils
from keras.models import load_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as k
from keras.optimizers import Adam
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

# Create a dictionary to convert the strings to integers
# Don't use this function anymore, rather use update_dict(list)
def create_conv_dict(lijst):
    lijst_dict = {}
    verzameling = set(lijst)
    for i, x in enumerate(verzameling):
        lijst_dict[x] = i
    
    return lijst_dict

# Can update a dictionary or create a new one.
def update_dict(lijst, dictionary = None):
    if dictionary == None:
        return create_conv_dict(lijst)

    set_list = set(lijst)
    set_dict = set( k for k, _ in dictionary.items() )
    for x in (set_list - set_dict):
        dictionary[x] = len(dictionary)
    
    return dictionary

# Converts some column in the dataset when column is specified.
# Changes version V51, now outputs a new dictionary
def convert_column_in_pandas(dataset, column, dictionary):
    # Have to think where to put the update. It is in convert_colums now.
    #dictionary = update_dict(dataset[column], dictionary)
    dataset[column] = dataset[column].apply(lambda x: dictionary[x])
    return dataset

# To swap keys and values in a dictionary
def swap_dict(dictionary):
    swapped_dict = dict((v,k) for k,v in dictionary.items())
    return swapped_dict

# Convert more than one column and make a dictionary to save the
# conversion dictionary.
def convert_columns(dataset, columns, data_dict = None):
    data_dictionary = {}

    if isinstance(columns, str):
        columns = [columns]

    for i in list(columns):
        data = data_dict[i] if isinstance(data_dict, dict) and i in data_dict else None
        dictionary = update_dict(dataset[i], data)
        data_dictionary[i] = dictionary
        dataset = convert_column_in_pandas(dataset, i, dictionary)
    
    if isinstance(data_dict, dict):
        data_dictionary.update(data_dict)

    return dataset, data_dictionary

# Clean the data
def clean_train(data, buckets):
    
    # Get bucket counts from buckets
    dep_time_bucket_count = buckets[0]
    arr_time_bucket_count = buckets[1]
    dist_bucket_count = buckets[2]
    
    # Fill missing age data with random values
    result = data
    result['sched_dep_time'] = result['sched_dep_time'].fillna(result['sched_dep_time'].median())
    result['sched_arr_time'] = result['sched_arr_time'].fillna(result['sched_arr_time'].median())
    result['distance'] = result['distance'].fillna(result['distance'].median())
    
    # Split data into the buckets
    #result['sched_dep_time'] = np.floor((result['sched_dep_time'] * dep_time_bucket_count / 2400))
    #result['sched_arr_time'] = np.floor((result['sched_arr_time'] * arr_time_bucket_count / 2400))
    #result['distance'] = np.floor(((result['distance'] - result['distance'].min()) * dist_bucket_count / (result['distance'].max() - result['distance'].min())))
    
    return result

def f(row):
    if(len(str(row)) > 2):
        output = int(str(row)[:-2])*60 + int(str(row)[-2:])
    else:
        output = row
    return output
    
def time_parser(data):
    data['sched_dep_time'] = data['sched_dep_time'].apply(f)
    data['sched_arr_time'] = data['sched_arr_time'].apply(f)
    return data
    
def density(data):
    for index, row in data.iterrows():
        origin = row['origin']
        dep_time = row['sched_dep_time']
        month = row['month']
        day = row['day']
        temp = data[data['day'] == day]
        temp = temp[temp['month'] == month]
        temp = temp[temp['origin'] == origin]
        dep_dens = len(temp)
        data.iloc[index]['density'] = dep_dens
        print(index)
    return data
    
# Merge the data
def merge_weather(data, weather, buckets):
    
    # Get bucket counts from buckets
    temp_bucket_count = buckets[0]
    dewp_bucket_count = buckets[1]
    humid_bucket_count = buckets[2]
    wind_speed_bucket_count = buckets[3]
    precip_bucket_count = buckets[4]
    pressure_bucket_count = buckets[5]
    visib_bucket_count = buckets[6]
    
    # Merge the two files on time
    data['hour'] = np.floor(data['sched_dep_time'] / 100)
    result = pd.merge(data, weather, on = ['day', 'month', 'hour', 'origin'], how = 'left')
    result = result.drop_duplicates(subset = 'id')
    
    # Fill missing age data with random values
    result['temp'] = result['temp'].fillna(result['temp'].median())
    result['dewp'] = result['dewp'].fillna(result['dewp'].median())
    result['humid'] = result['humid'].fillna(result['humid'].median())
    result['wind_speed'] = result['wind_speed'].fillna(result['wind_speed'].median())
    result['precip'] = result['precip'].fillna(result['precip'].median())
    result['pressure'] = result['pressure'].fillna(result['pressure'].median())
    result['visib'] = result['visib'].fillna(result['visib'].median())
    result['wind_dir'] = result['wind_dir'].fillna(result['wind_dir'].median())
    result['wind_gust'] = result['wind_gust'].fillna(result['wind_gust'].median())

    # Split data into the buckets
    #result['temp'] = np.floor(((result['temp'] - result['temp'].min()) * temp_bucket_count / (result['temp'].max() - result['temp'].min())))
    #result['dewp'] = np.floor(((result['dewp'] - result['dewp'].min()) * dewp_bucket_count / (result['dewp'].max() - result['dewp'].min())))
    #result['humid'] = np.floor(((result['humid'] - result['humid'].min()) * humid_bucket_count / (result['humid'].max() - result['humid'].min())))
    #result['wind_speed'] = np.floor(((result['wind_speed'] - result['wind_speed'].min()) * wind_speed_bucket_count / (result['wind_speed'].max() - result['wind_speed'].min())))
    #result['precip'] = np.floor(((result['precip'] - result['precip'].min()) * precip_bucket_count / (result['precip'].max() - result['precip'].min())))
    #result['pressure'] = np.floor(((result['pressure'] - result['pressure'].min()) * pressure_bucket_count / (result['pressure'].max() - result['pressure'].min())))
    #result['visib'] = np.floor(((result['visib'] - result['visib'].min()) * visib_bucket_count / (result['visib'].max() - result['visib'].min())))
    
    # Always drop these, they are redundant
    #result = result.drop(columns = ['month', 'day', 'hour'])
    
    # Set origin to ints
#    mask = result.origin == 'EWR'
#    result.loc[mask, 'origin'] = 1
#    mask = result.origin == 'JFK'
#    result.loc[mask, 'origin'] = 2
#    mask = result.origin == 'LGA'
#    result.loc[mask, 'origin'] = 3
    
    return result
    
# Merge the data
def merge_airports(data, airports, buckets):
    
    # Get bucket counts from buckets
    lat_bucket_count = buckets[0]
    lon_bucket_count = buckets[1]
    alt_bucket_count = buckets[2]
    
    # Merge the two files on airports
    airports['dest'] = airports['faa']
    airports = airports.drop(columns = ['faa'])
    result = pd.merge(data, airports, on = ['dest'], how = 'left')
    result = result.drop_duplicates(subset = 'id')
    
    # Fill missing age data with random values
    result['lat'] = result['lat'].fillna(result['lat'].median())
    result['lon'] = result['lon'].fillna(result['lon'].median())
    result['alt'] = result['alt'].fillna(result['alt'].median())
    result['tz'] = result['tz'].fillna(result['tz'].median())
    result['dst'] = result['dst'].fillna(0)
    
    # Set tz to ints
    mask = result.dst == 'A'
    result.loc[mask, 'dst'] = 1
    mask = result.dst == 'U'
    result.loc[mask, 'dst'] = 2
    mask = result.dst == 'N'
    result.loc[mask, 'dst'] = 3

    # Split data into the buckets
    #result['lat'] = np.floor(((result['lat'] - result['lat'].min()) * lat_bucket_count / (result['lat'].max() - result['lat'].min())))
    #result['lon'] = np.floor(((result['lon'] - result['lon'].min()) * lon_bucket_count / (result['lon'].max() - result['lon'].min())))
    #result['alt'] = np.floor(((result['alt'] - result['alt'].min()) * alt_bucket_count / (result['alt'].max() - result['alt'].min())))
    
    return result
    
def prepare_test(test):
    
    # Fill empty entries
    test['sched_dep_time'] = test['sched_dep_time'].fillna(test['sched_dep_time'].median())
    test['sched_arr_time'] = test['sched_arr_time'].fillna(test['sched_arr_time'].median())
    test['distance'] = test['distance'].fillna(test['distance'].median())
    
    return test
    
def prepare_weather(test, weather):
    
    # Merge with weather
    test['hour'] = np.floor(test['sched_dep_time'] / 100)
    test = pd.merge(test, weather, on = ['day', 'month', 'hour', 'origin'], how = 'left')
    test = test.drop_duplicates(subset = 'id')
    
    # Fill empty entries
    test['temp'] = test['temp'].fillna(test['temp'].median())
    test['dewp'] = test['dewp'].fillna(test['dewp'].median())
    test['humid'] = test['humid'].fillna(test['humid'].median())
    test['wind_speed'] = test['wind_speed'].fillna(test['wind_speed'].median())
    test['precip'] = test['precip'].fillna(test['precip'].median())
    test['pressure'] = test['pressure'].fillna(test['pressure'].median())
    test['visib'] = test['visib'].fillna(test['visib'].median())
    test['wind_dir'] = test['wind_dir'].fillna(test['wind_dir'].median())
    test['wind_gust'] = test['wind_gust'].fillna(test['wind_gust'].median())
    
    # Set origin to ints
#    mask = test.origin == 'EWR'
#    test.loc[mask, 'origin'] = 1
#    mask = test.origin == 'JFK'
#    test.loc[mask, 'origin'] = 2
#    mask = test.origin == 'LGA'
#    test.loc[mask, 'origin'] = 3
    
    return test
    
def prepare_airports(test, airports):
    
    # Merge with airports
    airports['dest'] = airports['faa']
    airports = airports.drop(columns = ['faa'])
    test = pd.merge(test, airports, on = ['dest'], how = 'left')
    
    # Set tz to ints
    mask = test.dst == 'A'
    test.loc[mask, 'dst'] = 1
    mask = test.dst == 'U'
    test.loc[mask, 'dst'] = 2
    mask = test.dst == 'N'
    test.loc[mask, 'dst'] = 3
    test = test.drop_duplicates(subset = 'id')
    
    # Fill empty entries
    test['lat'] = test['lat'].fillna(test['lat'].median())
    test['lon'] = test['lon'].fillna(test['lon'].median())
    test['alt'] = test['alt'].fillna(test['alt'].median())
    test['tz'] = test['alt'].fillna(test['tz'].median())
    test['dst'] = test['dst'].fillna(0)
    
    return test

# Keras model
def keras_model(cleaned, train_test, test, used_data, save_id):
    model = Sequential()
    
    X_train = cleaned[used_data].as_matrix()
    Y_train = np_utils.to_categorical(cleaned['is_delayed'].as_matrix())
    input_dim = X_train.shape[1]
    
    X_test = np_utils.normalize(train_test[used_data].as_matrix(), axis = 0)
    
    model.add(Dense(2*input_dim, input_dim = input_dim, activation= 'relu'))
    model.add(Dense(input_dim, activation= 'relu'))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model.fit(X_train, Y_train, epochs = 50, batch_size = 200)
    predictions = model.predict_proba(X_test)
    predictions0 = predictions[:,0]
    predictions1 = predictions[:,1]
    print('keras')
    print(roc_auc_score(train_test['is_delayed'], predictions0))
    print(roc_auc_score(train_test['is_delayed'], predictions1))
    
    # Make submission and save to file
    #submission = pd.DataFrame()
    #submission['id'] = save_id
    #submission['is_delayed'] = predictions
    #submission.to_csv('submission_by_distance.csv', index=False)

# Plot data   
def plot_data(cleaned):
    corr = cleaned.corr()
    cmap = sns.diverging_palette(5, 250, as_cmap = True)
    sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, vmin=0, vmax=1)
    #sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, vmin=-1, vmax=0, cmap="BuPu")

# MLP Model    
def MLP_model(cleaned, train_test, test, used_data, save_id):
    numpy_cleaned = cleaned[used_data].values
    min_max_scaler = preprocessing.MinMaxScaler()
    numpy_scaled = min_max_scaler.fit_transform(numpy_cleaned)
    cleaned_train = pd.DataFrame(numpy_scaled)
    
    numpy_cleaned = train_test[used_data].values
    min_max_scaler = preprocessing.MinMaxScaler()
    numpy_scaled = min_max_scaler.fit_transform(numpy_cleaned)
    cleaned_train_test = pd.DataFrame(numpy_scaled)
    
    numpy_cleaned = test[used_data].values
    min_max_scaler = preprocessing.MinMaxScaler()
    numpy_scaled = min_max_scaler.fit_transform(numpy_cleaned)
    cleaned_test = pd.DataFrame(numpy_scaled)
    
    clf = MLPClassifier(solver='sgd', alpha=1e-4, max_iter = 10000)
    clf.fit(cleaned_train, cleaned['is_delayed'])
    predictions = clf.predict_proba(cleaned_train_test)
    predictions = predictions[:,1]
    print(roc_auc_score(train_test['is_delayed'], predictions))
    
    # Make submission and save to file
    #submission = pd.DataFrame()
    #submission['id'] = save_id
    #submission['is_delayed'] = predictions
    #submission.to_csv('submission_by_distance.csv', index=False)
    
def tensorflow_model(cleaned,train_test, test, used_data):
    
    lencleanedhalf = int(len(cleaned)/2)
    training_half = cleaned[:lencleanedhalf]
    eval_half = cleaned[lencleanedhalf:]
    
    numpy_cleaned = training_half[used_data].values
    min_max_scaler = preprocessing.MinMaxScaler()
    numpy_scaled = min_max_scaler.fit_transform(numpy_cleaned)
    training_data = pd.DataFrame(numpy_scaled, columns= used_data)

    training_data['carrier'] = cleaned['carrier']
    training_data['dest'] = cleaned['dest']
    training_data['origin'] = cleaned['origin']
    
    test['is_delayed'] = 0.0
    
    numpy_cleaned = eval_half[used_data].values
    min_max_scaler = preprocessing.MinMaxScaler()
    numpy_scaled = min_max_scaler.fit_transform(numpy_cleaned)
    eval_data = pd.DataFrame(numpy_scaled, columns= used_data)
    

    #eval_data['is_delayed'] = 0.0
    
    eval_data.columns = used_data
    eval_data['carrier'] = cleaned['carrier']
    eval_data['dest'] = cleaned['dest']
    eval_data['origin'] = cleaned['origin']


#    print(eval_data)
#    print(training_data)
    
    
    training_label = training_data.pop('is_delayed')
    eval_label = eval_data.pop('is_delayed')
    
    training_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_data, y=training_label, batch_size=64, shuffle=False, num_epochs=None)

    eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=eval_data, y=eval_label, batch_size = 64, shuffle=False)
    
    destinations = training_data['dest'].values
    carriers = training_data['carrier'].values
    vocab_dest = []
    for word in destinations:
        iswordinlist = False
        for dest in vocab_dest:
            if word == dest:
                iswordinlist = True
        if not iswordinlist:
            vocab_dest.append(word)
    vocab_carr = []
    for word in carriers:
        iswordinlist = False
        for carr in vocab_carr:
            if word == carr:
                iswordinlist = True
        if not iswordinlist:
            vocab_carr.append(word)
    
    
    Month = tf.feature_column.numeric_column('month')
    Day = tf.feature_column.numeric_column('day')
    Dep_time = tf.feature_column.numeric_column('sched_dep_time')
    Arr_time = tf.feature_column.numeric_column('sched_arr_time')
    Carrier = tf.feature_column.categorical_column_with_vocabulary_list('carrier', vocabulary_list = vocab_carr)
    Origin = tf.feature_column.categorical_column_with_vocabulary_list('origin', vocabulary_list = ['EWR', 'JFK', 'LGA'])
    dest = tf.feature_column.categorical_column_with_vocabulary_list('dest', vocabulary_list = vocab_dest)
    Distance = tf.feature_column.numeric_column('distance')
    temp = tf.feature_column.numeric_column('temp')
    dewp = tf.feature_column.numeric_column('dewp')
    humid = tf.feature_column.numeric_column('humid')
    wind_dir = tf.feature_column.numeric_column('wind_dir')
    wind_speed = tf.feature_column.numeric_column('wind_speed')
    wind_gust = tf.feature_column.numeric_column('wind_gust')
    precip = tf.feature_column.numeric_column('precip')
    pressure = tf.feature_column.numeric_column('pressure')
    visib = tf.feature_column.numeric_column('visib')
    lat = tf.feature_column.numeric_column('lat')
    lon = tf.feature_column.numeric_column('lon')
    alt = tf.feature_column.numeric_column('alt')
    
#    linear_features = [Month, Day, Dep_time, Arr_time, Distance,
#        temp, dewp, humid, wind_dir, wind_speed, wind_gust, precip, pressure,
#        visib, lat, lon, alt, Carrier, Origin, dest]
#    regressor = tf.contrib.learn.LinearRegressor(feature_columns=linear_features)
#    regressor.fit(input_fn=training_input_fn, steps=15000)
#    regressor.evaluate(input_fn=eval_input_fn)
#    
#    predictionslin = list(regressor.predict_scores(input_fn=eval_input_fn))
#    print(predictionslin)
#    print(len(predictionslin))
#    print(len(eval_data))
    
    dnn_features = [
        #numerical features
        Month, Day, Dep_time, Arr_time, Distance,
        temp, dewp, humid, wind_dir, wind_speed, wind_gust, precip, pressure,
        visib, lat, lon, alt,
        # densify categorical features:
        tf.feature_column.indicator_column(Carrier),
        tf.feature_column.indicator_column(Origin),
        tf.feature_column.indicator_column(dest)]
    dnnregressor = tf.contrib.learn.DNNRegressor(feature_columns=dnn_features, hidden_units=[100,50,40])
    dnnregressor.fit(input_fn=training_input_fn, steps=5000)
    dnnregressor.evaluate(input_fn=eval_input_fn)
    
    predictions = list(dnnregressor.predict(input_fn=eval_input_fn))
    print(predictions)
    print(roc_auc_score(eval_half['is_delayed'], predictions))


#    with open("predictionsemielscaledlin.txt", "wb") as fp:
#        pickle.dump(predictionslin,fp)
#    
    
def run():
    # Read the data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    weather = pd.read_csv('weather.csv')
    airports = pd.read_csv('airports.csv')

    # Drop useless
    train = train.drop(columns = ['year'])
    weather = weather.drop(columns = ['year', 'time_hour'])
    airports = airports.drop(columns = ['name', 'tzone'])

    # Bucket counts
    dep_time_bucket_count = 24
    arr_time_bucket_count = 24
    dist_bucket_count = 20
    
    train_buckets = np.array([dep_time_bucket_count, arr_time_bucket_count, dist_bucket_count])
    
    temp_bucket_count = 20
    dewp_bucket_count = 20
    humid_bucket_count = 20
    wind_speed_bucket_count = 20
    precip_bucket_count = 20
    pressure_bucket_count = 20 
    visib_bucket_count = 20
    
    weather_buckets = np.array([temp_bucket_count, dewp_bucket_count, humid_bucket_count, 
    wind_speed_bucket_count, precip_bucket_count, pressure_bucket_count, visib_bucket_count])
    
    lat_bucket_count = 20
    lon_bucket_count = 20
    alt_bucket_count = 20
    
    airports_buckets = np.array([lat_bucket_count, lon_bucket_count, alt_bucket_count])
    
    cleaned = clean_train(train, train_buckets)
    #cleaned = density(cleaned)
    cleaned = merge_weather(cleaned, weather, weather_buckets)
    cleaned = merge_airports(cleaned, airports, airports_buckets)
    
    # Make the test data ready
    save_id = test['id']
    test = prepare_test(test)
    test = prepare_weather(test, weather)
    test = prepare_airports(test, airports)
    
    # test on the training set
    train_test = prepare_test(train)
    train_test = prepare_weather(train_test, weather)
    train_test = prepare_airports(train_test, airports)
    
    # Apply dictionaries
    #test, data_dict = convert_columns(test, 'carrier')
    #test, data_dict = convert_columns(test, 'dest')
    #train_test, data_dict = convert_columns(train_test, 'carrier', data_dict)
    #train_test, data_dict = convert_columns(train_test, 'dest', data_dict)
    #cleaned, data_dict = convert_columns(cleaned, 'carrier', data_dict)
    #cleaned, data_dict = convert_columns(cleaned, 'dest', data_dict)
    print(cleaned.columns)
    
    # create_airp_dens(train, 2)
    # parse time to minutes
    #plot_data(cleaned)

    # Do the magic
    #used_data = ['sched_dep_time', 'sched_arr_time', 'distance', 'carrier', 'dest', 'temp', 'dewp', 'humid', 'wind_speed', 'wind_dir', 'precip', 'pressure', 'visib', 'lat', 'lon', 'alt', 'tz', 'dst']
    used_data = ['is_delayed', 'month', 'day', 'sched_dep_time', 'sched_arr_time',
       'distance', 'hour', 'temp', 'dewp',
       'humid', 'wind_dir', 'wind_speed', 'wind_gust', 'precip', 'pressure',
       'visib', 'lat', 'lon', 'alt', 'tz', 'dst']
    #used_data = ['sched_dep_time', 'visib']
    #used_data = [']
    #keras_model(cleaned, train_test, test, used_data, save_id)
    #MLP_model(cleaned, train_test, test, used_data, save_id)
    tensorflow_model(cleaned,train_test, test, used_data)
#    test['is_delayed'] = 0.0
#    numpy_cleaned = test[used_data].values
#    min_max_scaler = preprocessing.MinMaxScaler()
#    numpy_scaled = min_max_scaler.fit_transform(numpy_cleaned)
#    training_data = pd.DataFrame(numpy_scaled)
#    training_data.columns = used_data
#    training_data['carrier'] = cleaned['carrier']
#    training_data['dest'] = cleaned['dest']
#    training_data['origin'] = cleaned['origin']
#    print(training_data.columns)
#    print(training_data[['carrier', 'dest', 'origin']])
#    print(cleaned[['carrier', 'dest', 'origin']])
#    

run()




