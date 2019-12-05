########################################################################################
################### PREDICT HOURLY TEMPERATURE & HUMIDITY  #############################
########################################################################################


path = "/Users/haimannmok/Desktop/Lisa/Careers/SkillingUp/MachineLearning/Mastering_Machine_Learning/Projects/04 Deep Learning/LSTM_temperature/"

import sys
sys.path.append(path)
import glob as gb
import pandas as pd
import datetime as dt
from lstm_hour_functions import timestep_matrix, train, train_test, lstm_model_temp


# ==============================  PREPARE WEATHER FILES ==============================

###### Prepare historic weather file for input into LSTM model ######
weather_hrly_file = gb.glob(path + "weather_hourly_20160804_*")[0]               
weather_hrly = pd.read_csv(weather_hrly_file, parse_dates = ['date'], dayfirst = True)

weather_hrly['year'] = weather_hrly['date'].dt.strftime("%Y").astype(int)
weather_hrly['month'] = weather_hrly['date'].dt.strftime("%m").astype(int)
weather_hrly['day_of_month'] = weather_hrly['date'].dt.strftime("%d").astype(int)
weather_hrly['day_of_week'] = weather_hrly['date'].dt.strftime("%w").astype(int)
weather_hrly['hour'] = weather_hrly['date'].dt.strftime("%H").astype(int)
weather_hrly['source'] = 'HKO'
weather_hrly['dummy'] = 'dummy'

temperature_hrly = weather_hrly[['source','temperature', 'date', 'dummy', 'year', 'month', 'day_of_month', 'day_of_week', 'hour']]
humidity_hrly = weather_hrly[['source','humidity', 'date', 'dummy', 'year', 'month', 'day_of_month', 'day_of_week', 'hour']]
past_weather = [temperature_hrly, humidity_hrly]

######  LSTM model paramters ######
timesteps = 24
neurons = 2;
batch = 16;
epochs = 2;
dropout = 0.075;
learning = 0.048;
momentum = 0.98;
decay = learning / epochs
activation = 'softsign';
init = 'glorot_uniform'
test_obs = 1


### For each temperature and humidity
for p in range(len(past_weather)):

    # format past weather data for input into LSTM model
    features, scaler, reframed, output_y, scaled_y = timestep_matrix(timesteps, past_weather[p])
    train_X, train_Y = train(reframed, timesteps, features)

    # train model with past weather data
    model = lstm_model_temp(timesteps, features, neurons, dropout, learning, momentum, decay, init, activation)
    model.fit(train_X, train_Y, epochs=epochs, batch_size=batch, verbose=2)

    # one-step forecast temperature / humidity for the next hour - for each hour
    predictions_all = pd.DataFrame()

    for h in range(0,24):  # <- For each hour, predict for the next hour for today

        hour_weather = past_weather[p][past_weather[p]['hour']==h]   # <- seperate into tables by hour of the day

        features, scaler, reframed, output_y, scaled_y = timestep_matrix(timesteps, hour_weather)
        _, _, predict_X, _ = train_test(reframed, test_obs, timesteps, features)

        predicted = model.predict(predict_X)
        prediction = scaler.inverse_transform(predicted)[-1][0]   
        prediction_time = hour_weather['date'].max() + pd.offsets.DateOffset(hours=24)
        prediction_row = pd.DataFrame({'weather_type': [past_weather[p].columns[1]], 'date': prediction_time, 'prediction':prediction })
        predictions_all = pd.concat([predictions_all, prediction_row])

        update_hour_weather = pd.DataFrame(hour_weather)
        while prediction_time.date() < dt.datetime.today().date():           

            prediction_row['year'] = prediction_row['date'].dt.strftime("%Y").astype(int)
            prediction_row['month'] = prediction_row['date'].dt.strftime("%m").astype(int)
            prediction_row['day_of_month'] = prediction_row['date'].dt.strftime("%d").astype(int)
            prediction_row['day_of_week'] = prediction_row['date'].dt.strftime("%w").astype(int)
            prediction_row['hour'] = prediction_row['date'].dt.strftime("%H").astype(int)
            prediction_row['dummy'] = 'dummy'
            prediction_row['source'] = 'HKO'
            prediction_row = prediction_row[['source', 'prediction', 'date', 'dummy', 'year', 'month', 'day_of_month', 'day_of_week', 'hour']]
            prediction_row = prediction_row.rename(columns={'prediction': past_weather[p].columns[1]})
            update_hour_weather = pd.concat([update_hour_weather,prediction_row])

            features, scaler, reframed, output_y, scaled_y = timestep_matrix(timesteps, update_hour_weather)
            _, _, predict_X, _ = train_test(reframed, test_obs, timesteps, features)

            predicted = model.predict(predict_X)
            prediction = scaler.inverse_transform(predicted)[-1][0]  
            prediction_time = update_hour_weather['date'].max() + pd.offsets.DateOffset(hours=24)
            prediction_row = pd.DataFrame({'weather_type': [past_weather[p].columns[1]], 'date': prediction_time, 'prediction': prediction})
            predictions_all = pd.concat([predictions_all, prediction_row])
            predictions_all = predictions_all.sort_values('date')

    predictions_all.to_csv(path + "predictions_all_" + past_weather[p].columns[1] + ".csv", index = False)

print(predictions_all)


