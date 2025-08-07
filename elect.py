import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

#####------------INPUT---------------######

date = st.sidebar.date_input("choose the date")
time = st.sidebar.time_input("Input th time")
temp = st.sidebar.number_input("Input the average temperatue in Kelvin",min_value=262, max_value=315, value=295)
pressure = st.sidebar.number_input("Input the pressure in hPa", min_value=0, max_value= 1008371)
humidiity = st.sidebar.number_input("Input the relative humidity percentage", min_value=0, max_value= 100)
wind_speed = st.sidebar.number_input("Input the windspeed in meter per seconds", min_value=0, max_value=133)
wind_deg = st.sidebar.number_input("Input the wind direction in degree", min_value=0, max_value=360)
rain_1h = st.sidebar.number_input("Input the rain in the last  hour in mm", min_value=0, max_value=12)
rain_3h = st.sidebar.number_input("Input rain amount in the last 3 hours in mm", min_value=0.000, max_value=2.315)
snow_3h = st.sidebar.number_input("Input the amount of snow in the last 3 hours in mm", min_value=0.0, max_value=21.5)
clouds_all = st.sidebar.number_input("Input the percentage of cloud cover", min_value=0, max_value=100)
weather_main = st.sidebar.selectbox("Choose what best describes the weather",['clear','clouds','rain','snow','fog','mist',
                                                                            'haze','dust','drizzle','thunderstorm',
                                                                            'smoke','squall'])
submit = st.sidebar.checkbox("Submit?",[True,False])

if submit :

    date_time = datetime.combine(date, time)

    data = {'dt_iso':[date_time], 'temp':[temp],
        'pressure':[pressure], 'humidity':[humidiity], 'wind_speed':[wind_speed], 'wind_deg':[wind_deg], 
        'rain_1h':[rain_1h],'rain_3h':[rain_3h],'snow_3h':[snow_3h], 'clouds_all':[clouds_all], 
        'weather_main':[weather_main]}

    df = pd.DataFrame(data)

    df['dt_iso'] = pd.to_datetime(df['dt_iso'],utc=True)


    df['year'] = df['dt_iso'].dt.year
    df['month'] = df['dt_iso'].dt.month
    df['day_of_week'] = df['dt_iso'].dt.day_name()
    df['day_of_month'] = df['dt_iso'].dt.day
    df['hour'] = df['dt_iso'].dt.hour

    df['diff_y'] = 2025 - df['year']

    dt = [ 'month','day_of_month', 'hour']
    for i in dt :
        df[i] = df[i].astype("object")

    odf = df.copy()

    df.drop(["dt_iso","year"],axis=1, inplace=True)

    cats = ['weather_main_clear',
        'weather_main_clouds', 'weather_main_drizzle', 'weather_main_dust',
        'weather_main_fog', 'weather_main_haze', 'weather_main_mist',
        'weather_main_rain', 'weather_main_smoke', 'weather_main_snow',
        'weather_main_squall', 'weather_main_thunderstorm', 'month_1',
        'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
        'month_8', 'month_9', 'month_10', 'month_11', 'month_12',
        'day_of_week_Friday', 'day_of_week_Monday', 'day_of_week_Saturday',
        'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday',
        'day_of_week_Wednesday', 'day_of_month_1', 'day_of_month_2',
        'day_of_month_3', 'day_of_month_4', 'day_of_month_5', 'day_of_month_6',
        'day_of_month_7', 'day_of_month_8', 'day_of_month_9', 'day_of_month_10',
        'day_of_month_11', 'day_of_month_12', 'day_of_month_13',
        'day_of_month_14', 'day_of_month_15', 'day_of_month_16',
        'day_of_month_17', 'day_of_month_18', 'day_of_month_19',
        'day_of_month_20', 'day_of_month_21', 'day_of_month_22',
        'day_of_month_23', 'day_of_month_24', 'day_of_month_25',
        'day_of_month_26', 'day_of_month_27', 'day_of_month_28',
        'day_of_month_29', 'day_of_month_30', 'day_of_month_31', 'hour_0',
        'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7',
        'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13',
        'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19',
        'hour_20', 'hour_21', 'hour_22', 'hour_23']
    d = {i:[] for i in cats}
    cat_cols = ['weather_main', 'month','day_of_week', 'day_of_month', 'hour']
    df_cats = pd.get_dummies(df[cat_cols])

    for i in d.keys() :
        if i in df_cats.columns :
            d[i].append(True) 
        else :
            d[i].append(False)

    d = pd.DataFrame(d)

    df.drop(cat_cols,axis=1, inplace=True)

    df = df.join(d)
    st.dataframe(df)
    model = joblib.load('catboost.joblib')

    pred = model.predict(df)

    st.success("Prediction")
    st.write(f"The electricity used in date {date} and hour {odf['hour'][0]} is {pred[0]}")
