#!/usr/bin/env python
# coding: utf-8

# In[12]:


import requests # this library helps us to fetch data from API
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor # models for classification and regression tasks
from sklearn.metrics import mean_squared_error
from datetime import datetime,timedelta # to handle date and time
import pytz


# In[22]:





# In[13]:


API_KEY = "97808df2cc2698d6089f5312211f052f" #replace  with your actual api key
BASE_URL = "https://api.openweathermap.org/data/2.5/"


# In[14]:


def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric" #construct the API request URL
    response = requests.get(url) # send the ger request to API
    data = response.json()
    return{
        'city' :data['name'],
        'current_temp' : round(data['main']['temp']),
        'feels_like' : round(data['main']['feels_like']),
        'temp_min' : round(data['main']['temp_min']),
        'temp_max' : round(data['main']['temp_max']),
        'humidity' : round(data['main']['humidity']),
        'description' : data['weather'][0]['description'],
        'country': data['sys'].get('country', 'Unknown'),
        'wind_gust_dir' : data['wind']['deg'],
        'pressure' : data['main']['pressure'],
        'Wind_Gust_Speed' : data['wind']['speed']
    }


# In[15]:


filename = 'ML_Lab/weather.csv'
def read_historical_data(filename):
    df=pd.read_csv("weather.csv")
    df = df.dropna() #remove rows with missing 
    df = df.drop_duplicates()
    return df


# In[16]:


def prepare_data(data):
    le = LabelEncoder() #create a LabelEncoder instances
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    #define the feature variable and target variables
    x =data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']] #feature variables
    y = data['RainTomorrow']

    return x,y,le #return  feature variables,target variable and the label encoder
    


# In[17]:


def train_rain_model(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(x_train,y_train)#train the model
    y_pred = model.predict(x_test) #to make predictions on test split
    print("mean_squared_error for rain model")
    print(mean_squared_error(y_test,y_pred))
    return model


# In[18]:


def prepare_regression_data(data,feature):
    x,y =[],[] #initialize list for feature and target values
    for i  in range(len(data)-1):
        x.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    return x,y


# In[19]:


def train_regression_model(x,y):
    model = RandomForestRegressor(n_estimators =100,random_state=42)
    model.fit(x,y)
    return model


# In[20]:

def predict_future(model, current_value):
    predictions = [current_value]
    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:] 


# In[21]:


import streamlit as st
import pandas as pd
import pytz
from datetime import datetime, timedelta
import base64
import os


    
st.markdown("""
    <style>
     
        
        /* Style for the main title */
        .title {
            font-size: 35px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            padding: 10px;
            background-color: rgb(86, 191, 233);
            border-radius: 10px;
        }

        /* Styling for Weather Info Boxes */
        .info-box {
            background-color: rgba(255, 255, 255, 0.7);
            padding: 15px;
            margin: 10px;
            border-radius: 10px;
            border: 2px solid #ff9800;
            color: #333;
        }

        /* Text Styling */
        .info-box h3 {
            color: #ff5722;
            font-size: 20px;
            text-align: center;
        }

        /* Table Styling */
        .stDataFrame {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            border: 2px solid #ff9800;
        }

        /* Input & Button Styling */
        .stTextInput, .stButton button {
            border-radius: 10px;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)


def weather_view():
    
    st.markdown('<div class="title">🌤 Weather Prediction App ⛅</div>', unsafe_allow_html=True)

    city = st.text_input("Enter city Name:")
    if st.button("Get Weather"):
        current_weather = get_current_weather(city)

        # Load historical data
        historical_data = read_historical_data('/home/raju/rainfall_api/weather.csv')
        
        # Prepare and train the rain prediction model
        x, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(x, y)

        # Map wind direction to compass points
        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 348.75, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]
        compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        current_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': current_weather['wind_gust_dir'],
            'WindGustSpeed': current_weather['Wind_Gust_Speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp']
        }

        current_df = pd.DataFrame([current_data])
        
        # Rain prediction
        rain_prediction = rain_model.predict(current_df)[0]

        # Prepare regression model for temperature and humidity
        x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
        temp_model = train_regression_model(x_temp, y_temp)
        hum_model = train_regression_model(x_hum, y_hum)

        # Predict future temperature and humidity
        future_temp = predict_future(temp_model, current_weather['temp_min'])
        future_humidity = predict_future(hum_model, current_weather['humidity'])

        # Prepare time for future prediction
        timezone = pytz.timezone('Asia/Karachi')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        # Display results
        st.subheader("Current Weather Conditions:")
        st.write(f"🌡**Temperature:** {current_weather['current_temp']}°C")
        st.write(f"🥶 **Feels Like:** {current_weather['feels_like']}°C")
        st.write(f"🔽**Min Temperature:** {current_weather['temp_min']}°C")
        st.write(f"🔼 **Max Temperature:** {current_weather['temp_max']}°C")
        st.write(f"💧**Humidity:** {current_weather['humidity']}%")
        st.write(f"🌦**Weather Prediction:** {current_weather['description']}")
        st.write(f"☔**Rain Prediction:** {'Yes' if rain_prediction else 'No'}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box"><h3>Future Temperature Predictions</h3>', unsafe_allow_html=True)
        st.write(f"Length of future_times: {len(future_times)}")
        st.write(f"Length of future_temp: {len(future_temp)}")
        future_data = pd.DataFrame({"Time": future_times, "Predicted Temperature (°C)": future_temp})
        st.dataframe(future_data)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    weather_view()


















# %%
