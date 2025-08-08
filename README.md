# workshop-for-skill4future
about climate change and their effect on the ocean and also an ecosystem



import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_naive_regression(df, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression().fit(X_train, y_train)
    last = X_test.tail(1)
    return [model.predict(last)[0] for _ in range(5)]

# Prepare dummy data
np.random.seed(0)
N = 100
ocean_df = pd.DataFrame({
    'SST_lag1': np.random.uniform(20,30,N),
    'SST_lag2': np.random.uniform(20,30,N),
    'Salinity_lag1': np.random.uniform(30,40,N),
    'Salinity_lag2': np.random.uniform(30,40,N),
})
ocean_df['SST'] = 0.5*ocean_df['SST_lag1'] + 0.3*ocean_df['SST_lag2'] + 0.2*ocean_df['Salinity_lag1'] + np.random.normal(0,1,N)

weather_df = pd.DataFrame({
    'Temp_lag1': np.random.uniform(15,25,N),
    'Temp_lag2': np.random.uniform(15,25,N),
    'Humidity_lag1': np.random.uniform(40,60,N),
    'Humidity_lag2': np.random.uniform(40,60,N),
})
weather_df['Temp'] = 0.6*weather_df['Temp_lag1'] + 0.2*weather_df['Humidity_lag1'] + np.random.normal(0,1,N)

eco_df = pd.DataFrame({
    'Chl_lag1': np.random.uniform(0.5,2.0,N),
    'Chl_lag2': np.random.uniform(0.5,2.0,N),
    'Wave_lag1': np.random.uniform(0.5,1.5,N),
    'Wave_lag2': np.random.uniform(0.5,1.5,N),
})
eco_df['Chl'] = 0.4*eco_df['Chl_lag1'] + 0.3*eco_df['Wave_lag1'] + np.random.normal(0,0.1,N)

# Generate forecasts
ocean_forecast = train_naive_regression(ocean_df,
                                        ['SST_lag1','SST_lag2','Salinity_lag1','Salinity_lag2'],
                                        'SST')
weather_forecast = train_naive_regression(weather_df,
                                          ['Temp_lag1','Temp_lag2','Humidity_lag1','Humidity_lag2'],
                                          'Temp')
ecosystem_forecast = train_naive_regression(eco_df,
                                            ['Chl_lag1','Chl_lag2','Wave_lag1','Wave_lag2'],
                                            'Chl')

# Display results in a table
print("5-Day Forecasts")
print("------------------------------")
print(f"{'Day':<5}{'OCEAN SST (°C)':<20}{'WEATHER Temp (°C)':<20}{'ECOSYSTEM Chlorophyll':<25}")
for i in range(5):
    print(f"{i+1:<5}{ocean_forecast[i]:<20.2f}{weather_forecast[i]:<20.2f}{ecosystem_forecast[i]:<25.2f}")

