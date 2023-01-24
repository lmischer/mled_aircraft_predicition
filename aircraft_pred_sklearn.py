import pandas as pd
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


if __name__ == '__main__':
    train_df = pd.read_csv('training_6_category_1/preprocessed.csv')
    train_df['type_sensor_1'] = train_df['type_sensor_1'].astype('category')
    train_df['type_sensor_2'] = train_df['type_sensor_2'].astype('category')
    train_df['type_sensor_3'] = train_df['type_sensor_3'].astype('category')

    X = train_df.drop(columns=['latitude_aircraft', 'longitude_aircraft', 'geoAltitude', 'baroAltitude'])
    Y = train_df[['latitude_aircraft', 'longitude_aircraft', 'geoAltitude', 'baroAltitude']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Simple Linear Regression Prediction
    lr = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train[['longitude_aircraft',
                                                                                'latitude_aircraft']])
    pred = lr.predict(X_test)

    print('Linear Regression')
    print(f"r2 long: {r2_score(y_test['longitude_aircraft'], [long for long, lat in pred])}")
    print(f"r2 lat: {r2_score(y_test['latitude_aircraft'], [lat for long, lat in pred])}")
    print(f"mse long: {mean_squared_error(y_test['longitude_aircraft'], [long for long, lat in pred])}")
    print(f"mse lat: {mean_squared_error(y_test['latitude_aircraft'], [lat for long, lat in pred])}\n\n")

    # Multitask Elastic Net Prediction
    men = MultiTaskElasticNet(random_state=42).fit(X_train, y_train[['longitude_aircraft', 'latitude_aircraft']])

    pred = men.predict(X_test)

    print('Multitask Elastic Net')
    print(f"r2 long: {r2_score(y_test['longitude_aircraft'], [long for long, lat in pred])}")
    print(f"r2 lat: {r2_score(y_test['latitude_aircraft'], [lat for long, lat in pred])}")
    print(f"mse long: {mean_squared_error(y_test['longitude_aircraft'], [long for long, lat in pred])}")
    print(f"mse lat: {mean_squared_error(y_test['latitude_aircraft'], [lat for long, lat in pred])}\n\n")

    # Multitask Elastic Net Prediction with altitude
    men = MultiTaskElasticNet(random_state=42).fit(X_train, y_train)

    pred = men.predict(X_test)

    print('Multitask Elastic Net')
    print(f"r2 long: {r2_score(y_test['longitude_aircraft'], [long for lat, long, geo, baro  in pred])}")
    print(f"r2 lat: {r2_score(y_test['latitude_aircraft'], [lat for lat, long, geo, baro in pred])}")
    print(f"mse long: {mean_squared_error(y_test['longitude_aircraft'], [long for lat, long, geo, baro in pred])}")
    print(f"mse lat: {mean_squared_error(y_test['latitude_aircraft'], [lat for lat, long, geo, baro in pred])}\n\n")
