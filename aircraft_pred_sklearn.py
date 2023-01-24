import pandas as pd
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import r2_score, mean_squared_error


if __name__ == '__main__':
    dtypes = {
            'aircraft': 'int32', 'numMeasurements': 'int16', 'serial_1': 'int32', 'serial_2': 'int32',
            'serial_3': 'int32', 'type_sensor_1': 'category', 'type_sensor_2': 'category', 'type_sensor_3': 'category'
    }
    train_df = pd.read_csv(
            'training_6_category_1/prepped_data.csv', dtype=dtypes,
            parse_dates=['timeAtServer', 'timestamp_1', 'timestamp_2', 'timestamp_3']
        )[0:100]

    train_df.drop(columns=['aircraft', 'serial_1', 'serial_2', 'serial_3'], inplace=True)

    le = LabelEncoder()
    le.fit(train_df['type_sensor_1'].unique())
    train_df['type_sensor_1'] = le.transform(train_df['type_sensor_1'])
    train_df['type_sensor_2'] = le.transform(train_df['type_sensor_2'])
    train_df['type_sensor_3'] = le.transform(train_df['type_sensor_3'])

    # Normalize signal strength for both sensors GRX1090 and Radarcape
    ss_t = train_df[['type_sensor_1', 'signal_strength_1']].rename(columns={'type_sensor_1': 'type',
                                                                            'signal_strength_1': 'signal_strength'})
    ss_t.append(train_df[['type_sensor_2', 'signal_strength_2']].rename(columns={'type_sensor_2': 'type',
                                                                            'signal_strength_2': 'signal_strength'}))
    ss_t.append(train_df[['type_sensor_3', 'signal_strength_3']].rename(columns={'type_sensor_3': 'type',
                                                                            'signal_strength_3': 'signal_strength'}))

    ssn_g = Normalizer()
    ssn_g.fit(ss_t.loc[ss_t['type'] == 0, ['signal_strength']])
    ssn_r = Normalizer()
    ssn_r.fit(ss_t.loc[ss_t['type'] == 1, ['signal_strength']])

    for n in ['1', '2', '3']:
        g_indices = train_df.loc[train_df['type_sensor_' + n] == 0, ['signal_strength_' + n]].index
        r_indices = train_df.loc[train_df['type_sensor_' + n] == 1, ['signal_strength_' + n]].index
        train_df.loc[g_indices, ['signal_strength_' + n]] = ssn_g.transform(train_df.loc[g_indices, ['signal_strength_'
                                                                                        + n]].rename(columns={
                'signal_strength_' + n: 'signal_strength'}))
        train_df.loc[r_indices, ['signal_strength_' + n]] = ssn_r.transform(train_df.loc[r_indices, ['signal_strength_'
                                                                                        + n]].rename(columns={
                'signal_strength_' + n: 'signal_strength'}))

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

    # Multitask Elastic Net Prediction
    men = MultiTaskElasticNet(random_state=42).fit(X_train, y_train)

    pred = men.predict(X_test)

    print('Multitask Elastic Net')
    print(f"r2 long: {r2_score(y_test['longitude_aircraft'], [long for lat, long, geo, baro  in pred])}")
    print(f"r2 lat: {r2_score(y_test['latitude_aircraft'], [lat for lat, long, geo, baro in pred])}")
    print(f"mse long: {mean_squared_error(y_test['longitude_aircraft'], [long for lat, long, geo, baro in pred])}")
    print(f"mse lat: {mean_squared_error(y_test['latitude_aircraft'], [lat for lat, long, geo, baro in pred])}\n\n")
