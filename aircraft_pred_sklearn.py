import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, MultiTaskElasticNetCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    train_df = pd.read_csv('training_6_category_1/preprocessed.csv')
    train_df['type_sensor_1'] = train_df['type_sensor_1'].astype('category')
    train_df['type_sensor_2'] = train_df['type_sensor_2'].astype('category')
    train_df['type_sensor_3'] = train_df['type_sensor_3'].astype('category')

    X = train_df.drop(columns=['latitude_aircraft', 'longitude_aircraft', 'geoAltitude', 'baroAltitude'])
    Y = train_df[['latitude_aircraft', 'longitude_aircraft', 'geoAltitude', 'baroAltitude']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Simple Linear Regression Prediction
    print('Multi Output Regression')
    mlr = MultiOutputRegressor(LinearRegression()).fit(
            X_train,
            y_train[['longitude_aircraft', 'latitude_aircraft']]
    )
    with open('models/mlr_model_aircraft_pred_no_alt.pkl', 'wb') as mlr_no_alt_fh:
        pickle.dump(mlr, mlr_no_alt_fh)

    # Multitask Elastic Net Prediction
    print('Multitask Elastic Net')
    men = MultiTaskElasticNet(random_state=42).fit(X_train, y_train[['longitude_aircraft', 'latitude_aircraft']])
    with open('models/men_model_aircraft_pred_no_alt.pkl', 'wb') as men_no_alt_fh:
        pickle.dump(men, men_no_alt_fh)

    # Multitask Elastic Net Prediction with altitude
    print('Multitask Elastic Net with altitude')
    men = MultiTaskElasticNet(random_state=42).fit(X_train, y_train)
    with open('models/men_model_aircraft_pred.pkl', 'wb') as men_fh:
        pickle.dump(men, men_fh)

    # Multitask Elastic Net Prediction l1
    print('Multitask Elastic Net L1')
    men = MultiTaskElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_jobs=-1, random_state=42, verbose=1).fit(
            X_train,
            y_train
    )
    with open('models/men_model_aircraft_pred_l1.pkl', 'wb') as men_fh:
        pickle.dump(men, men_fh)
