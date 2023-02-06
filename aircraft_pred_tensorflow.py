import pandas as pd
from tensorflow.python.data import Dataset
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras import Model
from tensorflow.python.keras.metrics import Accuracy, MeanSquaredError
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    train_df = pd.read_csv('training_6_category_1/preprocessed.csv')
    train_df['type_sensor_1'] = train_df['type_sensor_1'].astype('category')
    train_df['type_sensor_2'] = train_df['type_sensor_2'].astype('category')
    train_df['type_sensor_3'] = train_df['type_sensor_3'].astype('category')

    X = train_df.drop(columns=['latitude_aircraft', 'longitude_aircraft', 'geoAltitude', 'baroAltitude'])
    Y = train_df[['latitude_aircraft', 'longitude_aircraft', 'geoAltitude', 'baroAltitude']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    dataset = Dataset.from_tensor_slices((dict(X_train), dict(y_train)))

    input_tensor = Input(shape=(len(X.columns),))
    drop_out_tensor = Dropout(.2)(input_tensor)
    hidden_tensor = Dense(10)(drop_out_tensor)
    output_tensor = Dense(len(Y.columns))(hidden_tensor)
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[Accuracy(), MeanSquaredError()])
    print(model.summary())
    # model.fit(X_train, y_train, epochs=10)

    # model.evaluate(X_test, y_test)
