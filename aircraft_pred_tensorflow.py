import pandas as pd
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras import Model
from tensorflow.python.keras.metrics import RootMeanSquaredError
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import adam_v2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# BatchNormalization


def get_model(input_shape):
    input_tensor = Input(shape=input_shape)
    drop_out_tensor = Dropout(.2)(input_tensor)
    hidden_tensor = Dense(10)(drop_out_tensor)
    output_tensor = Dense(len(Y.columns))(hidden_tensor)
    model = Model(input_tensor, output_tensor)

    return model


def get_wide_model(input_shape):
    input_tensor = Input(shape=input_shape)
    drop_out_tensor = Dropout(.2)(input_tensor)
    hidden_tensor = Dense(100)(drop_out_tensor)
    output_tensor = Dense(len(Y.columns))(hidden_tensor)
    model = Model(input_tensor, output_tensor)

    return model


def get_deep_model(input_shape):
    input_tensor = Input(shape=input_shape)
    drop_out_tensor = Dropout(.2)(input_tensor)
    hidden_tensor_1 = Dense(20)(drop_out_tensor)
    hidden_tensor_2 = Dense(20)(hidden_tensor_1)
    output_tensor = Dense(len(Y.columns))(hidden_tensor_2)
    model = Model(input_tensor, output_tensor)

    return model


def get_deep_wide_model(input_shape):
    input_tensor = Input(shape=input_shape)
    drop_out_tensor = Dropout(.2)(input_tensor)
    hidden_tensor_1 = Dense(100)(drop_out_tensor)
    hidden_tensor_2 = Dense(100)(hidden_tensor_1)
    output_tensor = Dense(len(Y.columns))(hidden_tensor_2)
    model = Model(input_tensor, output_tensor)

    return model


def get_deep_wider_model(input_shape):
    input_tensor = Input(shape=input_shape)
    drop_out_tensor = Dropout(.2)(input_tensor)
    hidden_tensor_1 = Dense(200)(drop_out_tensor)
    hidden_tensor_2 = Dense(200)(hidden_tensor_1)
    output_tensor = Dense(len(Y.columns))(hidden_tensor_2)
    model = Model(input_tensor, output_tensor)

    return model


if __name__ == '__main__':
    train_df = pd.read_csv('training_6_category_1/preprocessed.csv')
    train_df['type_sensor_1'] = train_df['type_sensor_1'].astype('category')
    train_df['type_sensor_2'] = train_df['type_sensor_2'].astype('category')
    train_df['type_sensor_3'] = train_df['type_sensor_3'].astype('category')

    X = train_df.drop(columns=['latitude_aircraft', 'longitude_aircraft', 'geoAltitude', 'baroAltitude'])
    Y = train_df[['latitude_aircraft', 'longitude_aircraft', 'geoAltitude', 'baroAltitude']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    optimizer = adam_v2.Adam(learning_rate=0.001)
    histories = []

    simple_model = get_model((len(X.columns),))
    simple_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[RootMeanSquaredError()])
    histories.append(simple_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stopping]))
    simple_model.save('models/nn_model_aircraft_pred_simple.h5')

    wide_model = get_wide_model((len(X.columns),))
    wide_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[RootMeanSquaredError()])
    histories.append(wide_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stopping]))
    wide_model.save('models/nn_model_aircraft_pred_wide.h5')

    deep_model = get_deep_model((len(X.columns),))
    deep_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[RootMeanSquaredError()])
    histories.append(deep_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stopping]))
    deep_model.save('models/nn_model_aircraft_pred_deep.h5')

    deep_wide_model = get_deep_wide_model((len(X.columns),))
    deep_wide_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[RootMeanSquaredError()])
    histories.append(deep_wide_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stopping]))
    deep_wide_model.save('models/nn_model_aircraft_pred_deep_wide.h5')

    deep_wider_model = get_deep_wider_model((len(X.columns),))
    deep_wider_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[RootMeanSquaredError()])
    histories.append(deep_wider_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stopping]))
    deep_wider_model.save('models/nn_model_aircraft_pred_deep_wider.h5')

    models = ['simple', 'wide', 'deep', 'deep_wide', 'deep_wider']
    plt.figure()
    for i, mh in enumerate(histories):
        plt.plot(mh.history['val_root_mean_squared_error'])
        pd.DataFrame.from_dict(mh.history).to_csv(''.join(['plots/', models[i], '.csv']), index=False)
    plt.title('model rmse')
    plt.ylabel('val_root_mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['simple', 'wide', 'deep', 'deep_wide', 'deep_wider'], loc='upper left')
    plt.savefig('plots/rmse.svg', format='svg')

    plt.figure()
    for mh in histories:
        plt.plot(mh.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['simple', 'wide', 'deep', 'deep_wide', 'deep_wider'], loc='upper left')
    plt.savefig('plots/mse.svg', format='svg')

