import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer, normalize


if __name__ == '__main__':
    normalize_by_sensor_type = False

    dtypes = {
            'type_sensor_1': 'category', 'type_sensor_2': 'category', 'type_sensor_3': 'category'
    }
    train_df = pd.read_csv(
            'training_6_category_1/prepped_data.csv', dtype=dtypes,
    )

    train_df.drop(columns=['aircraft', 'serial_1', 'serial_2', 'serial_3'], inplace=True)

    le = LabelEncoder()
    le.fit(train_df['type_sensor_1'].unique())
    train_df['type_sensor_1'] = le.transform(train_df['type_sensor_1'])
    train_df['type_sensor_2'] = le.transform(train_df['type_sensor_2'])
    train_df['type_sensor_3'] = le.transform(train_df['type_sensor_3'])

    train_df['type_sensor_1'] = train_df['type_sensor_1'].astype('category')
    train_df['type_sensor_2'] = train_df['type_sensor_2'].astype('category')
    train_df['type_sensor_3'] = train_df['type_sensor_3'].astype('category')

    # Normalize: time at server, nuMeasurements

    train_df[['timeAtServer', 'numMeasurements']] = normalize(train_df[['timeAtServer', 'numMeasurements']])

    # Normalize signal strength for both sensors GRX1090 and Radarcape
    ss_data = train_df[['type_sensor_1', 'signal_strength_1']].rename(
            columns={
                    'type_sensor_1': 'type',
                    'signal_strength_1': 'signal_strength'
            }
    )
    ss_data = pd.concat(
            [
                    ss_data, train_df[['type_sensor_2', 'signal_strength_2']].rename(
                    columns={
                            'type_sensor_2': 'type',
                            'signal_strength_2': 'signal_strength'
                    }
            )
            ], ignore_index=True
    )
    ss_data = pd.concat(
            [
                    ss_data, train_df[['type_sensor_3', 'signal_strength_3']].rename(
                    columns={
                            'type_sensor_3': 'type',
                            'signal_strength_3': 'signal_strength'
                    }
            )
            ], ignore_index=True
    )

    if normalize_by_sensor_type:
        ssn_g = Normalizer()
        ssn_g.fit(ss_data.loc[ss_data['type'] == 0, ['signal_strength']])
        ssn_r = Normalizer()
        ssn_r.fit(ss_data.loc[ss_data['type'] == 1, ['signal_strength']])

    # Collect timestamps
    ts_data = train_df['timestamp_1'].rename({'timestamp_1': 'timestamp'})
    ts_data = pd.concat([ts_data, train_df['timestamp_2'].rename({'timestamp_2': 'timestamp'})], ignore_index=True)
    ts_data = pd.concat([ts_data, train_df['timestamp_3'].rename({'timestamp_3': 'timestamp'})], ignore_index=True)

    # Collect sensors latitude
    slat_data = train_df['latitude_sensor_1'].rename({'latitude_sensor_1': 'latitude_sensor'})
    slat_data = pd.concat(
            [
                    slat_data, train_df['latitude_sensor_2'].rename({'latitude_sensor_2': 'latitude_sensor'})
            ], ignore_index=True
    )
    slat_data = pd.concat(
            [
                    slat_data, train_df['latitude_sensor_3'].rename({'latitude_sensor_3': 'latitude_sensor'})
            ], ignore_index=True
    )

    # Collect sensors longitude
    slong_data = train_df['longitude_sensor_1'].rename({'longitude_sensor_1': 'longitude_sensor'})
    slong_data = pd.concat(
            [
                    slong_data, train_df['longitude_sensor_2'].rename({'longitude_sensor_2': 'longitude_sensor'})
            ], ignore_index=True
    )
    slong_data = pd.concat(
            [
                    slong_data, train_df['longitude_sensor_3'].rename({'longitude_sensor_3': 'longitude_sensor'})
            ], ignore_index=True
    )

    # Collect sensors longitude
    sh_data = train_df['height_sensor_1'].rename({'height_sensor_1': 'height_sensor'})
    sh_data = pd.concat(
            [
                    sh_data, train_df['height_sensor_2'].rename({'height_sensor_2': 'height_sensor'})
            ], ignore_index=True
    )
    sh_data = pd.concat(
            [
                    sh_data, train_df['height_sensor_3'].rename({'height_sensor_3': 'height_sensor'})
            ], ignore_index=True
    )

    # Fit normalizer
    if normalize_by_sensor_type:
        fit_df = pd.concat([ts_data, slat_data, slong_data, sh_data], axis=1)
        fit_df.columns = ['timestamp', 'latitude_sensor', 'longitude_sensor', 'height_sensor']
    else:
        fit_df = pd.concat([ss_data['signal_strength'], ts_data, slat_data, slong_data, sh_data], axis=1)
        fit_df.columns = ['signal_strength', 'timestamp', 'latitude_sensor', 'longitude_sensor', 'height_sensor']
    print(fit_df.columns)
    normalizer = Normalizer()
    normalizer.fit(fit_df)

    for n in ['1', '2', '3']:
        if normalize_by_sensor_type:
            g_indices = train_df.loc[train_df['type_sensor_' + n] == 0, ['signal_strength_' + n]].index
            r_indices = train_df.loc[train_df['type_sensor_' + n] == 1, ['signal_strength_' + n]].index
            train_df.loc[g_indices, ['signal_strength_' + n]] = ssn_g.transform(
                    train_df.loc[g_indices, ['signal_strength_' + n]].rename(
                            columns={'signal_strength_' + n: 'signal_strength'}
                    )
            )
            train_df.loc[r_indices, ['signal_strength_' + n]] = ssn_r.transform(
                    train_df.loc[r_indices, ['signal_strength_' + n]].rename(
                            columns={'signal_strength_' + n: 'signal_strength'}
                    )
            )

        if normalize_by_sensor_type:
            train_df[['timestamp_' + n, 'latitude_sensor_' + n, 'longitude_sensor_' + n,
                      'height_sensor_' + n]] = normalizer.transform(
                    train_df[['timestamp_' + n,
                              'latitude_sensor_' +
                              n, 'longitude_sensor_' + n, 'height_sensor_' + n]].rename(
                            columns={
                                    'timestamp_' + n: 'timestamp', 'latitude_sensor_' + n: 'latitude_sensor',
                                    'longitude_sensor_' + n: 'longitude_sensor', 'height_sensor_' + n: 'height_sensor'
                            }
                    )
            )
        else:
            train_df[['signal_strength_' + n, 'timestamp_' + n, 'latitude_sensor_' + n, 'longitude_sensor_' + n,
                      'height_sensor_' + n]] = normalizer.transform(
                    train_df[['signal_strength_' + n, 'timestamp_' + n,
                              'latitude_sensor_' +
                              n, 'longitude_sensor_' + n, 'height_sensor_' + n]].rename(
                            columns={
                                    'signal_strength_' + n: 'signal_strength',
                                    'timestamp_' + n: 'timestamp', 'latitude_sensor_' + n: 'latitude_sensor',
                                    'longitude_sensor_' + n: 'longitude_sensor', 'height_sensor_' + n: 'height_sensor'
                            }
                    )
            )

    if normalize_by_sensor_type:
        train_df.to_csv('training_6_category_1/preprocessed_sensor_by_type.csv', index=False)
    else:
        train_df.to_csv('training_6_category_1/preprocessed.csv', index=False)
