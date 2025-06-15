import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def remove_outliers_turkey(X, y):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = (X >= lower_bound) & (X <= upper_bound)
    non_outlier_mask = np.all(mask, axis=1)
    return X[non_outlier_mask], y[non_outlier_mask]

def main():
    datasets = {
        'iris': ('data/iris.data', None),
        'wine': ('data/wine.data', None),
        'seeds': ('data/seeds_dataset.txt', r'\s+')
    }

    # Pre-processing IRIS
    iris_data = pd.read_csv(datasets['iris'][0], header=datasets['iris'][1])
    X_iris = iris_data.iloc[:, :-1].values
    y_iris_categorical = iris_data.iloc[:, -1].values
    label_encoder_iris = LabelEncoder()
    y_iris = label_encoder_iris.fit_transform(y_iris_categorical)

    X_iris_no_outliers, y_iris_no_outliers = remove_outliers_turkey(X_iris, y_iris)
    scaler_iris = MinMaxScaler()
    X_iris_scaled = scaler_iris.fit_transform(X_iris_no_outliers)

    print("Etykiety (y_iris):")
    print(np.unique(y_iris_categorical))
    print("Rozmiar przed eliminacją wartości skrajnych:", X_iris.shape)
    print("Rozmiar po eliminacji wartości skrajnych:", X_iris_no_outliers.shape)



    # Pre-processing WINE
    wine_data = pd.read_csv(datasets['wine'][0], header=datasets['wine'][1])
    X_wine = wine_data.iloc[:, 1:].values
    y_wine = wine_data.iloc[:, 0].values

    X_wine_no_outliers, y_wine_no_outliers = remove_outliers_turkey(X_wine, y_wine)
    scaler_wine = MinMaxScaler()
    X_wine_scaled = scaler_wine.fit_transform(X_wine_no_outliers)

    print("Etykiety (y_wine):")
    print(np.unique(y_wine))
    print("Rozmiar przed eliminacją wartości skrajnych:", X_wine.shape)
    print("Rozmiar po eliminacji wartości skrajnych:", X_wine_no_outliers.shape)

    # Pre-processing SEEDS
    seeds_data = pd.read_csv(datasets['seeds'][0], sep=datasets['seeds'][1], header=None, engine='python')
    seeds_data.dropna(inplace=True)
    X_seeds = seeds_data.iloc[:, :-1].values
    y_seeds = seeds_data.iloc[:, -1].values.astype(int)

    X_seeds_no_outliers, y_seeds_no_outliers = remove_outliers_turkey(X_seeds, y_seeds)
    scaler_seeds = MinMaxScaler()
    X_seeds_scaled = scaler_seeds.fit_transform(X_seeds_no_outliers)

    print("Etykiety (y_seeds):")
    print(np.unique(y_seeds))
    print("Rozmiar przed eliminacją wartości skrajnych:", X_seeds.shape)
    print("Rozmiar po eliminacji wartości skrajnych:", X_seeds_no_outliers.shape)

if __name__ == '__main__':
    main()