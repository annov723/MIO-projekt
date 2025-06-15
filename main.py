import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def remove_outliers_turkey(X, y):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = (X >= lower_bound) & (X <= upper_bound)
    non_outlier_mask = np.all(mask, axis=1)
    return X[non_outlier_mask], y[non_outlier_mask]

def create_iris_fuzzy_system():
    sepal_length = ctrl.Antecedent(np.arange(4.0, 8.5, 0.01), 'sepal_length')
    sepal_width = ctrl.Antecedent(np.arange(2.0, 5.0, 0.01), 'sepal_width')
    petal_length = ctrl.Antecedent(np.arange(1.0, 7.5, 0.01), 'petal_length')
    petal_width = ctrl.Antecedent(np.arange(0.0, 2.6, 0.01), 'petal_width')

    iris_class = ctrl.Consequent(np.arange(0, 3, 1), 'iris_class')

    sepal_length['short'] = fuzz.trapmf(sepal_length.universe, [4.0, 4.0, 5.0, 5.5])
    sepal_length['medium'] = fuzz.trimf(sepal_length.universe, [5.0, 5.8, 6.5])
    sepal_length['long'] = fuzz.trapmf(sepal_length.universe, [6.0, 6.5, 8.0, 8.5])
    sepal_width['narrow'] = fuzz.trapmf(sepal_width.universe, [2.0, 2.0, 2.8, 3.0])
    sepal_width['medium'] = fuzz.trimf(sepal_width.universe, [2.7, 3.2, 3.7])
    sepal_width['wide'] = fuzz.trapmf(sepal_width.universe, [3.5, 3.8, 5.0, 5.0])
    petal_length['short'] = fuzz.trapmf(petal_length.universe, [1.0, 1.0, 1.8, 2.5])
    petal_length['medium'] = fuzz.trimf(petal_length.universe, [2.5, 4.5, 5.0])
    petal_length['long'] = fuzz.trapmf(petal_length.universe, [4.8, 5.5, 7.0, 7.5])
    petal_width['thin'] = fuzz.trapmf(petal_width.universe, [0.0, 0.0, 0.3, 0.6])
    petal_width['medium'] = fuzz.trimf(petal_width.universe, [0.5, 1.2, 1.8])
    petal_width['thick'] = fuzz.trapmf(petal_width.universe, [1.5, 2.0, 2.6, 2.6])
    iris_class['setosa'] = fuzz.trimf(iris_class.universe, [0, 0, 1])
    iris_class['versicolor'] = fuzz.trimf(iris_class.universe, [0, 1, 2])
    iris_class['virginica'] = fuzz.trimf(iris_class.universe, [1, 2, 2])

    rule1 = ctrl.Rule(petal_length['short'] & petal_width['thin'], iris_class['setosa'])
    rule2 = ctrl.Rule(petal_length['medium'] & petal_width['medium'], iris_class['versicolor'])
    rule3 = ctrl.Rule(petal_length['long'] & petal_width['thick'], iris_class['virginica'])

    petal_length.view()
    petal_width.view()
    iris_class.view()

    iris_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(iris_ctrl)


def create_wine_fuzzy_system():
    alcohol = ctrl.Antecedent(np.arange(11, 16, 0.1), 'alcohol')
    malicacid = ctrl.Antecedent(np.arange(0, 6, 0.1), 'malicacid')
    color_intensity = ctrl.Antecedent(np.arange(0, 15, 0.1), 'color_intensity')

    wine_class = ctrl.Consequent(np.arange(1, 4, 1), 'wine_class')

    alcohol['low'] = fuzz.trimf(alcohol.universe, [11, 11, 12.5])
    alcohol['medium'] = fuzz.trimf(alcohol.universe, [12, 13.5, 15])
    alcohol['high'] = fuzz.trimf(alcohol.universe, [13.5, 15, 15])
    malicacid['low'] = fuzz.trimf(malicacid.universe, [0, 0, 2])
    malicacid['medium'] = fuzz.trimf(malicacid.universe, [1, 2.5, 4])
    malicacid['high'] = fuzz.trimf(malicacid.universe, [3, 5, 5])
    color_intensity['low'] = fuzz.trimf(color_intensity.universe, [0, 0, 5])
    color_intensity['medium'] = fuzz.trimf(color_intensity.universe, [4, 7, 10])
    color_intensity['high'] = fuzz.trimf(color_intensity.universe, [8, 13, 15])
    wine_class['class_1'] = fuzz.trimf(wine_class.universe, [1, 1, 1])
    wine_class['class_2'] = fuzz.trimf(wine_class.universe, [2, 2, 2])
    wine_class['class_3'] = fuzz.trimf(wine_class.universe, [3, 3, 3])

    rule1 = ctrl.Rule(alcohol['high'] & malicacid['low'] & color_intensity['high'], wine_class['class_1'])
    rule2 = ctrl.Rule(alcohol['medium'] & malicacid['medium'] & color_intensity['medium'], wine_class['class_2'])
    rule3 = ctrl.Rule(alcohol['low'] & malicacid['high'] & color_intensity['low'], wine_class['class_3'])

    alcohol.view()
    malicacid.view()
    color_intensity.view()
    wine_class.view()

    wine_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(wine_ctrl)



def create_seeds_fuzzy_system():
    area = ctrl.Antecedent(np.arange(10, 25, 0.1), 'area')
    perimeter = ctrl.Antecedent(np.arange(12, 18, 0.1), 'perimeter')
    compactness = ctrl.Antecedent(np.arange(0.80, 0.92, 0.001), 'compactness')
    length = ctrl.Antecedent(np.arange(4.5, 7, 0.1), 'length')
    width = ctrl.Antecedent(np.arange(2.5, 4.5, 0.1), 'width')
    asymmetry = ctrl.Antecedent(np.arange(0, 8, 0.1), 'asymmetry')
    groove = ctrl.Antecedent(np.arange(4, 7, 0.1), 'groove')

    seed_class = ctrl.Consequent(np.arange(1, 4, 1), 'seed_class')

    area['small'] = fuzz.trimf(area.universe, [10, 12, 15])
    area['medium'] = fuzz.trimf(area.universe, [13, 16, 19])
    area['large'] = fuzz.trimf(area.universe, [17, 20, 25])
    perimeter['small'] = fuzz.trimf(perimeter.universe, [12, 13, 14.5])
    perimeter['medium'] = fuzz.trimf(perimeter.universe, [14, 15, 16])
    perimeter['large'] = fuzz.trimf(perimeter.universe, [15.5, 16.5, 18])
    compactness['low'] = fuzz.trimf(compactness.universe, [0.80, 0.83, 0.86])
    compactness['medium'] = fuzz.trimf(compactness.universe, [0.85, 0.87, 0.89])
    compactness['high'] = fuzz.trimf(compactness.universe, [0.88, 0.90, 0.92])
    length['short'] = fuzz.trimf(length.universe, [4.5, 5.0, 5.5])
    length['medium'] = fuzz.trimf(length.universe, [5.3, 5.8, 6.3])
    length['long'] = fuzz.trimf(length.universe, [6.0, 6.5, 7.0])
    width['narrow'] = fuzz.trimf(width.universe, [2.5, 3.0, 3.4])
    width['medium'] = fuzz.trimf(width.universe, [3.2, 3.5, 3.8])
    width['wide'] = fuzz.trimf(width.universe, [3.7, 4.0, 4.5])
    asymmetry['low'] = fuzz.trimf(asymmetry.universe, [0, 1.5, 3])
    asymmetry['medium'] = fuzz.trimf(asymmetry.universe, [2.5, 4, 5.5])
    asymmetry['high'] = fuzz.trimf(asymmetry.universe, [5, 6, 8])
    groove['short'] = fuzz.trimf(groove.universe, [4, 4.5, 5])
    groove['medium'] = fuzz.trimf(groove.universe, [4.9, 5.4, 5.9])
    groove['long'] = fuzz.trimf(groove.universe, [5.7, 6.2, 7])
    seed_class['class1'] = fuzz.trimf(seed_class.universe, [1, 1, 1])
    seed_class['class2'] = fuzz.trimf(seed_class.universe, [2, 2, 2])
    seed_class['class3'] = fuzz.trimf(seed_class.universe, [3, 3, 3])

    rule1 = ctrl.Rule(area['small'] & compactness['high'] & groove['short'], seed_class['class3'])
    rule2 = ctrl.Rule(area['medium'] & compactness['medium'] & groove['medium'], seed_class['class1'])
    rule3 = ctrl.Rule(area['large'] & compactness['medium'] & groove['long'], seed_class['class2'])
    rule4 = ctrl.Rule(width['wide'] & length['long'], seed_class['class2'])
    rule5 = ctrl.Rule(asymmetry['high'], seed_class['class3'])

    area.view()
    compactness.view()
    asymmetry.view()
    seed_class.view()

    seeds_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    return ctrl.ControlSystemSimulation(seeds_ctrl)



def main():
    datasets = {
        'iris': ('data/iris.data', None),
        'wine': ('data/wine.data', None),
        'seeds': ('data/seeds_dataset.txt', r'\s+')
    }

    # Pre-processing IRIS
    iris_col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    iris_data = pd.read_csv(datasets['iris'][0], header=datasets['iris'][1], names=iris_col_names)
    X_iris = iris_data.iloc[:, :-1].values
    y_iris_categorical = iris_data.iloc[:, -1].values
    label_encoder_iris = LabelEncoder()
    y_iris = label_encoder_iris.fit_transform(y_iris_categorical)

    X_iris_no_outliers, y_iris_no_outliers = remove_outliers_turkey(X_iris, y_iris)

    print("Etykiety (y_iris):")
    print(np.unique(y_iris_categorical))
    print("Rozmiar przed eliminacją wartości skrajnych:", X_iris.shape)
    print("Rozmiar po eliminacji wartości skrajnych:", X_iris_no_outliers.shape)

    X_iris = X_iris_no_outliers
    y_iris = y_iris_no_outliers

    # Pre-processing WINE
    wine_col_names = ['class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
                      'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue',
                      'OD280/OD315_of_diluted_wines', 'Proline']
    wine_data = pd.read_csv(datasets['wine'][0], header=datasets['wine'][1], names=wine_col_names)
    X_wine = wine_data.iloc[:, 1:].values
    y_wine = wine_data.iloc[:, 0].values

    X_wine_no_outliers, y_wine_no_outliers = remove_outliers_turkey(X_wine, y_wine)

    print("Etykiety (y_wine):")
    print(np.unique(y_wine))
    print("Rozmiar przed eliminacją wartości skrajnych:", X_wine.shape)
    print("Rozmiar po eliminacji wartości skrajnych:", X_wine_no_outliers.shape)

    X_wine = X_wine_no_outliers
    y_wine = y_wine_no_outliers

    # Pre-processing SEEDS
    seeds_col_names = ['area', 'perimeter', 'compactness', 'length_of_kernel', 'width_of_kernel',
                       'asymmetry_coefficient', 'length_of_kernel_groove', 'class']
    seeds_data = pd.read_csv(datasets['seeds'][0], sep=datasets['seeds'][1], header=None, engine='python', names=seeds_col_names)
    seeds_data.dropna(inplace=True)
    X_seeds = seeds_data.iloc[:, :-1].values
    y_seeds = seeds_data.iloc[:, -1].values.astype(int)

    X_seeds_no_outliers, y_seeds_no_outliers = remove_outliers_turkey(X_seeds, y_seeds)

    print("Etykiety (y_seeds):")
    print(np.unique(y_seeds))
    print("Rozmiar przed eliminacją wartości skrajnych:", X_seeds.shape)
    print("Rozmiar po eliminacji wartości skrajnych:", X_seeds_no_outliers.shape)

    X_seeds = X_seeds_no_outliers
    y_seeds = y_seeds_no_outliers





    # Trenowanie i testowanie IRIS
    iris_fuzzy_system = create_iris_fuzzy_system()





    # Trenowanie i testowanie WINE
    wine_fuzzy_system = create_wine_fuzzy_system()





    # Trenowanie i testowanie SEEDS
    seeds_fuzzy_system = create_seeds_fuzzy_system()





if __name__ == '__main__':
    main()