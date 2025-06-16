import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skfuzzy import control as ctrl
import skfuzzy as fuzz


def remove_outliers_turkey(X, y):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = (X >= lower_bound) & (X <= upper_bound)
    non_outlier_mask = np.all(mask, axis=1)
    return X[non_outlier_mask], y[non_outlier_mask]

def create_iris_fuzzy_system(params=None):
    sepal_length = ctrl.Antecedent(np.arange(4.3, 8.0, 0.1), 'sepal_length')
    sepal_width = ctrl.Antecedent(np.arange(2.0, 4.5, 0.1), 'sepal_width')
    petal_length = ctrl.Antecedent(np.arange(1.0, 7.0, 0.1), 'petal_length')
    petal_width = ctrl.Antecedent(np.arange(0.1, 2.6, 0.1), 'petal_width')

    iris_class = ctrl.Consequent(np.arange(1, 4, 0.1), 'iris_class', defuzzify_method='centroid')

    if params is not None:
        print("not none")
    else:
        sepal_length['low'] = fuzz.trimf(sepal_length.universe, [4.3, 5.1, 5.8])
        sepal_length['medium'] = fuzz.trimf(sepal_length.universe, [5.4, 6.1, 6.9])
        sepal_length['high'] = fuzz.trimf(sepal_length.universe, [6.4, 7.1, 7.9])
        sepal_width['low'] = fuzz.trimf(sepal_width.universe, [2.0, 2.8, 3.4])
        sepal_width['medium'] = fuzz.trimf(sepal_width.universe, [2.9, 3.5, 4.1])
        sepal_width['high'] = fuzz.trimf(sepal_width.universe, [3.6, 4.0, 4.4])
        petal_length['low'] = fuzz.trapmf(petal_length.universe, [1.0, 1.0, 1.5, 1.9])
        petal_length['medium'] = fuzz.trapmf(petal_length.universe, [3.0, 4.0, 4.4, 5.1])
        petal_length['high'] = fuzz.trapmf(petal_length.universe, [4.5, 5.1, 6.9, 6.9])
        petal_width['low'] = fuzz.trapmf(petal_width.universe, [0.1, 0.2, 0.4, 0.6])
        petal_width['medium'] = fuzz.trapmf(petal_width.universe, [1.0, 1.2, 1.5, 1.8])
        petal_width['high'] = fuzz.trapmf(petal_width.universe, [1.5, 1.8, 2.4, 2.5])

    iris_class['setosa'] = fuzz.trimf(iris_class.universe, [0, 0, 1])
    iris_class['versicolor'] = fuzz.trimf(iris_class.universe, [0, 1, 2])
    iris_class['virginica'] = fuzz.trimf(iris_class.universe, [1, 2, 2])

    rule1 = ctrl.Rule(petal_length['low'] | petal_width['low'], species['setosa'])
    rule2 = ctrl.Rule(petal_length['medium'] & petal_width['medium'], species['versicolor'])
    rule3 = ctrl.Rule(petal_length['high'] & petal_width['high'], species['virginica'])

    petal_length.view()
    petal_width.view()
    iris_class.view()

    iris_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(iris_ctrl)

def iris_model():
    # Pre-processing IRIS
    iris_col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    iris_data = pd.read_csv('data/iris.data', header=None, names=iris_col_names)
    X_iris = iris_data.iloc[:, :-1].values
    y_iris_categorical = iris_data.iloc[:, -1].values
    label_encoder_iris = LabelEncoder()
    y_iris = label_encoder_iris.fit_transform(y_iris_categorical)

    X_iris_no_outliers, y_iris_no_outliers = remove_outliers_turkey(X_iris, y_iris)

    print("Etykiety (y_iris):")
    print(np.unique(y_iris_categorical))
    print("Rozmiar przed eliminacją wartości skrajnych:", X_iris.shape)
    print("Rozmiar po eliminacji wartości skrajnych:", X_iris_no_outliers.shape)

    # Trenowanie i testowanie IRIS
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
    )

    print("\n\n\nPodział danych IRIS:")
    print(f"Training: {X_train.shape[0]}")
    print(f"Test: {X_test.shape[0]}")

    original_fuzzy_system = create_iris_fuzzy_system(params=None)
    y_pred_original = predict_for_iris(original_fuzzy_system, X_test)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    print(f"Accuracy ORIGINAL Fuzzy System: {accuracy_original:.4f}")



if __name__ == '__main__':
    iris_model()
