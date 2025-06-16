import numpy as np
import pandas as pd
from mealpy import FloatVar
from mealpy.swarm_based.GWO import OriginalGWO
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from skfuzzy import control as ctrl
import skfuzzy as fuzz

from helper import remove_outliers_turkey, display_metrics, train_and_evaluate_genfis


def create_iris_fuzzy_system(params=None):
    sepal_length = ctrl.Antecedent(np.arange(4, 8, 0.1), 'sepal_length')
    sepal_width = ctrl.Antecedent(np.arange(2, 5, 0.1), 'sepal_width')
    petal_length = ctrl.Antecedent(np.arange(1, 7, 0.1), 'petal_length')
    petal_width = ctrl.Antecedent(np.arange(0, 3, 0.1), 'petal_width')

    iris_class = ctrl.Consequent(np.arange(0, 3, 1), 'iris_class')

    if params is not None:
        p_sl = np.sort(params[:9])
        p_sw = np.sort(params[9:18])
        p_pl = np.sort(params[18:27])
        p_pw = np.sort(params[27:36])

        sepal_length['short'] = fuzz.trimf(sepal_length.universe, [p_sl[0], p_sl[1], p_sl[2]])
        sepal_length['medium'] = fuzz.trimf(sepal_length.universe, [p_sl[3], p_sl[4], p_sl[5]])
        sepal_length['long'] = fuzz.trimf(sepal_length.universe, [p_sl[6], p_sl[7], p_sl[8]])
        sepal_width['narrow'] = fuzz.trimf(sepal_width.universe, [p_sw[0], p_sw[1], p_sw[2]])
        sepal_width['medium'] = fuzz.trimf(sepal_width.universe, [p_sw[3], p_sw[4], p_sw[5]])
        sepal_width['wide'] = fuzz.trimf(sepal_width.universe, [p_sw[6], p_sw[7], p_sw[8]])
        petal_length['small'] = fuzz.trimf(petal_length.universe, [p_pl[0], p_pl[1], p_pl[2]])
        petal_length['medium'] = fuzz.trimf(petal_length.universe, [p_pl[3], p_pl[4], p_pl[5]])
        petal_length['large'] = fuzz.trimf(petal_length.universe, [p_pl[6], p_pl[7], p_pl[8]])
        petal_width['thin'] = fuzz.trimf(petal_width.universe, [p_pw[0], p_pw[1], p_pw[2]])
        petal_width['medium'] = fuzz.trimf(petal_width.universe, [p_pw[3], p_pw[4], p_pw[5]])
        petal_width['thick'] = fuzz.trimf(petal_width.universe, [p_pw[6], p_pw[7], p_pw[8]])

    else:
        sepal_length['short'] = fuzz.trimf(sepal_length.universe, [4, 4, 5.5])
        sepal_length['medium'] = fuzz.trimf(sepal_length.universe, [5, 5.8, 6.5])
        sepal_length['long'] = fuzz.trimf(sepal_length.universe, [6, 7, 7])
        sepal_width['narrow'] = fuzz.trimf(sepal_width.universe, [2, 2, 3])
        sepal_width['medium'] = fuzz.trimf(sepal_width.universe, [2.5, 3, 3.5])
        sepal_width['wide'] = fuzz.trimf(sepal_width.universe, [3, 4, 4])
        petal_length['small'] = fuzz.trimf(petal_length.universe, [1, 1, 2.5])
        petal_length['medium'] = fuzz.trimf(petal_length.universe, [2, 3.5, 5])
        petal_length['large'] = fuzz.trimf(petal_length.universe, [4.5, 6, 6])
        petal_width['thin'] = fuzz.trimf(petal_width.universe, [0, 0, 1])
        petal_width['medium'] = fuzz.trimf(petal_width.universe, [0.5, 1.2, 1.8])
        petal_width['thick'] = fuzz.trimf(petal_width.universe, [1.5, 2.5, 2.5])

    iris_class.defuzzify_method = 'centroid'
    iris_class['setosa'] = fuzz.trimf(iris_class.universe, [0, 0, 1])
    iris_class['versicolor'] = fuzz.trimf(iris_class.universe, [0, 1, 2])
    iris_class['virginica'] = fuzz.trimf(iris_class.universe, [1, 2, 2])

    rules=[
        ctrl.Rule(sepal_length['short'] & sepal_width['wide'] &
                  petal_length['small'] & petal_width['thin'], iris_class['setosa']),
        ctrl.Rule(petal_length['small'] & petal_width['thin'], iris_class['setosa']),
        ctrl.Rule(sepal_length['medium'] & sepal_width['medium'] &
                  petal_length['medium'] & petal_width['medium'], iris_class['versicolor']),
        ctrl.Rule(petal_length['medium'] & petal_width['medium'], iris_class['versicolor']),
        ctrl.Rule(sepal_length['long'] & sepal_width['narrow'] &
                  petal_length['large'] & petal_width['thick'], iris_class['virginica']),
        ctrl.Rule(petal_length['large'] & petal_width['thick'], iris_class['virginica']),
    ]

    iris_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(iris_ctrl)

def predict_for_iris(fuzzy_system, data):
    DEFAULT_PREDICTION = 1

    predictions = []
    for row in data:
        try:
            fuzzy_system.input['sepal_length'] = row[0]
            fuzzy_system.input['sepal_width'] = row[1]
            fuzzy_system.input['petal_length'] = row[2]
            fuzzy_system.input['petal_width'] = row[3]
            fuzzy_system.compute()
            predicted_value = fuzzy_system.output.get('iris_class')

            if predicted_value is None:
                predictions.append(DEFAULT_PREDICTION)
            else:
                predicted_class = int(round(predicted_value))
                predictions.append(predicted_class)
        except Exception as e:
            print(f'Error: {e}')
            predictions.append(DEFAULT_PREDICTION)
    return np.array(predictions)

def iris_model_cv(X_iris, y_iris, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    original_scores = []
    optimized_scores = []

    for fold, (train_index, test_index) in enumerate(kfold.split(X_iris, y_iris)):
        print(f"\n===== FOLD {fold + 1}/{n_splits} =====\n")
        X_train, X_test = X_iris[train_index], X_iris[test_index]
        y_train, y_test = y_iris[train_index], y_iris[test_index]

        original_fuzzy_system = create_iris_fuzzy_system(params=None)
        y_pred_original = predict_for_iris(original_fuzzy_system, X_test)
        accuracy_original = accuracy_score(y_test, y_pred_original)
        original_scores.append(accuracy_original)
        print(f"FOLD {fold + 1} >> Accuracy ORIGINAL Fuzzy System: {accuracy_original:.4f}")

        def objective_function_iris(solution):
            temp_fuzzy_system = create_iris_fuzzy_system(params=solution)
            y_pred = predict_for_iris(temp_fuzzy_system, X_train)
            return 1.0 - accuracy_score(y_train, y_pred)

        problem_dict = {
            "obj_func": objective_function_iris,
            "bounds": FloatVar(
                lb=[
                    3.5, 4.0, 4.5, 4.5, 5.5, 6.5, 5.5, 6.5, 7.5,
                    1.5, 2.0, 2.5, 2.0, 2.8, 3.5, 2.8, 3.3, 4.2,
                    0.5, 1.0, 2.0, 1.5, 3.0, 4.5, 3.5, 5.0, 7.0,
                    0.0, 0.1, 0.5, 0.3, 1.0, 1.5, 1.0, 1.8, 2.5
                ],
                ub=[
                    4.5, 5.0, 6.0, 6.0, 7.0, 8.0, 7.0, 8.0, 9.0,
                    2.5, 3.0, 3.8, 3.5, 4.0, 4.5, 4.0, 4.5, 5.0,
                    2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0,
                    0.5, 1.0, 1.5, 1.5, 2.0, 2.3, 2.3, 2.7, 3.0
                ]
            ),
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        }
        optimizer = OriginalGWO(epoch=50, pop_size=30)
        result = optimizer.solve(problem_dict)
        best_solution = result.solution

        optimized_fuzzy_system = create_iris_fuzzy_system(params=best_solution)
        y_pred_optimized = predict_for_iris(optimized_fuzzy_system, X_test)
        accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
        optimized_scores.append(accuracy_optimized)
        print(f"FOLD {fold + 1} >> Accuracy GWO-OPTIMIZED Fuzzy System: {accuracy_optimized:.4f}")

    print("\n\n--- CROSS-VALIDATION FINAL RESULTS ---\n")
    print(f"Original model accuracy: {np.mean(original_scores):.4f} +/- {np.std(original_scores):.4f}")
    print(f"GWO-Optimized model accuracy: {np.mean(optimized_scores):.4f} +/- {np.std(optimized_scores):.4f}")

def iris_model():
    # Pre-processing IRIS
    iris_col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    iris_data = pd.read_csv('data/iris.data', header=None, engine='python', names=iris_col_names)
    iris_data.dropna(inplace=True)
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

    # Trenowanie i testowanie IRIS
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
    )

    print("\nPodział danych IRIS:")
    print(f"Training: {X_train.shape[0]}")
    print(f"Test: {X_test.shape[0]}")

    original_fuzzy_system = create_iris_fuzzy_system(params=None)
    y_pred_original = predict_for_iris(original_fuzzy_system, X_test)

    def objective_function_iris(solution):
        temp_fuzzy_system = create_iris_fuzzy_system(params=solution)
        y_pred = predict_for_iris(temp_fuzzy_system, X_train)
        return 1.0 - accuracy_score(y_train, y_pred)

    problem_dict = {
        "obj_func": objective_function_iris,
        "bounds": FloatVar(
            lb=[
                3.5, 4.0, 4.5, 4.5, 5.5, 6.5, 5.5, 6.5, 7.5,
                1.5, 2.0, 2.5, 2.0, 2.8, 3.5, 2.8, 3.3, 4.2,
                0.5, 1.0, 2.0, 1.5, 3.0, 4.5, 3.5, 5.0, 7.0,
                0.0, 0.1, 0.5, 0.3, 1.0, 1.5, 1.0, 1.8, 2.5
            ],
            ub=[
                4.5, 5.0, 6.0, 6.0, 7.0, 8.0, 7.0, 8.0, 9.0,
                2.5, 3.0, 3.8, 3.5, 4.0, 4.5, 4.0, 4.5, 5.0,
                2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0,
                0.5, 1.0, 1.5, 1.5, 2.0, 2.3, 2.3, 2.7, 3.0
            ]
        ),
        "minmax": "min",
        "log_to": "console",
        "save_population": False,
    }
    optimizer = OriginalGWO(epoch=50, pop_size=30)
    result = optimizer.solve(problem_dict)

    best_solution = result.solution
    best_fitness = result.target.fitness

    print(f"\nZakończono GWO dla IRIS.")
    print(f"Best Fitness (Error Rate on Train Set): {best_fitness:.4f}")
    print(f"Najlepsze rozwiązanie: \n{best_solution}\n")

    optimized_fuzzy_system = create_iris_fuzzy_system(params=best_solution)
    y_pred_optimized = predict_for_iris(optimized_fuzzy_system, X_test)

    display_metrics(y_test, y_pred_original, "ORIGINAL IRIS")
    display_metrics(y_test, y_pred_optimized, "GWO-OPTIMIZED IRIS")

    iris_model_cv(X_iris, y_iris)
    train_and_evaluate_genfis(X_iris, y_iris, n_splits=5)



if __name__ == '__main__':
    iris_model()
