import numpy as np
import pandas as pd
from mealpy import FloatVar
from mealpy.swarm_based.GWO import OriginalGWO
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.model_selection import train_test_split
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

def create_iris_fuzzy_system(params=None):
    sepal_length = ctrl.Antecedent(np.arange(4.0, 8.5, 0.01), 'sepal_length')
    sepal_width = ctrl.Antecedent(np.arange(2.0, 5.0, 0.01), 'sepal_width')
    petal_length = ctrl.Antecedent(np.arange(1.0, 7.5, 0.01), 'petal_length')
    petal_width = ctrl.Antecedent(np.arange(0.0, 2.6, 0.01), 'petal_width')

    iris_class = ctrl.Consequent(np.arange(0, 3, 1), 'iris_class')

    if params is not None:
        p_sl = np.sort(params[:11])
        p_sw = np.sort(params[11:22])
        p_pl = np.sort(params[22:33])
        p_pw = np.sort(params[33:44])

        sepal_length['short'] = fuzz.trapmf(sepal_length.universe, [p_sl[0], p_sl[1], p_sl[2], p_sl[3]])
        sepal_length['medium'] = fuzz.trimf(sepal_length.universe, [p_sl[4], p_sl[5], p_sl[6]])
        sepal_length['long'] = fuzz.trapmf(sepal_length.universe, [p_sl[7], p_sl[8], p_sl[9], p_sl[10]])

        sepal_width['narrow'] = fuzz.trapmf(sepal_width.universe, [p_sw[0], p_sw[1], p_sw[2], p_sw[3]])
        sepal_width['medium'] = fuzz.trimf(sepal_width.universe, [p_sw[4], p_sw[5], p_sw[6]])
        sepal_width['wide'] = fuzz.trapmf(sepal_width.universe, [p_sw[7], p_sw[8], p_sw[9], p_sw[10]])

        petal_length['short'] = fuzz.trapmf(petal_length.universe, [p_pl[0], p_pl[1], p_pl[2], p_pl[3]])
        petal_length['medium'] = fuzz.trimf(petal_length.universe, [p_pl[4], p_pl[5], p_pl[6]])
        petal_length['long'] = fuzz.trapmf(petal_length.universe, [p_pl[7], p_pl[8], p_pl[9], p_pl[10]])

        petal_width['thin'] = fuzz.trapmf(petal_width.universe, [p_pw[0], p_pw[1], p_pw[2], p_pw[3]])
        petal_width['medium'] = fuzz.trimf(petal_width.universe, [p_pw[4], p_pw[5], p_pw[6]])
        petal_width['thick'] = fuzz.trapmf(petal_width.universe, [p_pw[7], p_pw[8], p_pw[9], p_pw[10]])
    else:
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

    rule1 = ctrl.Rule(petal_length['short'] | petal_width['thin'], iris_class['setosa'])
    rule2 = ctrl.Rule(petal_length['medium'] & petal_width['medium'], iris_class['versicolor'])
    rule3 = ctrl.Rule(petal_length['long'] & petal_width['thick'], iris_class['virginica'])

    iris_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(iris_ctrl)

def predict_for_iris(fuzzy_system, X_data):
    predictions = []
    for sample in X_data:
        try:
            fuzzy_system.input['sepal_length'] = sample[0]
            fuzzy_system.input['sepal_width'] = sample[1]
            fuzzy_system.input['petal_length'] = sample[2]
            fuzzy_system.input['petal_width'] = sample[3]
            fuzzy_system.compute()
            predicted_class = np.round(fuzzy_system.output['iris_class'])
            predictions.append(predicted_class)
        except Exception as e:
            predictions.append(1)
    return np.array(predictions)

def create_wine_fuzzy_system(params=None):
    alcohol = ctrl.Antecedent(np.arange(10, 16, 0.1), 'alcohol')
    malic_acid = ctrl.Antecedent(np.arange(0, 6, 0.01), 'malic_acid')
    flavanoids = ctrl.Antecedent(np.arange(0, 5, 0.01), 'flavanoids')
    color_intensity = ctrl.Antecedent(np.arange(0, 15, 0.01), 'color_intensity')

    wine_class = ctrl.Consequent(np.arange(1, 4, 1), 'wine_class')

    if params is not None:
        p_al = np.sort(params[:9])
        p_ma = np.sort(params[9:18])
        p_fl = np.sort(params[18:27])
        p_ci = np.sort(params[27:36])

        alcohol['low'] = fuzz.trimf(alcohol.universe, [p_al[0], p_al[1], p_al[2]])
        alcohol['medium'] = fuzz.trimf(alcohol.universe, [p_al[3], p_al[4], p_al[5]])
        alcohol['high'] = fuzz.trimf(alcohol.universe, [p_al[6], p_al[7], p_al[8]])
        malic_acid['low'] = fuzz.trimf(malic_acid.universe, [p_ma[0], p_ma[1], p_ma[2]])
        malic_acid['medium'] = fuzz.trimf(malic_acid.universe, [p_ma[3], p_ma[4], p_ma[5]])
        malic_acid['high'] = fuzz.trimf(malic_acid.universe, [p_ma[6], p_ma[7], p_ma[8]])
        flavanoids['low'] = fuzz.trimf(flavanoids.universe, [p_fl[0], p_fl[1], p_fl[2]])
        flavanoids['medium'] = fuzz.trimf(flavanoids.universe, [p_fl[3], p_fl[4], p_fl[5]])
        flavanoids['high'] = fuzz.trimf(flavanoids.universe, [p_fl[6], p_fl[7], p_fl[8]])
        color_intensity['low'] = fuzz.trimf(color_intensity.universe, [p_ci[0], p_ci[1], p_ci[2]])
        color_intensity['medium'] = fuzz.trimf(color_intensity.universe, [p_ci[3], p_ci[4], p_ci[5]])
        color_intensity['high'] = fuzz.trimf(color_intensity.universe, [p_ci[6], p_ci[7], p_ci[8]])

    else:
        alcohol['low'] = fuzz.trimf(alcohol.universe, [10, 11, 12.5])
        alcohol['medium'] = fuzz.trimf(alcohol.universe, [12, 13, 14])
        alcohol['high'] = fuzz.trimf(alcohol.universe, [13, 14.5, 16])
        malic_acid['low'] = fuzz.trimf(malic_acid.universe, [0, 1, 2])
        malic_acid['medium'] = fuzz.trimf(malic_acid.universe, [1.5, 2.5, 3.5])
        malic_acid['high'] = fuzz.trimf(malic_acid.universe, [3, 4.5, 6])
        flavanoids['low'] = fuzz.trimf(flavanoids.universe, [0, 1, 2])
        flavanoids['medium'] = fuzz.trimf(flavanoids.universe, [1.5, 2.5, 3.5])
        flavanoids['high'] = fuzz.trimf(flavanoids.universe, [3, 4, 5])
        color_intensity['low'] = fuzz.trimf(color_intensity.universe, [0, 2, 5])
        color_intensity['medium'] = fuzz.trimf(color_intensity.universe, [4, 6, 9])
        color_intensity['high'] = fuzz.trimf(color_intensity.universe, [8, 12, 15])

    wine_class['class_1'] = fuzz.trimf(wine_class.universe, [1, 1, 1])
    wine_class['class_2'] = fuzz.trimf(wine_class.universe, [2, 2, 2])
    wine_class['class_3'] = fuzz.trimf(wine_class.universe, [3, 3, 3])

    rule1 = ctrl.Rule(alcohol['high'] & flavanoids['high'], wine_class['class_1'])
    rule2 = ctrl.Rule(alcohol['medium'] & flavanoids['medium'], wine_class['class_2'])
    rule3 = ctrl.Rule(alcohol['low'] & flavanoids['low'], wine_class['class_2'])
    rule4 = ctrl.Rule(color_intensity['high'] & flavanoids['low'], wine_class['class_3'])
    rule5 = ctrl.Rule(color_intensity['medium'] & flavanoids['low'], wine_class['class_2'])
    rule6 = ctrl.Rule(alcohol['high'] & color_intensity['low'], wine_class['class_1'])
    rule7 = ctrl.Rule(alcohol['low'] & color_intensity['high'], wine_class['class_3'])
    rule8 = ctrl.Rule(alcohol['high'] & flavanoids['medium'] & color_intensity['low'], wine_class['class_1'])
    rule9 = ctrl.Rule(alcohol['medium'] & malic_acid['high'] & flavanoids['low'], wine_class['class_3'])
    rule10 = ctrl.Rule(flavanoids['high'] & color_intensity['medium'] & malic_acid['low'], wine_class['class_1'])
    rule11 = ctrl.Rule(alcohol['low'] & malic_acid['medium'] & color_intensity['medium'], wine_class['class_2'])

    wine_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11])
    return ctrl.ControlSystemSimulation(wine_ctrl)

def predict_for_wine(fuzzy_system, X_data):
    predictions = []
    for sample in X_data:
        try:
            fuzzy_system.input['alcohol'] = sample[0]
            fuzzy_system.input['malicacid'] = sample[1]
            fuzzy_system.input['color_intensity'] = sample[2]
            fuzzy_system.compute()
            predicted_class = np.round(fuzzy_system.output['wine_class'])
            predictions.append(predicted_class)
        except Exception:
            predictions.append(2)
    return np.array(predictions)

def create_seeds_fuzzy_system(params=None):

    area = ctrl.Antecedent(np.arange(10, 25, 0.1), 'area')
    perimeter = ctrl.Antecedent(np.arange(12, 18, 0.1), 'perimeter')
    compactness = ctrl.Antecedent(np.arange(0.80, 0.92, 0.001), 'compactness')
    length = ctrl.Antecedent(np.arange(4.5, 7, 0.1), 'length')
    width = ctrl.Antecedent(np.arange(2.5, 4.5, 0.1), 'width')
    asymmetry = ctrl.Antecedent(np.arange(0, 8, 0.1), 'asymmetry')
    groove = ctrl.Antecedent(np.arange(4, 7, 0.1), 'groove')

    seed_class = ctrl.Consequent(np.arange(1, 4, 1), 'seed_class')

    if params is not None:
        p_area = np.sort(params[0:9])
        p_perim = np.sort(params[9:18])
        p_comp = np.sort(params[18:27])
        p_len = np.sort(params[27:36])
        p_width = np.sort(params[36:45])
        p_asym = np.sort(params[45:54])
        p_groove = np.sort(params[54:63])

        area['small'] = fuzz.trimf(area.universe, [p_area[0], p_area[1], p_area[2]])
        area['medium'] = fuzz.trimf(area.universe, [p_area[3], p_area[4], p_area[5]])
        area['large'] = fuzz.trimf(area.universe, [p_area[6], p_area[7], p_area[8]])
        perimeter['small'] = fuzz.trimf(perimeter.universe, [p_perim[0], p_perim[1], p_perim[2]])
        perimeter['medium'] = fuzz.trimf(perimeter.universe, [p_perim[3], p_perim[4], p_perim[5]])
        perimeter['large'] = fuzz.trimf(perimeter.universe, [p_perim[6], p_perim[7], p_perim[8]])
        compactness['low'] = fuzz.trimf(compactness.universe, [p_comp[0], p_comp[1], p_comp[2]])
        compactness['medium'] = fuzz.trimf(compactness.universe, [p_comp[3], p_comp[4], p_comp[5]])
        compactness['high'] = fuzz.trimf(compactness.universe, [p_comp[6], p_comp[7], p_comp[8]])
        length['short'] = fuzz.trimf(length.universe, [p_len[0], p_len[1], p_len[2]])
        length['medium'] = fuzz.trimf(length.universe, [p_len[3], p_len[4], p_len[5]])
        length['long'] = fuzz.trimf(length.universe, [p_len[6], p_len[7], p_len[8]])
        width['narrow'] = fuzz.trimf(width.universe, [p_width[0], p_width[1], p_width[2]])
        width['medium'] = fuzz.trimf(width.universe, [p_width[3], p_width[4], p_width[5]])
        width['wide'] = fuzz.trimf(width.universe, [p_width[6], p_width[7], p_width[8]])
        asymmetry['low'] = fuzz.trimf(asymmetry.universe, [p_asym[0], p_asym[1], p_asym[2]])
        asymmetry['medium'] = fuzz.trimf(asymmetry.universe, [p_asym[3], p_asym[4], p_asym[5]])
        asymmetry['high'] = fuzz.trimf(asymmetry.universe, [p_asym[6], p_asym[7], p_asym[8]])
        groove['short'] = fuzz.trimf(groove.universe, [p_groove[0], p_groove[1], p_groove[2]])
        groove['medium'] = fuzz.trimf(groove.universe, [p_groove[3], p_groove[4], p_groove[5]])
        groove['long'] = fuzz.trimf(groove.universe, [p_groove[6], p_groove[7], p_groove[8]])

    else:
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

    seed_class.defuzzify_method = 'centroid'
    seed_class['class_1'] = fuzz.trimf(seed_class.universe, [1, 1, 2])
    seed_class['class_2'] = fuzz.trimf(seed_class.universe, [1, 2, 3])
    seed_class['class_3'] = fuzz.trimf(seed_class.universe, [2, 3, 3])

    rule1 = ctrl.Rule(area['large'] | width['wide'] | length['long'], seed_class['class_2'])
    rule2 = ctrl.Rule(groove['long'] | perimeter['large'], seed_class['class_2'])
    rule3 = ctrl.Rule(compactness['medium'] | area['medium'], seed_class['class_1'])
    rule4 = ctrl.Rule(asymmetry['high'] | area['small'], seed_class['class_3'])
    rule5 = ctrl.Rule(groove['short'] | compactness['high'], seed_class['class_3'])
    seeds_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    return ctrl.ControlSystemSimulation(seeds_ctrl)

def predict_for_seeds(fuzzy_system, data):
    predictions = []
    for row in data:
        try:
            fuzzy_system.input['area'] = row[0]
            fuzzy_system.input['perimeter'] = row[1]
            fuzzy_system.input['compactness'] = row[2]
            fuzzy_system.input['length'] = row[3]
            fuzzy_system.input['width'] = row[4]
            fuzzy_system.input['asymmetry'] = row[5]
            fuzzy_system.input['groove'] = row[6]
            fuzzy_system.compute()
            prediction = fuzzy_system.output['seed_class']

            predictions.append(round(prediction))
        except Exception as e:
            predictions.append(1)
    return np.array(predictions)

def iris_model(X_iris, y_iris):
    # Trenowanie i testowanie IRIS
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
    )

    print("\n\n\nPodział danych IRIS:")
    print(f"Training: {X_train.shape[0]}")
    print(f"Test: {X_test.shape[0]}")

    def objective_function_iris(solution):
        temp_fuzzy_system = create_iris_fuzzy_system(params=solution)
        y_pred = predict_for_iris(temp_fuzzy_system, X_train)
        return 1.0 - accuracy_score(y_train, y_pred)

    problem_dict = {
        "obj_func": objective_function_iris,
        "bounds": FloatVar(
            lb=[
                4.0, 4.0, 4.5, 5.0, 4.8, 5.5, 6.2, 5.8, 6.3, 7.0, 7.5,  # Sepal Length
                2.0, 2.0, 2.5, 2.8, 2.5, 3.0, 3.5, 3.3, 3.6, 4.5, 4.5,  # Sepal Width
                1.0, 1.0, 1.5, 2.0, 2.0, 4.0, 4.8, 4.5, 5.0, 6.5, 7.0,  # Petal Length
                0.0, 0.0, 0.2, 0.5, 0.4, 1.0, 1.6, 1.4, 1.8, 2.4, 2.4  # Petal Width
            ],
            ub=[
                5.0, 5.0, 5.5, 6.0, 5.5, 6.2, 7.0, 6.5, 7.5, 8.5, 8.5,  # Sepal Length
                2.8, 2.8, 3.0, 3.3, 3.2, 3.8, 4.0, 4.0, 4.5, 5.0, 5.0,  # Sepal Width
                2.0, 2.0, 2.5, 3.0, 4.5, 5.0, 5.5, 5.5, 6.5, 7.5, 7.5,  # Petal Length
                0.3, 0.3, 0.6, 0.9, 1.2, 1.8, 2.0, 2.0, 2.5, 2.6, 2.6  # Petal Width
            ]
        ),
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }
    optimizer = OriginalGWO(epoch=50, pop_size=30)
    result = optimizer.solve(problem_dict)

    best_solution = result.solution
    best_fitness = result.target.fitness

    print(f"Zakończono dla IRIS.")
    print(f"Best Fitness (Error Rate on Train Set): {best_fitness:.4f}")
    print(f"Best Parameters Found: \n{best_solution}\n")

    original_fuzzy_system = create_iris_fuzzy_system(params=None)
    y_pred_original = predict_for_iris(original_fuzzy_system, X_test)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    print(f"Accuracy ORIGINAL Fuzzy System: {accuracy_original:.4f}")

    optimized_fuzzy_system = create_iris_fuzzy_system(params=best_solution)
    y_pred_optimized = predict_for_iris(optimized_fuzzy_system, X_test)
    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    print(f"Accuracy GWO-OPTIMIZED Fuzzy System: {accuracy_optimized:.4f} \n")

    display_metrics(y_test, y_pred_original, "Accuracy ORIGINAL IRIS")
    display_metrics(y_test, y_pred_optimized, "Accuracy GWO-OPTIMIZED IRIS")

def wine_model(X_wine, y_wine):
    # Trenowanie i testowanie WINE
    X_train, X_test, y_train, y_test = train_test_split(
        X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
    )
    print("\n\n\nPodział danych WINE:")
    print(f"Training: {X_train.shape[0]}")
    print(f"Test: {X_test.shape[0]}")

    def objective_function_wine(solution):
        temp_fuzzy_system = create_wine_fuzzy_system(params=solution)
        y_pred = predict_for_wine(temp_fuzzy_system, X_train)
        return 1.0 - accuracy_score(y_train, y_pred)

    problem_dict = {
        "obj_func": objective_function_wine,
        "bounds": FloatVar(
            lb=[
                11.0, 11.2, 11.8, 11.5, 12.5, 13.5, 13.0, 14.0, 14.5,  # Alcohol
                0.5, 1.0, 1.5, 1.4, 2.0, 3.0, 2.8, 4.0, 5.0,  # Malic Acid
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5,  # Flavanoids
                1.0, 2.0, 3.5, 3.0, 4.5, 6.5, 6.0, 8.0, 12.0,  # Color Intensity
            ],
            ub=[
                12.0, 12.5, 13.0, 13.0, 13.8, 14.5, 14.0, 15.0, 15.0,  # Alcohol
                1.5, 2.0, 2.5, 2.8, 3.5, 4.5, 4.0, 5.5, 6.0,  # Malic Acid
                1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 4.8, 5.0,  # Flavanoids
                3.0, 4.0, 5.0, 6.0, 7.5, 8.5, 9.0, 13.0, 14.0,  # Color Intensity
            ]
        ),
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    optimizer = OriginalGWO(epoch=50, pop_size=30)
    result = optimizer.solve(problem_dict)

    best_solution = result.solution
    best_fitness = result.target.fitness

    print(f"Zakończono dla WINE.")
    print(f"Best Fitness (Error Rate on Train Set): {best_fitness:.4f}")
    print(f"Best Parameters Found: \n{best_solution}\n")

    original_fuzzy_system = create_wine_fuzzy_system(params=None)
    y_pred_original = predict_for_wine(original_fuzzy_system, X_test)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    print(f"Accuracy ORIGINAL Fuzzy System: {accuracy_original:.4f}")

    optimized_fuzzy_system = create_wine_fuzzy_system(params=best_solution)
    y_pred_optimized = predict_for_wine(optimized_fuzzy_system, X_test)
    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    print(f"Accuracy GWO-OPTIMIZED Fuzzy System: {accuracy_optimized:.4f} \n")

    display_metrics(y_test, y_pred_original, "Accuracy ORIGINAL WINE")
    display_metrics(y_test, y_pred_optimized, "Accuracy GWO-OPTIMIZED WINE")

def seeds_model(X_seeds, y_seeds):
    # Trenowanie i testowanie SEEDS
    X_train, X_test, y_train, y_test = train_test_split(
        X_seeds, y_seeds, test_size=0.2, random_state=42, stratify=y_seeds
    )

    print("\n\n\nPodział danych SEEDS:")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    def objective_function_seeds(solution):
        temp_fuzzy_system = create_seeds_fuzzy_system(params=solution)
        y_pred = predict_for_seeds(temp_fuzzy_system, X_train)
        return 1.0 - accuracy_score(y_train, y_pred)

    problem_dict = {
        "obj_func": objective_function_seeds,
        "bounds": FloatVar(
            lb=[
                10, 11, 13, 12, 14, 17, 16, 18, 22,  # Area
                12, 12.5, 13.5, 13, 14.5, 15.5, 15, 16, 17,  # Perimeter
                0.80, 0.82, 0.84, 0.83, 0.86, 0.88, 0.87, 0.89, 0.91,  # Compactness
                4.5, 4.8, 5.2, 5.0, 5.5, 6.0, 5.8, 6.2, 6.8,  # Length
                2.5, 2.8, 3.1, 3.0, 3.3, 3.6, 3.5, 3.8, 4.2,  # Width
                0, 1, 2.5, 2, 3.5, 5, 4.5, 5.5, 7.5,  # Asymmetry
                4, 4.2, 4.8, 4.5, 5.2, 5.7, 5.5, 6.0, 6.8,  # Groove
            ],
            ub=[
                13, 14, 16, 17, 18, 20, 22, 24, 25,  # Area
                13.5, 14, 15, 15.5, 16, 16.5, 17, 17.5, 18,  # Perimeter
                0.84, 0.85, 0.87, 0.88, 0.89, 0.90, 0.90, 0.91, 0.92,  # Compactness
                5.2, 5.4, 5.8, 6.0, 6.2, 6.5, 6.8, 6.9, 7.0,  # Length
                3.1, 3.3, 3.5, 3.6, 3.8, 4.0, 4.2, 4.4, 4.5,  # Width
                2.5, 3, 4.5, 5, 5.5, 6.5, 7.5, 7.8, 8.0,  # Asymmetry
                4.8, 5.0, 5.5, 5.7, 5.9, 6.2, 6.8, 6.9, 7.0,  # Groove
            ]
        ),
        "minmax": "min",
        "log_to": None,
        "save_population": False,
    }

    optimizer = OriginalGWO(epoch=50, pop_size=30)
    result = optimizer.solve(problem_dict)

    best_solution = result.solution
    best_fitness = result.target.fitness

    print(f"Zakończono dla SEEDS.")
    print(f"Best Fitness (Error Rate on Train Set): {best_fitness:.4f}")
    print(f"Best Parameters Found: \n{best_solution}\n")

    original_fuzzy_system = create_seeds_fuzzy_system(params=None)
    y_pred_original = predict_for_seeds(original_fuzzy_system, X_test)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    print(f"Accuracy ORIGINAL Fuzzy System: {accuracy_original:.4f}")

    optimized_fuzzy_system = create_seeds_fuzzy_system(params=best_solution)
    y_pred_optimized = predict_for_seeds(optimized_fuzzy_system, X_test)
    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    print(f"Accuracy GWO-OPTIMIZED Fuzzy System: {accuracy_optimized:.4f} \n")

    display_metrics(y_test, y_pred_original, "Accuracy ORIGINAL SEEDS")
    display_metrics(y_test, y_pred_optimized, "Accuracy GWO-OPTIMIZED SEEDS")

def display_metrics(y_true, y_pred, title):
    print(f"--- {title} ---")

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix\n{cm}")
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\nDokładność (Accuracy): {accuracy:.4f}")
    print(f"Precyzja (Precision): {precision:.4f}")
    print(f"Czułość (Recall): {recall:.4f}")
    print(f"Miara F1 (F1-Score): {f1:.4f}\n")

    print("Raport klasyfikacji:")
    print(classification_report(y_true, y_pred, zero_division=0, target_names=['Klasa 1', 'Klasa 2', 'Klasa 3']))





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

    print("\nEtykiety (y_wine):")
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

    print("\nEtykiety (y_seeds):")
    print(np.unique(y_seeds))
    print("Rozmiar przed eliminacją wartości skrajnych:", X_seeds.shape)
    print("Rozmiar po eliminacji wartości skrajnych:", X_seeds_no_outliers.shape)

    X_seeds = X_seeds_no_outliers
    y_seeds = y_seeds_no_outliers

    # iris_model(X_iris, y_iris)
    wine_model(X_wine, y_wine)
    # seeds_model(X_seeds, y_seeds)



if __name__ == '__main__':
    main()