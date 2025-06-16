import numpy as np
import pandas as pd
from mealpy import FloatVar
from mealpy.swarm_based.GWO import OriginalGWO
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from helper import remove_outliers_turkey, display_metrics
plt.legend(loc='best')



def create_iris_fuzzy_system(params=None):
    sepal_length = ctrl.Antecedent(np.arange(4.3, 8.0, 0.1), 'sepal_length')
    sepal_width = ctrl.Antecedent(np.arange(2.0, 4.5, 0.1), 'sepal_width')
    petal_length = ctrl.Antecedent(np.arange(1.0, 7.0, 0.1), 'petal_length')
    petal_width = ctrl.Antecedent(np.arange(0.1, 2.6, 0.1), 'petal_width')

    iris_class = ctrl.Consequent(np.arange(1, 4, 0.1), 'iris_class', defuzzify_method='centroid')

    if params is not None:

        sepal_length['low'] = fuzz.trimf(sepal_length.universe, [4.3, 5.1, 5.8])
        sepal_length['medium'] = fuzz.trimf(sepal_length.universe, [5.4, 6.1, 6.9])
        sepal_length['high'] = fuzz.trimf(sepal_length.universe, [6.4, 7.1, 7.9])
        sepal_width['low'] = fuzz.trimf(sepal_width.universe, [2.0, 2.8, 3.4])
        sepal_width['medium'] = fuzz.trimf(sepal_width.universe, [2.9, 3.5, 4.1])
        sepal_width['high'] = fuzz.trimf(sepal_width.universe, [3.6, 4.0, 4.4])
        petal_length['low'] = fuzz.trimf(petal_length.universe, [1.0, 1.5, 1.9])
        petal_length['medium'] = fuzz.trimf(petal_length.universe, [3.0, 4.2, 5.1])
        petal_length['high'] = fuzz.trimf(petal_length.universe, [4.5, 5.7, 6.9])
        petal_width['low'] = fuzz.trapmf(petal_width.universe, [0.1, 0.2, 0.4, 0.6])
        petal_width['medium'] = fuzz.trapmf(petal_width.universe, [1.0, 1.2, 1.5, 1.8])
        petal_width['high'] = fuzz.trapmf(petal_width.universe, [1.5, 1.8, 2.4, 2.5])

    else:
        sepal_length['low'] = fuzz.trimf(sepal_length.universe, [4.3, 5.1, 5.8])
        sepal_length['medium'] = fuzz.trimf(sepal_length.universe, [5.4, 6.1, 6.9])
        sepal_length['high'] = fuzz.trimf(sepal_length.universe, [6.4, 7.1, 7.9])
        sepal_width['low'] = fuzz.trimf(sepal_width.universe, [2.0, 2.8, 3.4])
        sepal_width['medium'] = fuzz.trimf(sepal_width.universe, [2.9, 3.5, 4.1])
        sepal_width['high'] = fuzz.trimf(sepal_width.universe, [3.6, 4.0, 4.4])
        petal_length['low'] = fuzz.trimf(petal_length.universe, [1.0, 1.5, 1.9])
        petal_length['medium'] = fuzz.trimf(petal_length.universe, [3.0, 4.2, 5.1])
        petal_length['high'] = fuzz.trimf(petal_length.universe, [4.5, 5.7, 6.9])
        petal_width['low'] = fuzz.trapmf(petal_width.universe, [0.1, 0.2, 0.4, 0.6])
        petal_width['medium'] = fuzz.trapmf(petal_width.universe, [1.0, 1.2, 1.5, 1.8])
        petal_width['high'] = fuzz.trapmf(petal_width.universe, [1.5, 1.8, 2.4, 2.5])

    iris_class['setosa'] = fuzz.trimf(iris_class.universe, [1, 1, 2])
    iris_class['versicolor'] = fuzz.trimf(iris_class.universe, [2, 2.2, 2.5])
    iris_class['virginica'] = fuzz.trimf(iris_class.universe, [2, 3, 3])

    rule1 = ctrl.Rule(petal_width['low'], iris_class['setosa'])
    rule2 = ctrl.Rule(petal_length['low'], iris_class['setosa'])
    rule3 = ctrl.Rule(petal_width['medium'] & petal_length['medium'], iris_class['versicolor'])
    rule4 = ctrl.Rule(petal_width['high'], iris_class['virginica'])
    rule5 = ctrl.Rule(petal_length['high'], iris_class['virginica'])

    petal_length.view()
    petal_width.view()
    iris_class.view()

    iris_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
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
    print("\n$$$", predictions)
    return np.array(predictions)

def create_wine_fuzzy_system(params=None):
    alcohol = ctrl.Antecedent(np.arange(11, 15, 0.01), 'alcohol')
    malic_acid = ctrl.Antecedent(np.arange(0.5, 6, 0.01), 'malic_acid')
    flavanoids = ctrl.Antecedent(np.arange(0.5, 6, 0.01), 'flavanoids')
    proline = ctrl.Antecedent(np.arange(200, 1700, 10), 'proline')

    wine_class = ctrl.Consequent(np.arange(0, 4, 1), 'wine_class')

    if params is not None:
        p_al = np.sort(params[:11])
        p_ma = np.sort(params[11:22])
        p_fl = np.sort(params[22:33])
        p_pr = np.sort(params[33:44])

        alcohol['low'] = fuzz.trapmf(alcohol.universe, [p_al[0], p_al[1], p_al[2], p_al[3]])
        alcohol['medium'] = fuzz.trimf(alcohol.universe, [p_al[4], p_al[5], p_al[6]])
        alcohol['high'] = fuzz.trapmf(alcohol.universe, [p_al[7], p_al[8], p_al[9], p_al[10]])
        malic_acid['low'] = fuzz.trapmf(malic_acid.universe, [p_ma[0], p_ma[1], p_ma[2], p_ma[3]])
        malic_acid['medium'] = fuzz.trimf(malic_acid.universe, [p_ma[4], p_ma[5], p_ma[6]])
        malic_acid['high'] = fuzz.trapmf(malic_acid.universe, [p_ma[7], p_ma[8], p_ma[9], p_ma[10]])
        flavanoids['low'] = fuzz.trapmf(flavanoids.universe, [p_fl[0], p_fl[1], p_fl[2], p_fl[3]])
        flavanoids['medium'] = fuzz.trimf(flavanoids.universe, [p_fl[4], p_fl[5], p_fl[6]])
        flavanoids['high'] = fuzz.trapmf(flavanoids.universe, [p_fl[7], p_fl[8], p_fl[9], p_fl[10]])
        proline['low'] = fuzz.trapmf(proline.universe, [p_pr[0], p_pr[1], p_pr[2], p_pr[3]])
        proline['medium'] = fuzz.trimf(proline.universe, [p_pr[4], p_pr[5], p_pr[6]])
        proline['high'] = fuzz.trapmf(proline.universe, [p_pr[7], p_pr[8], p_pr[9], p_pr[10]])

    else:
        alcohol['low'] = fuzz.trapmf(alcohol.universe, [11, 11, 12.5, 13])
        alcohol['medium'] = fuzz.trimf(alcohol.universe, [12.5, 13.5, 14])
        alcohol['high'] = fuzz.trapmf(alcohol.universe, [13.5, 14, 15, 15])
        malic_acid['low'] = fuzz.trapmf(malic_acid.universe, [0.5, 0.5, 1.5, 2])
        malic_acid['medium'] = fuzz.trimf(malic_acid.universe, [1.5, 2.5, 3.5])
        malic_acid['high'] = fuzz.trapmf(malic_acid.universe, [3, 3.5, 6, 6])
        flavanoids['low'] = fuzz.trapmf(flavanoids.universe, [0.1, 0.1, 1, 1.5])
        flavanoids['medium'] = fuzz.trimf(flavanoids.universe, [1, 2, 3])
        flavanoids['high'] = fuzz.trapmf(flavanoids.universe, [2.5, 3, 6, 6])
        proline['low'] = fuzz.trapmf(proline.universe, [200, 200, 800, 1000])
        proline['medium'] = fuzz.trimf(proline.universe, [800, 1000, 1200])
        proline['high'] = fuzz.trapmf(proline.universe, [1100, 1300, 1700, 1700])

    wine_class['class_1'] = fuzz.trimf(wine_class.universe, [0, 1, 1.5])
    wine_class['class_2'] = fuzz.trimf(wine_class.universe, [1, 2, 2.5])
    wine_class['class_3'] = fuzz.trimf(wine_class.universe, [2, 3, 3.5])

    # rule1 = ctrl.Rule(alcohol['high'] & flavanoids['high'] & proline['high'], wine_class['class_1'])
    # rule2 = ctrl.Rule(alcohol['medium'] & malic_acid['high'] & flavanoids['low'], wine_class['class_2'])
    # rule3 = ctrl.Rule(alcohol['low'] & malic_acid['high'] & flavanoids['low'] & proline['low'], wine_class['class_3'])
    # rule4 = ctrl.Rule(alcohol['medium'] & malic_acid['medium'] & flavanoids['medium'], wine_class['class_2'])
    # rule5 = ctrl.Rule(alcohol['high'] & malic_acid['low'] & flavanoids['high'], wine_class['class_1'])
    # rule6 = ctrl.Rule(alcohol['low'] & malic_acid['high'] & proline['medium'], wine_class['class_3'])
    # rule7 = ctrl.Rule(alcohol['medium'] & flavanoids['medium'] & proline['medium'], wine_class['class_2'])
    # rule8 = ctrl.Rule(alcohol['high'] & malic_acid['medium'] & flavanoids['high'], wine_class['class_1'])
    rules = [
        ctrl.Rule(alcohol['high'] & flavanoids['high'] & malic_acid['low'], wine_class['class_1']),
        ctrl.Rule(alcohol['high'] & proline['high'] & malic_acid['low'], wine_class['class_1']),
        ctrl.Rule(flavanoids['high'] & proline['high'] & malic_acid['medium'], wine_class['class_1']),
        ctrl.Rule(alcohol['medium'] & malic_acid['medium'] & flavanoids['medium'], wine_class['class_2']),
        ctrl.Rule(alcohol['medium'] & malic_acid['high'] & flavanoids['low'], wine_class['class_2']),
        ctrl.Rule(alcohol['medium'] & proline['medium'] & malic_acid['high'], wine_class['class_2']),
        ctrl.Rule(alcohol['low'] & malic_acid['high'] & flavanoids['low'], wine_class['class_3']),
        ctrl.Rule(alcohol['low'] & proline['low'] & malic_acid['high'], wine_class['class_3']),
        ctrl.Rule(flavanoids['low'] & proline['low'] & malic_acid['high'], wine_class['class_3']),
        ctrl.Rule(alcohol['high'] & flavanoids['medium'] & malic_acid['medium'], wine_class['class_1']),
        ctrl.Rule(alcohol['medium'] & flavanoids['medium'] & malic_acid['medium'], wine_class['class_2']),
        ctrl.Rule(alcohol['low'] & flavanoids['low'] & malic_acid['medium'], wine_class['class_3'])
    ]

    wine_ctrl = ctrl.ControlSystem(rules)
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
            print(f'Error: {e}')
            predictions.append(1)
    return np.array(predictions)

def iris_model(X_iris, y_iris):
    # Trenowanie i testowanie IRIS
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.1, random_state=42, stratify=y_iris
    )

    print("\n\n\nPodział danych IRIS:")
    print(f"Training: {X_train.shape[0]}")
    print(f"Test: {X_test.shape[0]}")

    # def objective_function_iris(solution):
    #     temp_fuzzy_system = create_iris_fuzzy_system(params=solution)
    #     y_pred = predict_for_iris(temp_fuzzy_system, X_train)
    #     return 1.0 - accuracy_score(y_train, y_pred)
    #
    # problem_dict = {
    #     "obj_func": objective_function_iris,
    #     "bounds": FloatVar(
    #         lb=[
    #             4.0, 4.0, 4.5, 5.0, 4.8, 5.5, 6.2, 5.8, 6.3, 7.0, 7.5,  # Sepal Length
    #             2.0, 2.0, 2.5, 2.8, 2.5, 3.0, 3.5, 3.3, 3.6, 4.5, 4.5,  # Sepal Width
    #             1.0, 1.0, 1.5, 2.0, 2.0, 4.0, 4.8, 4.5, 5.0, 6.5, 7.0,  # Petal Length
    #             0.0, 0.0, 0.2, 0.5, 0.4, 1.0, 1.6, 1.4, 1.8, 2.4, 2.4  # Petal Width
    #         ],
    #         ub=[
    #             5.0, 5.0, 5.5, 6.0, 5.5, 6.2, 7.0, 6.5, 7.5, 8.5, 8.5,  # Sepal Length
    #             2.8, 2.8, 3.0, 3.3, 3.2, 3.8, 4.0, 4.0, 4.5, 5.0, 5.0,  # Sepal Width
    #             2.0, 2.0, 2.5, 3.0, 4.5, 5.0, 5.5, 5.5, 6.5, 7.5, 7.5,  # Petal Length
    #             0.3, 0.3, 0.6, 0.9, 1.2, 1.8, 2.0, 2.0, 2.5, 2.6, 2.6  # Petal Width
    #         ]
    #     ),
    #     "minmax": "min",
    #     "log_to": None,
    #     "save_population": False,
    # }
    # optimizer = OriginalGWO(epoch=50, pop_size=30)
    # result = optimizer.solve(problem_dict)
    #
    # best_solution = result.solution
    # best_fitness = result.target.fitness
    #
    # print(f"Zakończono dla IRIS.")
    # print(f"Best Fitness (Error Rate on Train Set): {best_fitness:.4f}")
    # print(f"Best Parameters Found: \n{best_solution}\n")

    original_fuzzy_system = create_iris_fuzzy_system(params=None)
    y_pred_original = predict_for_iris(original_fuzzy_system, X_test)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    print(f"Accuracy ORIGINAL Fuzzy System: {accuracy_original:.4f}")

    # optimized_fuzzy_system = create_iris_fuzzy_system(params=best_solution)
    # y_pred_optimized = predict_for_iris(optimized_fuzzy_system, X_test)
    # accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    # print(f"Accuracy GWO-OPTIMIZED Fuzzy System: {accuracy_optimized:.4f} \n")

    display_metrics(y_test, y_pred_original, "Accuracy ORIGINAL IRIS")
    # display_metrics(y_test, y_pred_optimized, "Accuracy GWO-OPTIMIZED IRIS")

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
                # Alcohol (11 params: 4 for 'low', 3 for 'medium', 4 for 'high')
                11.0, 11.0, 11.5, 12.0,  # 'low' (trapmf: a, b, c, d)
                12.5, 13.0, 13.5,  # 'medium' (trimf: a, b, c)
                13.5, 14.0, 14.5, 15.0,  # 'high' (trapmf: a, b, c, d)

                # Malic Acid (11 params)
                0.0, 0.0, 1.0, 2.0,  # 'low' (trapmf)
                2.0, 3.0, 4.0,  # 'medium' (trimf)
                3.5, 4.0, 5.0, 6.0,  # 'high' (trapmf)

                # Flavanoids (11 params)
                0.0, 0.0, 1.0, 1.5,  # 'low' (trapmf)
                1.5, 2.5, 3.5,  # 'medium' (trimf)
                3.0, 4.0, 5.0, 5.0,  # 'high' (trapmf)

                # Proline (11 params)
                1.0, 1.0, 3.0, 5.0,  # 'low' (trapmf)
                5.0, 7.0, 9.0,  # 'medium' (trimf)
                8.0, 10.0, 12.0, 14.0,  # 'high' (trapmf)
            ],
            ub=[
                # Alcohol (11 params)
                11.5, 12.0, 12.5, 13.0,  # 'low' (trapmf)
                13.0, 13.5, 14.0,  # 'medium' (trimf)
                14.0, 14.5, 15.0, 15.0,  # 'high' (trapmf)

                # Malic Acid (11 params)
                1.0, 2.0, 2.5, 3.0,  # 'low' (trapmf)
                3.0, 3.5, 4.5,  # 'medium' (trimf)
                4.5, 5.0, 6.0, 6.0,  # 'high' (trapmf)

                # Flavanoids (11 params)
                1.0, 1.5, 2.0, 2.5,  # 'low' (trapmf)
                2.5, 3.5, 4.5,  # 'medium' (trimf)
                4.0, 5.0, 5.0, 5.0,  # 'high' (trapmf)

                # Proline (11 params)
                3.0, 5.0, 6.0, 7.0,  # 'low' (trapmf)
                7.0, 8.0, 10.0,  # 'medium' (trimf)
                11.0, 13.0, 14.0, 14.0,  # 'high' (trapmf)
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



def seeds_model_cv(X_seeds, y_seeds, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    original_scores = []
    optimized_scores = []

    for fold, (train_index, test_index) in enumerate(kfold.split(X_seeds, y_seeds)):
        print(f"\n===== FOLD {fold + 1}/{n_splits} =====\n")
        X_train, X_test = X_seeds[train_index], X_seeds[test_index]
        y_train, y_test = y_seeds[train_index], y_seeds[test_index]

        original_fuzzy_system = create_seeds_fuzzy_system(params=None)
        y_pred_original = predict_for_seeds(original_fuzzy_system, X_test)
        accuracy_original = accuracy_score(y_test, y_pred_original)
        original_scores.append(accuracy_original)
        print(f"FOLD {fold + 1} >> Accuracy ORIGINAL Fuzzy System: {accuracy_original:.4f}")

        def objective_function_seeds(solution):
            temp_fuzzy_system = create_seeds_fuzzy_system(params=solution)
            y_pred = predict_for_seeds(temp_fuzzy_system, X_train)
            return 1.0 - accuracy_score(y_train, y_pred)

        problem_dict = {
            "obj_func": objective_function_seeds,
            "bounds": FloatVar(
            lb=[
                10, 11, 13, 12, 14, 17, 16, 18, 22,
                12, 12.5, 13.5, 13, 14.5, 15.5, 15, 16, 17,
                0.80, 0.82, 0.84, 0.83, 0.86, 0.88, 0.87, 0.89, 0.91,
                4.5, 4.8, 5.2, 5.0, 5.5, 6.0, 5.8, 6.2, 6.8,
                2.5, 2.8, 3.1, 3.0, 3.3, 3.6, 3.5, 3.8, 4.2,
                0, 1, 2.5, 2, 3.5, 5, 4.5, 5.5, 7.5,
                4, 4.2, 4.8, 4.5, 5.2, 5.7, 5.5, 6.0, 6.8,
            ],
            ub=[
                13, 14, 16, 17, 18, 20, 22, 24, 25,
                13.5, 14, 15, 15.5, 16, 16.5, 17, 17.5, 18,
                0.84, 0.85, 0.87, 0.88, 0.89, 0.90, 0.90, 0.91, 0.92,
                5.2, 5.4, 5.8, 6.0, 6.2, 6.5, 6.8, 6.9, 7.0,
                3.1, 3.3, 3.5, 3.6, 3.8, 4.0, 4.2, 4.4, 4.5,
                2.5, 3, 4.5, 5, 5.5, 6.5, 7.5, 7.8, 8.0,
                4.8, 5.0, 5.5, 5.7, 5.9, 6.2, 6.8, 6.9, 7.0,
            ]),
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        }

        optimizer = OriginalGWO(epoch=50, pop_size=30)
        result = optimizer.solve(problem_dict)
        best_solution = result.solution

        optimized_fuzzy_system = create_seeds_fuzzy_system(params=best_solution)
        y_pred_optimized = predict_for_seeds(optimized_fuzzy_system, X_test)
        accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
        optimized_scores.append(accuracy_optimized)
        print(f"FOLD {fold + 1} >> Accuracy GWO-OPTIMIZED Fuzzy System: {accuracy_optimized:.4f}")

    print("\n\n--- CROSS-VALIDATION FINAL RESULTS ---\n")
    print(f"Original model accuracy: {np.mean(original_scores):.4f} +/- {np.std(original_scores):.4f}")
    print(f"GWO-Optimized model accuracy: {np.mean(optimized_scores):.4f} +/- {np.std(optimized_scores):.4f}")

def train_and_evaluate_genfis(X, y, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    n_classes = len(np.unique(y))

    print(f"\n\n--- Evaluating GENFIS (Fuzzy C-Means) Model with {n_splits}-Fold CV ---\n")

    for fold, (train_index, test_index) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        centers, u_train, _, _, _, _, _ = fuzz.cluster.cmeans(
            X_train.T, c=n_classes, m=2, error=0.005, maxiter=1000, init=None
        )

        cluster_labels = np.zeros(n_classes)
        train_labels_by_cluster = np.argmax(u_train, axis=0)
        for i in range(n_classes):
            points_in_cluster = y_train[train_labels_by_cluster == i]
            if len(points_in_cluster) > 0:
                cluster_labels[i] = np.bincount(points_in_cluster).argmax()
            else:
                cluster_labels[i] = np.random.choice(np.unique(y))

        u_test, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X_test.T, centers, m=2, error=0.005, maxiter=1000
        )

        predicted_clusters = np.argmax(u_test, axis=0)
        y_pred = np.array([cluster_labels[c] for c in predicted_clusters])

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"FOLD {fold + 1} >> Genfis Accuracy: {accuracy:.4f}")

    print("\n--- GENFIS FINAL RESULTS ---\n")
    print(f"Genfis (FCM) model accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")




def main():
    datasets = {
        'iris': ('data/iris.data', None),
        'wine': ('data/wine.data', None),
        'seeds': ('data/seeds_dataset.txt', r'\s+')
    }

    # Pre-processing IRIS
    iris_col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    iris_data = pd.read_csv(datasets['iris'][0], header=None, names=iris_col_names)
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
    wine_data = pd.read_csv(datasets['wine'][0], header=None, names=wine_col_names)
    wine_data.dropna(inplace=True)
    X_wine = wine_data.iloc[:, 1:].values
    y_wine = wine_data.iloc[:, 0].values.astype(int)

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

    iris_model(X_iris, y_iris)
    # wine_model(X_wine, y_wine)
    # seeds_model(X_seeds, y_seeds) #liczy się około 13 minut



if __name__ == '__main__':
    main()