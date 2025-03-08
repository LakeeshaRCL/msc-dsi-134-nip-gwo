import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.interchange import dataframe

population = {
    "x1" :[5.34, 2.45, 4.45, 1.45],
    "x2": [ 7.68, -4.54,  3.25,  3.5 ]
}

def get_fitness(x1, x2):
    return math.pow((x1 + 2*x2 -7),2) + math.pow((2*x1 + x2 -5), 2)

def find_best_wolves(dataframe: DataFrame, fitness_key="fitness"):
    minimums = dataframe.nsmallest(3,fitness_key)
    # print(minimums)

    return minimums.iloc[0].copy(), minimums.iloc[1].copy(), minimums.iloc[2].copy()

def view_best_wolves(x_alpha, x_beta, x_delta):
    print()
    print("x_alpha")
    print(x_alpha)
    print()

    print("x_beta")
    print(x_beta)
    print()

    print("x_delta")
    print(x_delta)
    print()
def gwo(max_iterations: int, dataframe: DataFrame, x_alpha:pd.Series, x_beta:pd.Series, x_delta:pd.Series):
    print("Running GWO....")
    print()
    # other parameters were hardcoded
    r1 = 0.4
    r2 = 0.2
    a = 1

    A = round(2 * a * r1 - a,2)
    C = 2 * r2

    print("A", A)
    print("C", C)

    for iteration in range(max_iterations):
        for index, row in dataframe.iterrows():

            X_t = np.array([row["x1"], row["x2"]])
            X_alpha = np.array([x_alpha["x1"], x_alpha["x2"]])
            X_beta = np.array([x_beta["x1"], x_beta["x2"]])
            X_delta = np.array([x_delta["x1"], x_delta["x2"]])
            # print("X_t: ", X_t)
            print("X_alpha: ", X_alpha)
            # print("X_beta: ", X_beta)
            # print("X_delta: ", X_delta)

            D_alpha = abs(C * X_alpha - X_t)
            D_beta = abs(C * X_beta - X_t)
            D_delta = abs(C * X_delta - X_t)

            X1 = X_alpha - A * D_alpha
            X2 = X_beta - A * D_beta
            X3 = X_delta - A * D_delta

            x_new = (X1 + X2 + X3) / 3
            new_fitness = get_fitness(x_new[0], x_new[1])

            print("D_alpha: ", D_alpha)
            print("D_beta: ", D_beta)
            print("D_delta: ", D_delta)
            print("X1: ", X1)
            print("X2: ", X2)
            print("X3: ", X3)
            print("x_new: ", x_new)
            print("new_fitness: ", new_fitness)
            print("current_fitness: ", row["fitness"])
            print()

            if new_fitness < row["fitness"]:
                # update dataframe
                print("updating values for the index {0}...".format(index))
                dataframe.at[index, "x1"] = x_new[0]
                dataframe.at[index, "x2"] = x_new[1]
                dataframe.at[index, "fitness"] = new_fitness
    print("GWO Ends....")
    return dataframe


population_df = pd.DataFrame(population)



# calculate initial fitness
for index, row in population_df.iterrows():
    population_df.at[index, 'fitness'] = get_fitness(row["x1"], row["x2"])


print(population_df)

x_alpha, x_beta, x_delta = find_best_wolves(population_df)
view_best_wolves(x_alpha, x_beta, x_delta)

# optimize
new_df = gwo(1, population_df, x_alpha, x_beta, x_delta)
print("Optimized DF")
print(new_df)


x_alpha, x_beta, x_delta = find_best_wolves(population_df)
view_best_wolves(x_alpha, x_beta, x_delta)