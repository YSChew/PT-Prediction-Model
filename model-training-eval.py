import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA   
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

SELECTED_RANDOM_STATE = 32
CROSS_VAL_FOLDS = 10

features_df = pd.read_csv("data_X.csv")
labels_df = pd.read_csv("data_Y.csv")

features_df = features_df.drop(columns=["name", "slug", "courses", "reviews", "school_PROF"], axis=1)

training_array = features_df.drop(["professor_id", "avg_course_GPA"], axis=1).to_numpy()
labels_array = labels_df.drop(["professor_id"], axis=1).to_numpy()

pca = PCA(n_components=0.95)
training_array = pca.fit_transform(training_array)
print(f"Variance kept: {np.sum(pca.explained_variance_ratio_)}")

def cv_score_eval(cv_score):
    print("Cross-validated R2 scores:", cv_score)
    print("Mean Cross-validated R2:", np.mean(cv_score))
    print("Standard Deviation:", np.std(cv_score))
    print("Variance:", np.var(cv_score))
    return

def scale_x(xtrain_arr, xtest_arr):
    scaler = StandardScaler()
    xtrain_arr = scaler.fit_transform(xtrain_arr)
    xtest_arr = scaler.fit_transform(xtest_arr)
    return xtrain_arr, xtest_arr

def rf_regressor(xtrain_arr, ytrain_arr, xtest_arr, ytest_arr, max_d, criterion_sel, max_feat, rand_st=None):
    k_folds = KFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=rand_st)

    rfr = RandomForestRegressor(max_depth=max_d,random_state=rand_st, criterion=criterion_sel, max_features=max_feat)

    cv_scores = cross_val_score(rfr, xtrain_arr, ytrain_arr.ravel(), cv=k_folds, scoring="r2")
    cv_score_eval(cv_scores)
    
    rfr.fit(xtrain_arr, ytrain_arr.ravel())

    Y_pred = rfr.predict(xtest_arr)

    Y2_pred = rfr.predict(xtrain_arr)
    print("V2 R2 Score:", r2_score(ytrain_arr, Y2_pred))

    print("MAE:", mean_absolute_error(ytest_arr, Y_pred))
    print("MSE:", mean_squared_error(ytest_arr, Y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(ytest_arr, Y_pred)))
    print("R2 Score:", r2_score(ytest_arr, Y_pred))
    
def mlp_regressor(xtrain_arr, ytrain_arr, xtest_arr, ytest_arr, max_itr, layer_size, activ_func, a_val, solve, rand_st=None):
    k_folds = KFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=rand_st)

    xtrain_arr, xtest_arr = scale_x(xtrain_arr, xtest_arr)
    mlpr = MLPRegressor(random_state=rand_st, hidden_layer_sizes=layer_size, 
                        max_iter=max_itr, activation=activ_func, alpha=a_val, solver=solve)

    cv_scores = cross_val_score(mlpr, xtrain_arr, ytrain_arr.ravel(), cv=k_folds, scoring="r2")
    cv_score_eval(cv_scores)
    
    mlpr.fit(xtrain_arr, ytrain_arr.ravel())

    Y_pred = mlpr.predict(xtest_arr)

    Y2_pred = mlpr.predict(xtrain_arr)
    print("V2 R2 Score:", r2_score(ytrain_arr, Y2_pred))

    print("MAE:", mean_absolute_error(ytest_arr, Y_pred))
    print("MSE:", mean_squared_error(ytest_arr, Y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(ytest_arr, Y_pred)))
    print("R2 Score:", r2_score(ytest_arr, Y_pred))

def knn_regressor(xtrain_arr, ytrain_arr, xtest_arr, ytest_arr, n_neighbors):
    k_folds = KFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=SELECTED_RANDOM_STATE)

    knnr = KNeighborsRegressor(n_neighbors=n_neighbors, weights="uniform")

    xtrain_arr, xtest_arr = scale_x(xtrain_arr, xtest_arr)
    knnr.fit(xtrain_arr, ytrain_arr)

    Y_pred = knnr.predict(xtest_arr)

    print("R2 Score:", r2_score(ytest_arr, Y_pred))

    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    weight = 1 / rmse if rmse != 0 else 1
    
    def rmse_weight(distances):
        return np.ones_like(distances) * weight
    
    knnr = KNeighborsRegressor(n_neighbors=n_neighbors, weights=rmse_weight)

    knnr.fit(xtrain_arr, ytrain_arr)
    Y_pred = knnr.predict(xtest_arr)
    Y2_pred = knnr.predict(xtrain_arr)

    cv_scores = cross_val_score(knnr, xtrain_arr, ytrain_arr.ravel(), cv=k_folds, scoring="r2")
    cv_score_eval(cv_scores)
    print("V2 R2:", r2_score(ytrain_arr, Y2_pred))

    print("MAE:", mean_absolute_error(ytest_arr, Y_pred))
    print("MSE:", mean_squared_error(ytest_arr, Y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(ytest_arr, Y_pred)))
    print("R2 Score:", r2_score(ytest_arr, Y_pred))

def sv_regression(xtrain_arr, ytrain_arr, xtest_arr, ytest_arr, kernel_value, c_value, epsilon_value):
    k_folds = KFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=SELECTED_RANDOM_STATE)

    svr = SVR(kernel=kernel_value, C=c_value, epsilon=epsilon_value)

    xtrain_arr, xtest_arr = scale_x(xtrain_arr, xtest_arr)
    ytrain_arr, ytest_arr = scale_x(ytrain_arr, ytest_arr)

    cv_scores = cross_val_score(svr, xtrain_arr, ytrain_arr.ravel(), cv=k_folds, scoring="r2")
    cv_score_eval(cv_scores)

    svr.fit(xtrain_arr, ytrain_arr.ravel())

    Y_pred = svr.predict(xtest_arr)

    Y2_pred = svr.predict(xtrain_arr)
    print("V2 R2 Score:", r2_score(ytrain_arr, Y2_pred))

    print("MAE:", mean_absolute_error(ytest_arr, Y_pred))
    print("MSE:", mean_squared_error(ytest_arr, Y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(ytest_arr, Y_pred)))
    print("R2 Score:", r2_score(ytest_arr, Y_pred))

def mlp_fine_tuning(xtrain_arr, ytrain_arr, xtest_arr, ytest_arr):
    param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01],
    }

    grid_search_mlp = GridSearchCV(MLPRegressor(max_iter=2000, random_state=32), param_grid, cv=5, scoring='r2')
    grid_search_mlp.fit(xtrain_arr, ytrain_arr.ravel())

    print("Best parameters:", grid_search_mlp.best_params_)
    print("Best CV R2:", grid_search_mlp.best_score_)

def rf_fine_tuning(xtrain_arr, ytrain_arr, xtest_arr, ytest_arr):
    param_grid = {
        'max_depth': [6, 7, 8, 9, 10, 12, 15, None],
        'n_estimators': [50, 100, 150, 200, 150, 300],
        'max_features': ['sqrt', 'log2', None]
    }

    grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=32), param_grid, cv=5, scoring='r2')
    grid_search_rf.fit(xtrain_arr, ytrain_arr.ravel())

    print("Best parameters:", grid_search_rf.best_params_)
    print("Best CV R2:", grid_search_rf.best_score_)

X_train, X_test, Y_train, Y_test = train_test_split(training_array, labels_array, test_size=0.2, random_state=SELECTED_RANDOM_STATE)

print("---------------------------------------------------------------------------")
print("Random Forest Regression")
#rf_regressor(X_train, Y_train, X_test, Y_test, 7, "friedman_mse", "sqrt", SELECTED_RANDOM_STATE)

print("---------------------------------------------------------------------------")
print("MLP/Neural Network Regression")
mlp_regressor(X_train, Y_train, X_test, Y_test, 2000, (50, 50), "tanh", 0.0001, "adam", SELECTED_RANDOM_STATE)

print("---------------------------------------------------------------------------")
print("KNN Regression")
#knn_regressor(X_train, Y_train, X_test, Y_test, 18)

print("---------------------------------------------------------------------------")
print("SVM Regression")
#sv_regression(X_train, Y_train, X_test, Y_test, 'rbf', 10, 0.6)

print("---------------------------------------------------------------------------")
print("Fine Tune Testing")
#mlp_fine_tuning(X_train, Y_train, X_test, Y_test)
print("---------------------------------------------------------------------------")
#rf_fine_tuning(X_train, Y_train, X_test, Y_test)