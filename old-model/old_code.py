'''
This was a 1st attempt at the problem. This version has been superseded by a newer, more redundant
version in the parent folder of this repo.
'''

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns
sns.set_theme()
import requests
from flair.data import Sentence
from flair.models import TextClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA   
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import re
import ast

SELECTED_RANDOM_STATE = 32
CROSS_VAL_FOLDS = 10


# SECTION 1: Data Collection
# V This section is used to convert API data into json then load into a csv file V
def process_reviews(rev_array):
    cleaned_rev = list()
    for review in rev_array:
        # Reviews from 2020 and 2021 were removed due to the P/F nature of courses for that period
        rev_yr = review['created'].split('-')[0]
        # Filter for reviews after 2017 to remove old reviews
        if((rev_yr not in ['2020','2021']) and (int(rev_yr) >= 2019)):
            cleaned_rev.append([review['course'],review['review'],review['expected_grade']])
    return cleaned_rev

# Process umd.io courses data and store into .csv file for easier future access
for k in range(1,790):
    response = requests.get("https://api.umd.io/v1/courses", params={"per_page":100, "sort":"course_id", "page":k, "semester":"201901|geq"})
    ctemp_df = pd.json_normalize(response.json())

    ctemp_df = ctemp_df.drop(labels=["name", "department", "description", "sections", 
                          "relationships.coreqs", "relationships.formerly", "relationships.restrictions",
                          "relationships.additional_info", "relationships.also_offered_as"], axis=1)
    
    ctemp_df = ctemp_df.sort_values(by=["course_id","semester"])
    ctemp_df = ctemp_df.drop_duplicates(subset=["course_id"], keep="last") # Keep latest ed of course
    # ctemp_df["gen_ed"] = ctemp_df["gen_ed"].apply(lambda x: len(x)) # Num of gen_eds subject is under
    
    print(f"Courses Set {k} cleaning complete")
    course_df = pd.concat([course_df, ctemp_df])

course_df.to_csv('course_data.csv', index=False)
print("Courses .csv uploaded.") 

# Process PlanetTerp Professor Data and store into .csv file for easier future access
for i in range(135):
    response = requests.get("https://planetterp.com/api/v1/professors",params={"reviews":'true', "limit":100, "offset":i*100})
    ptemp_df = pd.json_normalize(response.json())

    ptemp_df = ptemp_df[ptemp_df["type"] == "professor"] # Remove TAs
    ptemp_df = ptemp_df.drop(labels=["type"], axis=1) # Drop unnecessary columns

    # Converts reviews array into more efficient format, removing unnecessary fields
    ptemp_df["reviews"] = ptemp_df["reviews"].apply(lambda x:process_reviews(x))
    
    # Filter for rows with reviews and current classes only
    ptemp_df = ptemp_df[(ptemp_df['reviews'].map(len)>0) & (ptemp_df['courses'].map(len)>0)] 
   
    print(f"Professor Set {i} cleaning complete")
    prof_df = pd.concat([prof_df,ptemp_df])

prof_df.to_csv('prof_data.csv', index=False)
print("Professor .csv uploaded.")


# Process PlanetTerp courses data for average GPA from reviews and store into .csv file for easier future access
for h in range(157):
    response = requests.get("https://planetterp.com/api/v1/courses",params={"reviews":'false', "limit":100, "offset":h*100})
    ptc_temp_df = pd.json_normalize(response.json())
    
    ptc_temp_df = ptc_temp_df.drop(labels=["title", "description", "is_recent", "geneds"], axis=1)
    
    ptc_temp_df["professors"] = ptc_temp_df["professors"].apply(lambda x: tuple(set(x))) # Eliminate duplicates from multiple semesters
    ptc_temp_df["average_gpa"] = ptc_temp_df["average_gpa"].fillna(0)

    print(f"Courses v2 Set {h} cleaning complete")
    pt_course_df = pd.concat([pt_course_df, ptc_temp_df])

pt_course_df.to_csv('PTcourse_data.csv', index=False)
print("Planet Terp Courses .csv uploaded.")

# ----------------------------------------------------------------------------------------------------------------------------------------

# SECTION 2: Data Cleaning

tagger = TextClassifier.load("sentiment-fast")

def generate_score(review):
    # Function generates sentiment analysis score for review
    parts = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", review) if s.strip()] # Split review into sentences
    rev_len = 1 / len(review)
    comment_score = 0

    sentences = [Sentence(i) for i in parts]
    tagger.predict(sentences, mini_batch_size=4) 
    # Analysis done in batches due to time and memory constraints
    
    for section in sentences:
        sentiment = -1 if section.labels[0].value == "NEGATIVE" else 1
        comment_score += (len(section.text) * rev_len * section.score * sentiment)

    return comment_score

def profreview_data(prof_data, course_data):
    # Function that extracts reviews data and returns a df for each review
    course_data = course_data.set_index('course_id')
    review_data = pd.DataFrame()
    data_rows = []

    for prof in prof_data.itertuples():
        for review in prof[2]:
            if (review[0] == None) or (review[0] not in course_data.index):
                school_value = "PROF"
                if (review[0] == None):
                    school_value = "PROF"
                else:
                    school_value = "OTHER"
                avg_gpa = 0
                course_lvl = 0
                credits = 0
            else:
                target_row = course_data.loc[review[0]]
                school_value = target_row["school"]
                avg_gpa = target_row["average_gpa"]
                course_lvl = target_row["course_lvl_bin"]
                credits = target_row["credits"]

            new_row = {'professor_id': prof[4], 
                        "review_score": review[1],
                        "expected_grade": review[2],
                        "school": school_value,
                        "course_avgGPA": avg_gpa,
                        "course_lvl": course_lvl,
                        "credits": credits
                        }
            data_rows.append(new_row)

    review_data = pd.DataFrame(data_rows)

    print(len(review_data))
    review_data["expected_grade"] = review_data["expected_grade"].map(convert_grade)
    review_data["review_score"] = review_data["review_score"].map(generate_score)
    return review_data

review_df = profreview_data(prof_df, course_df)
review_df.to_csv('review_data.csv', index=True)


def convert_grade(grade):
    # Convert expected grade into respective GPA points
    grade_dict = {"A+": 4, "A": 4, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7, 
                  "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "D-": 0.7}
    if grade in grade_dict.keys():
        return float(grade_dict[grade])
    else:
        return 0

school_map = {
    "AGNR": ["AGNR", "AGST", "ANSC", "AREC", "ENST", "ENTM", "NFSC", "PLSC", "INAG"],
    "ARCH": ["ARCH", "RDEV", "HISP", "URSP", "CPSS"],
    "ARHU": ["AAAS", "AAST", "CINE", "ARAB", "ARTH", "ARTT", "CLAS", "CHIN", "CMLT", "COMM","DANC", "ENGL", "FREN",
        "GERM", "GERS", "GREK", "HEBR", "IMMR", "ITAL", "JAPN", "JWST", "KORA", "LATN", "LGBT", "LING", 
        "MUED", "MUSC", "MUSP", "PHIL", "PORT", "RELS", "RUSS", "SPAN", "THET", "WGSS", "WMST", "SLLC", "TDPS"],
    "BSOS": ["AASP", "ANTH", "CCJS", "ECON", "GEOG", "GVPT", "PSYC", "SOCY", "BSST", "LACS", "USLT", "SLAA", "SURV", "CPBE", "CPSN"],
    "BMGT": ["BMGT", "BUAC", "BUDT", "BUFN", "BULM", "BUMK", "BUMO", "BUSI", "BUSM", "BUSO", "EMBA", "BDBA", "BMSO", "BMIN"],
    "CMNS": ["AMSC", "ASTR", "BCHM", "BIOL", "BSCI", "CBMG", "CHEM", "CHPH", "CLFS", "CMSC", "GEOL", "MATH", "PHYS", 
             "STAT", "AOSC", "BIOE", "BIOI", "BIOM", "BIPH", "BISI", "NACS", "NEUR", "MOCB", "MSML", "MSQC", "CPMS", "CPSF"],
    "EDUC": ["EDCP", "EDHD", "EDHI", "EDMS", "EDPS", "EDSP", "TLPL", "TLTC", "CHSE"],
    "ENGR": ["ENAE", "ENBC", "ENCE", "ENCH", "ENCO", "ENEB", "ENED", "ENEE", "ENES", "ENFP", "ENMA", "ENME", "ENMT", "ENNU", "ENPM", "ENRE", "ENSE", "MEES"],
    "INFO": ["INST", "INFM", "DATA", "INFO", "CPDJ"],
    "JOUR": ["JOUR"],
    "SPP": ["PLCY"],
    "SPH": ["HLTH", "HLSA", "HLSC", "MIEH", "EPIB", "FMSC", "PHSC", "SPHL"],
    "UGST": ["UNIV", "HONR", "HNUH", "HDCC", "HHUM", "HGLO", "HESI", "LEAD", "FIRE", "IDEA", "MLAW", "CPCV", "CPET", 
             "CPGH", "CPJT", "CPPL", "CPSA", "CPSD", "CPSG", "CPSP", "CRLN", "GEMS", "HBUS", "SMLP", "VIPS", "WEID"],
    "OTHER": ["ARMY", "NAVY", "OURS", "PEER", "XPER", "ABRM", "BSCV", "BSGC", "FGSM", "HACS", "HEIP",
        "IMDM", "ISRL", "MAIT", "MITH", "MLSC", "NIAP", "NIAS", "NIAV", "PHPE", "SMLP", "UMEI", "SUMM", "EXST"]
}

def assign_schl(code):
    for school in school_map:
        if code in school_map[school]:
            return school
    return "OTHER"

def bin_course(course_id):
    # Course binning function
    course_id = course_id.strip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if course_id[0] == "0":
        # No degree courses
        return 0
    elif int(course_id[0]) in range(1,5):
        # UG courses
        return 1 
    elif int(course_id[0]) in range(5,9):
        # Grad courses excl. doctoral courses
        return 2
    elif course_id[0] == "9":
        return 3

def create_courselist(reviews_arr):
    # Create list of courses taught for each professor
    courses = []
    for review in reviews_arr:
        if (review[0] not in courses) and (review[0] != None):
            courses.append(review[0])
    
    return courses

def convert_grademethod(grading_array):
    # Find grading methods for array of reviews
    if "Regular" in grading_array:
        return 2
    elif ("Pass-Fail" in grading_array) or ("Sat-Fail" in grading_array):
        return 1
    else:
        return 0

def clean_coursedf(df):
    # Data cleaning for umd.io course df
    df["grading_method"] = df["grading_method"].apply(ast.literal_eval)
    df["gen_ed"] = df["gen_ed"].apply(ast.literal_eval)
    df["gen_ed"] = df["gen_ed"].apply(lambda x: len(x) if len(x) > 0 else 0)
    df["core"] = df["core"].apply(ast.literal_eval)
    df["core"] = df["core"].apply(lambda x: len(x) if len(x) > 0 else 0)
    df["grading_method"] = df["grading_method"].apply(convert_grademethod)
    df = df.sort_values(by=["course_id","semester"])
    df = df.drop_duplicates(subset=["course_id"], keep="last")
    df = df.drop(labels=["semester", "dept_id"], axis=1)

    return df

def clean_profdf(df):
    # Data cleaning for prof df
    df["reviews"] = df["reviews"].apply(ast.literal_eval)
    df["courses"] = df["reviews"].apply(lambda x: create_courselist(x))

    return df

def clean_ptcoursedf(df):
    # Data cleaning for PlanetTerp course df
    df["course_id"] = df["name"]
    df = df.drop(labels=["course_number", "credits", "name", "professors"], axis=1)

    return df

def clean_mergedcourse(df):
    df["school"] = (df["course_id"].apply(lambda x: x[0:4])).apply(lambda x: assign_schl(x))
    df["course_lvl_bin"] = df["course_id"].apply(lambda x: bin_course(x))
    df["course_lvl"] = df["course_id"].apply(lambda x: x.strip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[0])

    return df

def classify_reviews(review_score):
    # Classify review scores into 1 of 3 categories after applying Flair
    if(review_score > 0.4):
        return "POSITIVE"
    elif (review_score < -0.4):
        return "NEGATIVE"
    else:
        return "NEUTRAL"

course_df = pd.read_csv('course_data.csv')
course_df = clean_coursedf(course_df)

prof_df = pd.read_csv('prof_data.csv')
prof_df = clean_profdf(prof_df)

pt_course_df = pd.read_csv('PTcourse_data.csv')
pt_course_df = clean_ptcoursedf(pt_course_df)

course_df = pd.merge(course_df, pt_course_df, on="course_id", how="left")
course_df = clean_mergedcourse(course_df)
school_avg = course_df.dropna(subset=["average_gpa"]).groupby("school")["average_gpa"].mean().sort_values().to_dict()

course_df["average_gpa"] = course_df["average_gpa"].fillna(course_df["school"].map(school_avg))

review_df = pd.read_csv("review_data.csv")
review_df["rating"] = review_df["review_score"].map(classify_reviews)
review_df["difference"] = review_df["expected_grade"] - review_df["course_avgGPA"]


review2_df = review_df.groupby(["professor_id", "rating"]).size().unstack(fill_value=0)
school_count = review_df.groupby("professor_id")["school"].value_counts()
school_count = school_count.groupby("professor_id").idxmax()
review2_df["school"] = school_count.apply(lambda x: x[1])

# Quick analysis on data to find relevant features
avg_exp_grade = review_df.groupby("professor_id")["expected_grade"].mean()
avg_courseGPA = review_df.groupby("professor_id")["course_avgGPA"].mean()
avg_courselvl = review_df.groupby("professor_id")["course_lvl"].mean()
avg_credits = review_df.groupby("professor_id")["credits"].mean()
# pivot1 = pd.crosstab(review_df["review_score"], review_df["expected_grade"])
# print(stats.chi2_contingency(pivot1).pvalue)
# pivot2 = pd.crosstab(review_df["review_score"], review_df["course_avgGPA"])
# print(stats.chi2_contingency(pivot2).pvalue)
# pivot3 = pd.crosstab(review_df["review_score"], review_df["school"])
# print(stats.chi2_contingency(pivot3).pvalue)
# pivot4 = pd.crosstab(review_df["review_score"], review_df["credits"])
# print(stats.chi2_contingency(pivot4).pvalue)
# pivot5 = pd.crosstab(review_df["review_score"], review_df["expected_grade"] - review_df["course_avgGPA"])
# print(stats.chi2_contingency(pivot5).pvalue)

# Merging datasets before training
review2_df = review2_df.merge(avg_exp_grade.rename("avg_expected_grade"), on="professor_id")
review2_df = review2_df.merge(avg_courseGPA.rename("avg_course_GPA"), on="professor_id")
review2_df = review2_df.merge(avg_courselvl.rename("avg_course_lvl"), on="professor_id")
review2_df = review2_df.merge(avg_credits.rename("avg_credits"), on="professor_id")
prof_df["professor_id"] = prof_df["slug"]

training_df = prof_df.merge(review2_df, on="professor_id")
training_df = training_df.set_index("professor_id")

training_df = pd.get_dummies(training_df, columns=["school"])
labels = training_df["average_rating"]
training_df = training_df.drop(columns=["average_rating"], axis=1)

training_df.to_csv("data_X.csv")
labels.to_csv("data_Y.csv")

# ----------------------------------------------------------------------------------------------------------------------------------------
# Section 3: Model Training and Evaluation

features_df = pd.read_csv("data_X.csv")
labels_df = pd.read_csv("data_Y.csv")

features_df = features_df.drop(columns=["name", "slug", "courses", "reviews", "school_PROF"], axis=1)

training_array = features_df.drop(["professor_id", "avg_course_GPA"], axis=1).to_numpy()
labels_array = labels_df.drop(["professor_id"], axis=1).to_numpy()

pca = PCA(n_components=0.99)
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