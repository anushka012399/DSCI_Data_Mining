import xgboost as xgb
import numpy as np
import json
import time
import csv
import json
import os
import sys
import math
from itertools import combinations
from pyspark import SparkConf, SparkContext
from operator import add

#Method Description:
#This code is a combination of collaborative filtering and XGBoost for predicting user ratings for businesses in a recommendation system. It utilizes PySpark for data processing and XGBoost for training and predicting.
#The main part of the code starts by initializing the Spark context, loading the training and test data, and creating user-business dictionaries. 
#Then, it computes rating averages for both users and businesses. The test data is processed, and collaborative filtering predictions are generated using the calculate_final_prediction() function.
#Next, the code loads and processes user, business, and review features. The training and test datasets are created using these features. 
#The XGBoost model is trained with the best parameters obtained through hyperparameter tuning (using Optuna and 5-fold cross-validation). The XGBoost model is then used to predict ratings on the test dataset.
#Finally, the RMSE and error distribution for the XGBoost predictions are calculated using the compute_rmse_and_error_distribution() function.
#The features used in this code for improving the RMSE (Root Mean Square Error) of the prediction model are a combination of user, business, and review-related features.
#For user-related features, the following features are considered: useful, compliment_hot, fans, review_count, average_stars, compliment_funny, compliment_more, compliment_cool, compliment_profile, compliment_note, compliment_cute, compliment_list, compliment_plain, compliment_writer, and compliment_photos.
#For business-related features, the following features are considered: review_count, stars, is_open, latitude, longitude, price range, acceptance of credit cards, bike parking, outdoor seating, restaurants good for groups, restaurants delivery, caters, hasTV, restaurants reservations, restaurants table service, outdoor seating, by appointment only, restaurants takeout, accepts insurance, wheelchair accessible, and good for kids.
#For review-related features, the following features are considered: stars, useful, funny, and cool.
#Using these features in combination with the collaborative filtering technique helped in improving the RMSE of the prediction model. Additionally, hyperparameter tuning using Optuna with 5-fold cross-validation was used to fine-tune the XGBoost model's hyperparameters to obtain better results.
#Overall, these features' combination helped in better capturing the user's preferences, the business's characteristics, and the reviews' sentiment, thus improving the prediction 

#Error Distribution
#>=0 and <1: 102633
#>=1 and <2: 32447
#>=2 and <3: 6114
#>=3 and <4: 849
#>=4 and <inf: 1

#RMSE: 
#0.9763456442333556

#Execution Time:
#1066.1266582012177s

def create_user_and_business_dictionary(data):

    user = data.map(lambda x: ((x[0]), ((x[1]), float(x[2])))).groupByKey().sortByKey()
    user_rating = user.mapValues(dict).collectAsMap()

    business = data.map(lambda x: ((x[1]), ((x[0]), float(x[2])))).groupByKey().sortByKey()
    business_rating = business.mapValues(dict).collectAsMap()

    return user_rating,business_rating

def pearson(other, user, business, business_avg):
    business_ratings = []
    other_ratings = []
    other_business_ratings = business_rating.get(other)
    other_avg_rating = business_rating_average.get(other)

    for user_id in user:
        if other_business_ratings.get(user_id) == True:
            business_score = business.get(user_id)
            other_score = other_business_ratings.get(user_id)
            business_ratings.append(business_score)
            other_ratings.append(other_score)

    if len(business_ratings) > 0:
        numerator = 0
        business_denominator = 0
        other_denominator = 0

        for i in range(len(business_ratings)):
            diff_business_rating = business_ratings[i] - business_avg
            diff_other_rating = other_ratings[i] - other_avg_rating

            numerator += diff_business_rating * diff_other_rating
            business_denominator += diff_business_rating ** 2
            other_denominator += diff_other_rating ** 2

        overall_denominator = (business_denominator * other_denominator) ** 0.5

        if numerator != 0 and overall_denominator != 0:
            pearson_coef = numerator / overall_denominator
        elif numerator == 0 and overall_denominator != 0:
            pearson_coef = 0
        elif numerator == 0 and overall_denominator == 0:
            pearson_coef = 1
        else:
            pearson_coef = -1
    else:
        pearson_coef = float(business_avg / other_avg_rating)

    return pearson_coef


def calculate_final_prediction(rdd):
    user_id, business_id = rdd[0], rdd[1]

    if business_id in business_rating:
        avg_business_rating = business_rating_average.get(business_id)
        users = list(business_rating.get(business_id))
        user_single_business_ratings = business_rating.get(business_id)

        if user_rating.get(user_id) is not None:
            user_rating_info_list = list(user_rating.get(user_id))

            if len(user_rating_info_list) > 0:
                pearson_coefficients = []

                for other_user in user_rating_info_list:
                    current_neighbor_score = business_rating.get(other_user).get(user_id)
                    pearson_coefficient = pearson(other_user, users, user_single_business_ratings, avg_business_rating)

                    if pearson_coefficient > 0:
                        if pearson_coefficient > 1:
                            pearson_coefficient = 1 / pearson_coefficient
                        pearson_coefficients.append((pearson_coefficient, current_neighbor_score))

                numerator = 0
                denominator = 0
                sorted_coefficients = sorted(pearson_coefficients, key=lambda x: -x[0])

                for i in range(len(sorted_coefficients)):
                    coef_value = sorted_coefficients[i][0]
                    coef_weight = sorted_coefficients[i][1]
                    numerator += coef_value * coef_weight
                    denominator += abs(coef_value)

                prediction_value = numerator / denominator
                return prediction_value
            else:
                return avg_business_rating
        else:
            return avg_business_rating
    else:
        return str(user_rating_average.get(user_id))

def preprocess_data(data, user_data, business_data, review_data, is_test):
    user_id, business_id = data[0], data[1]
    rating = -1 if is_test else data[2]

    if user_id in user_data and business_id in business_data and user_id in review_data:
        user_features = user_data[user_id]
        business_features = business_data[business_id]
        review_features = review_data[user_id]

        combined_features = list(user_features) + list(business_features) + list(review_features) + [float(rating)]
        combined_features = [float(feature) for feature in combined_features]

        return [user_id, business_id] + combined_features
    else:
        return [user_id, business_id] + [None] * (len(user_data[user_id]) + len(business_data[business_id]) + len(review_data[user_id]) + 1)

def create_dataset(data, is_test):
    preprocessed_data = data.map(lambda record: preprocess_data(record, user_features, business_features, review_train_features, is_test)).collect()
    preprocessed_data_array = np.array(preprocessed_data)
    feature_count = len(preprocessed_data_array[0])
    x = np.array(preprocessed_data_array[:, 2:feature_count - 1], dtype='float')
    y = np.array(preprocessed_data_array[:, -1], dtype='float')
    return x, y

def write_csv(path, write_data):
    with open(path, mode='w', newline='') as file:
        csv_writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['user_id', ' business_id', ' prediction'])
        
        for record in write_data:
            csv_writer.writerow([str(record[0]), str(record[1]), float(record[2])])

def load_training_data(data_path):
    data = sc.textFile(data_path)
    header = data.first()
    data = data.filter(lambda i: i != header).map(lambda i: i.split(","))
    return data

def compute_rating_avg(rating_average):
    rating_average = rating_average.mapValues(lambda i:i[0]/i[1])
    rating_average = {i:j for i, j in rating_average.collect()}
    return rating_average

def load_from_different_data_files():
    user_path = folder_path + '/user.json'
    user_data = sc.textFile(user_path)

    business_path = folder_path + '/business.json'
    business_data = sc.textFile(business_path)

    review_train_path = folder_path + '/review_train.json'
    review_train_data = sc.textFile(review_train_path)

    return user_data,business_data,review_train_data

def get_business_features():
     
    def extract_price_range(attributes, key):
        if attributes:
            if key in attributes:
                return int(attributes[key])
        return 0

    def convert_true_false_to_binary(attributes,key):
        if attributes:
            if key in attributes:
                return 1 if attributes[key] == 'True' else 0
        return 0
    
    business_features = (
        business_data
        .map(json.loads)
        .map(lambda i: (
            i["business_id"],
            (
                i["review_count"], i["stars"], i['is_open'], i['latitude'], i['longitude'],
                extract_price_range(i['attributes'], 'RestaurantsPriceRange2'),
                convert_true_false_to_binary(i['attributes'], 'BusinessAcceptsCreditCards'),
                convert_true_false_to_binary(i['attributes'], 'BikeParking'),
                convert_true_false_to_binary(i['attributes'], 'OutdoorSeating'),
                convert_true_false_to_binary(i['attributes'], 'RestaurantsGoodForGroups'),
                convert_true_false_to_binary(i['attributes'], 'RestaurantsDelivery'),
                convert_true_false_to_binary(i['attributes'], 'Caters'),
                convert_true_false_to_binary(i['attributes'], 'HasTV'),
                convert_true_false_to_binary(i['attributes'], 'RestaurantsReservations'),
                convert_true_false_to_binary(i['attributes'], 'RestaurantsTableService'),
                convert_true_false_to_binary(i['attributes'], 'OutdoorSeating'),
                convert_true_false_to_binary(i['attributes'], 'ByAppointmentOnly'),
                convert_true_false_to_binary(i['attributes'], 'RestaurantsTakeOut'),
                convert_true_false_to_binary(i['attributes'], 'AcceptsInsurance'),
                convert_true_false_to_binary(i['attributes'], 'WheelchairAccessible'),
                convert_true_false_to_binary(i['attributes'], 'GoodForKids')
            )
        ))
        .collectAsMap()
        )
    return business_features

#Performed hyperparameter tuning using Optuna with 5-fold cross validation

# # Define the objective function for Optuna
# def objective(trial):
#     # Define the hyperparameter search space
#     params = {
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 1),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
#     }
    
#     xgb_model = xgb.XGBRegressor(**params, random_state=42)

#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     rmse_scores = -1 * cross_val_score(xgb_model, x_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
    
#     return rmse_scores.mean()

def compute_rmse_and_error_distribution(test_file, output_file):
    test_data = sc.textFile(test_file)
    header_test = test_data.first()
    valid_test = test_data.filter(lambda row: row != header_test).map(lambda row: row.split(",")).sortBy(lambda row: (row[0], row[1])).persist()

    output_data = sc.textFile(output_file)
    header_output = output_data.first()
    output_rows = output_data.filter(lambda row: row != header_output).map(lambda row: row.split(','))
    output_pairs = output_rows.map(lambda row: ((row[0], row[1]), float(row[2])))
    test_pairs = valid_test.map(lambda row: ((row[0], row[1]), float(row[2])))

    differences = test_pairs.join(output_pairs).map(lambda row: abs(row[1][0] - row[1][1]))
    mse = differences.map(lambda diff: diff**2).reduce(lambda x, y: x + y)
    rmse = math.sqrt(mse / output_pairs.count())

    print('Error Distributions:')
    distribution_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, float('inf'))]
    for lower_bound, upper_bound in distribution_ranges:
        count = differences.filter(lambda diff: lower_bound <= diff < upper_bound).count()
        print(f'>={lower_bound} and <{upper_bound}: {count}')

    print(f'RMSE: {rmse}')

if __name__ == '__main__':

    start_time = time.time()
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    conf = SparkConf().setAppName("553Competition")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    training_data_path = folder_path + '/yelp_train.csv'
    training_data = load_training_data(training_data_path)

    test_data = load_training_data(test_file)
    test_data_sorted = test_data.sortBy(lambda i:((i[0]),(i[1]))).persist()
    
    user_rating,business_rating = create_user_and_business_dictionary(training_data)

    user_rating_average = training_data.map(lambda i: (i[0], float(i[2]))).combineByKey(lambda rating: (rating, 1), lambda acc, rating: (acc[0] + rating, acc[1] + 1), lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]))
    user_rating_average = compute_rating_avg(user_rating_average)

    business_rating_average =training_data.map(lambda i: (i[1], float(i[2]))).combineByKey(lambda rating: (rating, 1), lambda acc, rating: (acc[0] + rating, acc[1] + 1), lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]))
    business_rating_average = compute_rating_avg(business_rating_average)

    resulting_predictions = test_data_sorted.map(calculate_final_prediction).collect()
    collaborative_filtering_results = np.asarray(resulting_predictions,dtype='float')

    user_data,business_data,review_train_data=load_from_different_data_files()

    user_features = (
        user_data
        .map(json.loads)
        .map(lambda i: (
            i["user_id"], 
            (
                i["useful"], i['compliment_hot'], i['fans'], i["review_count"], i["average_stars"],
                i['compliment_funny'], i['compliment_more'], i['compliment_cool'], i['compliment_profile'],
                i['compliment_note'], i['compliment_cute'], i['compliment_list'], i['compliment_plain'],
                i['compliment_writer'], i['compliment_photos']
            )
        ))
        .collectAsMap()
        )

    business_features = get_business_features()

    review_train_features = (
        review_train_data
        .map(json.loads)
        .map(lambda i: (
            i["user_id"], 
            (
                i["stars"], i["useful"], i['funny'], i['cool']
            )
        ))
        .collectAsMap()
    )

    x_train, y_train = create_dataset(training_data, False)
    x_test, y_test = create_dataset(test_data, True)

    test_dataset = test_data.map(lambda record: preprocess_data(record, user_features, business_features, review_train_features, True)).collect()

    # #Create an Optuna study and run optimization
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=50)
    ## Get the best parameters
    # best_params = study.best_params
    # print(best_params)

    #The code for hyperparameter tuning has been commented out due to time constraints and the parameters used below are the best extracted parameters from the optimization study across all the 50 trials 
    best_params = {'learning_rate': 0.01962697840799163, 'n_estimators': 867, 'max_depth': 9, 'min_child_weight': 9, 'subsample': 0.7292260459330953, 'colsample_bytree': 0.6608330637967573, 'gamma': 0.25715761554791394, 'reg_alpha': 0.826253233861312, 'reg_lambda': 0.4333607603784695}

    xgb_model = xgb.XGBRegressor(**best_params)
    xgb_model.fit(x_train,y_train)

    xgb_model_results = xgb_model.predict(x_test)

    combined_results = 0.99999*xgb_model_results + (1-0.99999)*collaborative_filtering_results
    test_dataset_array = np.array(test_dataset)
    final_results_combined = np.c_[test_dataset_array[:, :2], combined_results]

    write_csv(output_file,final_results_combined)

    end_time = time.time()-start_time

    compute_rmse_and_error_distribution(test_file, output_file)

    print("Execution Time: "+str(end_time))
    