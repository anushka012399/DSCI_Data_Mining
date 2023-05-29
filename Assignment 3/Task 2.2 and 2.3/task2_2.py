import time
from pyspark import SparkContext
import xgboost as xgb
import json
import sys

# Function to read input files
def read_input_files(folder_path: str):
    # Paths to input files
    input_path = folder_path + "/yelp_train.csv"
    user_file = folder_path + "/user.json"
    business_file = folder_path + "/business.json"

    # Read Yelp train data
    data_rdd = sc.textFile(input_path).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')
    # Map user, business, rating to ((user, business), rating)
    user_business_rating = data_rdd.map(lambda s: ((s[0], s[1]), float(s[2])))

    # Read validation data
    validation_data = sc.textFile(test_file_name).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')
    # Map user, business to (user, business)
    user_business_validation_data = validation_data.map(lambda s: (s[0], s[1]))
    # Collect the validation data as a list
    user_business_validation_data = user_business_validation_data.collect()

    # Read user data and map user_id to (review_count, average_stars)
    user_data = sc.textFile(user_file).map(lambda r: json.loads(r))
    features_of_users = user_data.map(lambda x: (x['user_id'], (float(x['review_count']), float(x['average_stars'])))).collectAsMap()

    # Read business data and map business_id to (review_count, stars)
    business_data = sc.textFile(business_file).map(lambda r: json.loads(r))
    features_of_business = business_data.map(lambda x: (x['business_id'], (float(x['review_count']), float(x['stars'])))).collectAsMap()

    return user_business_rating, user_business_validation_data, features_of_users, features_of_business

# Function to create train data from user and business features
def create_train(user: str, business: str, features_of_users: dict, features_of_business: dict):
    # If user or business not in features, return 0s
    if user not in features_of_users or business not in features_of_business:
        return [0, 0, 0, 0]
    # Get review count and stars data for user and business
    review_count_user_data = features_of_users[user][0]
    review_count_business_data = features_of_business[business][0]
    average_star_user_data = features_of_users[user][1]
    stars_business_data = features_of_business[business][1]
    # Return a list of features
    return [review_count_user_data, review_count_business_data, average_star_user_data, stars_business_data]

# Function to write output file
def write_output_file(output_path: str, user_business_validation_data: list, predicted_rating: list):
    with open(output_path, "w") as f:
        f.write("user_id, business_id, prediction\n")
        for j in range(len(user_business_validation_data)):
            f.write(f"{user_business_validation_data[j][0]},{user_business_validation_data[j][1]},{predicted_rating[j]}\n")

if __name__ == '__main__':
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    # Create Spark context
    sc = SparkContext.getOrCreate()
    # Set log level to Error
    sc.setLogLevel('Error')

    # Start timer
    start = time.time()

    # Read input files
    user_business_rating, user_business_validation_data, features_of_users, features_of_business = read_input_files(folder_path)

    # Create train data
    x_train = user_business_rating.map(lambda x: create_train(x[0][0], x[0][1], features_of_users, features_of_business)).collect()
    y_train = user_business_rating.map(lambda x: x[1]).collect()

    # Create test data
    x_test = sc.parallelize(user_business_validation_data).map(lambda x: create_train(x[0], x[1], features_of_users, features_of_business)).collect()

    # Print out user, business pairs in validation data that are not in the features
    for user_business in user_business_validation_data:
        if user_business[0] not in features_of_users.keys() or user_business[1] not in features_of_business.keys():
            print(user_business)

    # Create XGBoost model and fit to train data
    model_xgboost = xgb.XGBRegressor(objective='reg:linear', n_estimators=100, max_depth=5, n_jobs=-1)
    model_xgboost.fit(x_train, y_train)

    # Use model to predict ratings for test data
    predicted_rating = model_xgboost.predict(x_test)

    # Write output file
    write_output_file(output_file_name, user_business_validation_data, predicted_rating)

    # End timer and print out duration
    end = time.time()
    print('Duration:', end-start)