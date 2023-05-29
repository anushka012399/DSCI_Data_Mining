import time
import sys
from pyspark import SparkContext
import math
import xgboost as xgb
import json

def compute_weights(user, business, user_business_dictionary, business_user_dictionary,
                   user_business_rating_dictionary, business_avgrating_dictionary):
    weights = []
    rated_businesses = user_business_dictionary[user]
    for b in rated_businesses:
        if b!=business:
            common_users = list(set(business_user_dictionary[b]).intersection(set(business_user_dictionary[business])))

            numerator = 0
            denominator_1 = 0
            denominator_2 = 0

            if len(common_users)>1:
                for u in common_users:
                    numerator += (user_business_rating_dictionary[(u,b)]-business_avgrating_dictionary[b])*(user_business_rating_dictionary[(u,business)]-business_avgrating_dictionary[business])
                    denominator_1 += (user_business_rating_dictionary[(u,b)]-business_avgrating_dictionary[b])**2
                    denominator_2 += (user_business_rating_dictionary[(u,business)]-business_avgrating_dictionary[business])**2

                denominator = math.sqrt(denominator_1)*math.sqrt(denominator_2)

                if numerator==0 or denominator==0:
                    weights.append([0,0])
                elif numerator<0 or denominator<0:
                    continue
                else:
                    weight = numerator/denominator
                    weights.append([weight*user_business_rating_dictionary[(user,b)],weight])

            else:
                difference = abs(business_avgrating_dictionary[b]-business_avgrating_dictionary[business])
                if 0<=difference<=1:
                    weights.append([user_business_rating_dictionary[(user,b)],1])
                elif 1<difference<=2:
                    weights.append([0.5*user_business_rating_dictionary[(user,b)],0.5])
                else:
                    weights.append([0,0])

    return weights

def predict_rating(user, business, user_business_dictionary, business_user_dictionary,
                   user_business_rating_dictionary, business_avgrating_dictionary):
    if user not in user_business_dictionary.keys():
        return 3.0

    if business not in business_user_dictionary.keys():
        return 3.0

    weights = compute_weights(user, business, user_business_dictionary, business_user_dictionary,
                   user_business_rating_dictionary, business_avgrating_dictionary)

    if len(weights)==0:
        return 3.0

    weights.sort(key=lambda x: -x[1])

    numerator=0
    denominator=0
    for vals in weights[:100]:
        numerator+= vals[0]
        denominator+= abs(vals[1])

    pred_r = 0.0
    if numerator==0 or denominator==0:
        pred_r = 0.0
    else:
        pred_r = numerator/denominator

    return pred_r

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
    validation_data = sc.textFile(validation_file_name).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')
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


if __name__ == '__main__':

    start=time.time()
    sc = SparkContext.getOrCreate()

    folder_path = sys.argv[1]
    validation_file_name = sys.argv[2]
    ouput_file_name = sys.argv[3]

    input_path = folder_path + "/yelp_train.csv"
    userfile = folder_path + "/user.json"
    bsnessfile = folder_path + "/business.json"

    #model based

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
    predicted_rating_model_based = model_xgboost.predict(x_test)

    #item based
    train_rdd = sc.textFile(input_path).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')
    user_business_rating = train_rdd.map(lambda s: ((s[0],s[1]),float(s[2])))
    business_average_rating = train_rdd.map(lambda s:(s[1],float(s[2]))).groupByKey().map(lambda x: (x[0],sum(x[1])/len(x[1])))

    distinct_businesses = train_rdd.map(lambda s: s[1]).distinct().collect()
    user_business_pairs = train_rdd.map(lambda s: (s[0],s[1])).groupByKey().mapValues(list)
    business_user_pairs = train_rdd.map(lambda s: (s[1],s[0])).groupByKey().mapValues(list)

    business_user_dictionary = business_user_pairs.collectAsMap()
    user_business_dictionary = user_business_pairs.collectAsMap()
    user_business_rating_dictionary = user_business_rating.collectAsMap()
    business_avgrating_dictionary = business_average_rating.collectAsMap()

    validation_data_item_based = sc.textFile(validation_file_name).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')
    predicted_rating_item_based = validation_data_item_based.map(lambda x: predict_rating(x[0],x[1],user_business_dictionary, business_user_dictionary,
                               user_business_rating_dictionary, business_avgrating_dictionary)).collect()

    no_of_predictions = len(predicted_rating_model_based)

    #Calculating hybrid rating
    predicted_rating_hybrid = []
    for i in range(no_of_predictions):
        y = (0.8 * predicted_rating_model_based[i]) + (0.2 * predicted_rating_item_based[i])
        predicted_rating_hybrid.append(y)

    writer = open(ouput_file_name,"w")
    writer.write("user_id, business_id, prediction\n")

    for index in range(len(user_business_validation_data)):
        writer.write(user_business_validation_data[index][0]+","+user_business_validation_data[index][1]+","+str(predicted_rating_hybrid[index])+"\n")
    writer.close() 

    end=time.time()
    print('Duration:',end-start)