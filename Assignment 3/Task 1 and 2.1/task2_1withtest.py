from pyspark import SparkContext
import math
import time
import sys

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
        return [(user,business),3.0]

    if business not in business_user_dictionary.keys():
        return [(user,business),3.0]

    weights = compute_weights(user, business, user_business_dictionary, business_user_dictionary,
                   user_business_rating_dictionary, business_avgrating_dictionary)

    if len(weights)==0:
        return [(user,business),3.0]

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

    return [(user,business),pred_r]

def preprocess_data(input_path):
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

    return distinct_businesses, user_business_dictionary, business_user_dictionary, user_business_rating_dictionary, business_avgrating_dictionary

def load_test_data(test_path):
    return (
        sc.textFile(test_path)
        .map(lambda l: l.split(","))
        .filter(lambda l: l[0] != "user_id")
    )

def calculate_rmse(test_path, output_data):
    test_data = load_test_data(test_path)
    true_ratings = test_data.map(lambda x: ((x[0], x[1]), float(x[2])))

    # Convert output_data list to RDD
    output_data_rdd = sc.parallelize(output_data)
    predicted_ratings = output_data_rdd.map(lambda x: ((x[0][0], x[0][1]), float(x[1])))

    joined_data = true_ratings.join(predicted_ratings)
    squared_errors = joined_data.map(
        lambda x: ((x[1][0] - x[1][1]) ** 2, 1)
    )
    sum_squared_errors = squared_errors.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    rmse = math.sqrt(sum_squared_errors[0] / sum_squared_errors[1])

    return rmse

def main(input_path, test_path, output_path,truthfile_path):
    distinct_businesses, user_business_dictionary, business_user_dictionary, user_business_rating_dictionary, business_avgrating_dictionary = preprocess_data(input_path)

    test_data = sc.textFile(test_path).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')
    output_data = test_data.map(lambda x: predict_rating(x[0],x[1],user_business_dictionary, business_user_dictionary,
                               user_business_rating_dictionary, business_avgrating_dictionary)).collect()

    ouput_file = open(output_path,"w")
    ouput_file.write("user_id, business_id, prediction\n")

    for row in output_data:
        ouput_file.write(row[0][0]+","+row[0][1]+","+str(row[1])+"\n")
    ouput_file.close()    

    rmse = calculate_rmse(truthfile_path, output_data)  

    print('Duration:',time.time()-start)
    print("RMSE:", rmse)
    
if __name__ == "__main__":

    input_path, test_path, output_path, truthfile_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    sc = SparkContext.getOrCreate()
    start=time.time()
    
    main(input_path, test_path, output_path,truthfile_path)


                                 