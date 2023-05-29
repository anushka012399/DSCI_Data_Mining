from pyspark import SparkContext
from itertools import combinations
import collections
import time
import sys
import csv

def preprocessing(input_file_path):

    data_rdd = sc.textFile(input_file_path)
    header = data_rdd.first()

    customer_product_rdd = data_rdd.filter(lambda x : x != header).map(lambda x : (x.split(',')[0], x.split(',')[1], x.split(',')[5])).map(lambda pair : (pair[0][1:-1]+'-'+str(int(pair[1][1:-1])), int(pair[2][1:-1])))

    intermediate_file_path = 'customer_product.csv'

    with open(intermediate_file_path, 'w') as output:
        writer = csv.writer(output, quoting=csv.QUOTE_NONE)
        writer.writerow(["DATE-CUSTOMER_ID", "PRODUCT_ID"])
        for row in customer_product_rdd.collect():
            writer.writerow(row)
        output.close()

    return intermediate_file_path

def son(baskets_rdd,support_num):

    data_size = baskets_rdd.count()
    candidates_subsets_rdd = baskets_rdd.mapPartitions(lambda subset : generate_candidates(partitioned_data=subset, support=support_num, size=data_size))
    final_candidates = candidates_subsets_rdd.flatMap(lambda element : element).distinct().collect()
    final_frequent_itemsets = baskets_rdd.flatMap(lambda subset : itemsets_finalcount(partitioned_data=subset, candidate_list=final_candidates)).reduceByKey(lambda x,y : x+y).filter(lambda candidate_count: candidate_count[1] >= support_num).map(lambda x: x[0]).collect()

    with open(output_file_path, 'w') as output:
        output.write('Candidates:\n')
        for value in required_form(final_candidates).values():
            output.write(value[1:])
        output.write('Frequent Itemsets:\n')
        for value in required_form(final_frequent_itemsets).values():
            output.write(value[1:])
        output.close()

    return

def required_form(given_list):
    normalized_dict = collections.defaultdict(list)
    for element in given_list:
        if type(element) == str:
            element = tuple(element.split(','))
            element = str(str(element)[:-2] + ')')
            normalized_dict[1].append(element)
        else:
            element = sorted(element)
            value = str(tuple(element))
            normalized_dict[len(element)].append(value)

    for key,value in normalized_dict.items():
        normalized_dict[key] = sorted(value)

    printingvalue = {}
    i = 0
    for value in normalized_dict.values():
        printingvalue[i]=''
        for element in value:
            printingvalue[i] += ','+element
        printingvalue[i] += '\n\n'
        i+=1
    return printingvalue

def higher_candidates(higher_k_candidate_list):
    if higher_k_candidate_list is not None and len(higher_k_candidate_list) > 0:
        candidates_new = []
        if type(higher_k_candidate_list[0])==str:
            for pair in combinations(higher_k_candidate_list,2):
                candidates_new.append(pair)
        else:
            for index in range(len(higher_k_candidate_list)-1):
                current_tuple = higher_k_candidate_list[index]
                for next_tuple in higher_k_candidate_list[index+1:]:
                    if current_tuple[:-1] == next_tuple[:-1]:
                        required_tuple = tuple(sorted(list(set(current_tuple).union(set(next_tuple)))))
                        candidates_new.append(required_tuple)
                    else:
                        break
        return candidates_new


def generate_candidates(partitioned_data, support, size):
    if partitioned_data is None:
        return
    list_of_baskets = list(partitioned_data)
    #support for current parition
    new_scaled_support = support * float(len(list_of_baskets)/size)
    dictionary_count_items ={}
    dictionary_frequent_items={}

    #to count frequncy of each item
    for basket in list_of_baskets:
        for i in basket:
            if i not in dictionary_count_items.keys():
                dictionary_count_items[i] = 1
            else:
                dictionary_count_items[i] += 1
    
    frequent_items_dictionary = dict(filter(lambda count: count[1]>= new_scaled_support,dictionary_count_items.items()))
    frequent_single_item_list = list(frequent_items_dictionary.keys())
    candidates = frequent_single_item_list

    k = 1
    while candidates is not None and len(candidates)>0:
        count_in_baskets_dictionary = {}
        for basket in list_of_baskets:
            basket = list(set(basket).intersection(set(frequent_single_item_list)))
            for candidate in candidates:
                concerned_set =set()
                if type(candidate) == str:
                    #if candidate single itemset
                    concerned_set.add(candidate)
                else:
                    #if candidate multiple itemset
                    concerned_set = set(candidate)
                if concerned_set.issubset(set(basket)):
                    if candidate not in count_in_baskets_dictionary.keys():
                        count_in_baskets_dictionary[candidate]=1
                    else:
                        count_in_baskets_dictionary[candidate]+=1
        frequent_items_dictionary = dict(filter(lambda count: count[1]>= new_scaled_support, count_in_baskets_dictionary.items()))
        dictionary_frequent_items[k]=list(frequent_items_dictionary.keys())
        k +=1
        candidates = higher_candidates(sorted(list(frequent_items_dictionary.keys())))
    return dictionary_frequent_items.values()

def itemsets_finalcount(partitioned_data,candidate_list):
    count_in_baskets_dictionary = {}
    basket = list(partitioned_data)

    for candidate in candidate_list:
        concerned_set =set()
        if type(candidate) == str:
            #if candidate single itemset
            concerned_set.add(candidate)
        else:
             #if candidate multiple itemset
            concerned_set = set(candidate)
        if concerned_set.issubset(set(basket)):
            if candidate not in count_in_baskets_dictionary.keys():
                count_in_baskets_dictionary[candidate]=1
            else:
                count_in_baskets_dictionary[candidate]+=1
    return count_in_baskets_dictionary.items()

if __name__ == '__main__':
    
    start_time = time.time()

    sc = SparkContext()

    filter_threshold = int(sys.argv[1])
    support_num = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]
    
    intermediate_file_path = preprocessing(input_file_path = input_file_path)

    processed_data_rdd = sc.textFile(intermediate_file_path,6)
    header_line = processed_data_rdd.first()
    baskets_rdd = None

    user_business_rdd = processed_data_rdd.filter(lambda x : x != header_line and x != "").map(lambda x : (x.split(',')[0], x.split(',')[1])).groupByKey().mapValues(list).filter(lambda customer_product : len(customer_product[1]) > filter_threshold)
    baskets_rdd = user_business_rdd.map(lambda x : x[1])

    son(baskets_rdd,support_num)
    
    endtime = time.time()
    execution_time = endtime - start_time

    print('Duration: ',execution_time)