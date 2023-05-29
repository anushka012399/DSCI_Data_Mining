import math
import sys
from sklearn.cluster import KMeans
import numpy as np
import time

def load_data():
    input_file = sys.argv[1]
    cluster = int(sys.argv[2])
    output_file = sys.argv[3]
    return input_file,cluster,output_file

def make_partitions(data):
    return int(len(data) * 0.2)

def processing_first_partition(first_partition):

    beginning_index, last_index = 0, first_partition
    initial_load = input_data[beginning_index:last_index]

    indices, other_features_dict, initial_load_arr = {}, {}, []
    no_of_points = len(initial_load[0].split(",")) - 2
    checking = 2 * math.sqrt(no_of_points)

    for counter, row in enumerate(initial_load):
        row = row.split(",")
        index, other_features = row[0], row[2:]
        initial_load_arr.append(other_features)
        indices[counter] = index
        other_features_dict[str(other_features)] = index

    first_load_array = np.array(initial_load_arr)

    return initial_load_arr,checking,first_load_array,indices,other_features_dict,no_of_points,last_index,counter
    
def k_means(cluster):
    kmeans_model = KMeans(n_clusters=cluster, random_state=553)
    return kmeans_model

def compressed_set_creation(compression_set,rs_point_lst,indices, retained, combined):
    compression_set[indices] = {0: [list(retained_set_dictionary.keys())[list(retained_set_dictionary.values()).index(rs_point_lst[point])] for point in retained]}
    length = len(compression_set[indices][0])
    combined_float = combined[retained, :].astype(np.float)
    compression_set[indices].update({1: length, 2: np.sum(combined_float, axis=0), 3: np.sum(combined_float**2, axis=0)})
    compression_set[indices][4] = np.sqrt((compression_set[indices][3] / length) - (compression_set[indices][2]**2 / (length * length)))
    compression_set[indices][5] = compression_set[indices][2] / length

    return compression_set

def discard_Set_creation(discard_set,index, discard, combined):
    discard_set[index] = {0: [indices[point] for point in discard]}
    length = len(discard_set[index][0])
    combined_float = combined[discard, :].astype(np.float)
    discard_set[index].update({1: length, 2: np.sum(combined_float, axis=0), 3: np.sum(combined_float**2, axis=0)})
    discard_set[index][4] = np.sqrt((discard_set[index][3] / length) - (discard_set[index][2]**2 / (length * length)))
    discard_set[index][5] = discard_set[index][2] / length

    return discard_set

def mahalanobis(point, sets):
    initial_nearest_threshold, index_near_cluster = checking, -10000

    for indexing, set_data in sets.items():
        sd_float, centroid_float = set_data[4].astype(np.float), set_data[5].astype(np.float)
        mah_dis = np.sqrt(sum(((point[ss] - centroid_float[ss]) / sd_float[ss])**2 for ss in range(no_of_points)))
        
        if mah_dis < initial_nearest_threshold:
            initial_nearest_threshold, index_near_cluster = mah_dis, indexing

    return index_near_cluster


def ds_modify(modified_list, position, modified_point, old_centroid):
    centroid_data = modified_list[old_centroid]
    centroid_data[0].append(position)
    centroid_data[1] += 1
    centroid_data[2] += modified_point
    centroid_data[3] += modified_point ** 2
    centroid_data[4] = np.sqrt(centroid_data[3] / centroid_data[1] - (centroid_data[2] ** 2) / (centroid_data[1] ** 2))
    centroid_data[5] = centroid_data[2] / centroid_data[1]

def create_cluster(clusters):
    cluster_dict = {}
    for i in range(len(clusters)):
        cluster_name = clusters[i]
        if cluster_name in cluster_dict:
            cluster_dict[cluster_name].append(i)
        else:
            cluster_dict[cluster_name] = [i]
    return cluster_dict

def compressed_cluster_merging(set1, set2):
    assign_cluster = {}

    for i in set1:
        closest, min_dist = i, checking
        sd1, c1 = set1[i][4], set1[i][5]

        for j in set2:
            if i != j:
                sd2, c2 = set2[j][4], set2[j][5]
                g1_d, g2_d = (
                    sum(((c1[k] - c2[k]) / sd2[k]) ** 2 for k in range(no_of_points) if sd1[k] > 0 and sd2[k] > 0),
                    sum(((c2[k] - c1[k]) / sd1[k]) ** 2 for k in range(no_of_points) if sd1[k] > 0 and sd2[k] > 0),
                )
                min_d = min(np.sqrt(g1_d), np.sqrt(g2_d))

                if min_d < min_dist:
                    closest, min_dist = j, min_d

        assign_cluster[i] = closest

    return assign_cluster


def k_means_first_partition(cluster,first_partition_np):
    # Define number of clusters for the second step
    n_cluster_step2 = 5 * cluster

    # Initialize KMeans clustering model with the specified number of clusters and random seed
    kmeans_model = k_means(n_cluster_step2)

    # Apply KMeans clustering to the first partition of points and obtain the cluster assignments
    potential_rs_cluster = kmeans_model.fit_predict(first_partition_np)

    # Initialize dictionary to store the points in each cluster
    rs_cluster = {}

    for i, clusterid in enumerate(potential_rs_cluster):
        point = initial_load_Arr[i]
        rs_cluster.setdefault(clusterid, []).append(point)


    # Initialize dictionary to store the representative points for each cluster
    rs_dict = {}

    # Iterate over each cluster and its points, and add the representative point (if there is only one) to the rs_dict dictionary
    for iii in rs_cluster.keys():
        if len(rs_cluster[iii]) == 1:
            point_rscluster = rs_cluster[iii][0]
            dimen = initial_load_Arr.index(point_rscluster)
            rs_dict[indices[dimen]] = point_rscluster
            initial_load_Arr.remove(point_rscluster)
            indices.update({l: indices[l+1] for l in range(dimen, len(indices) - 1)})
    return potential_rs_cluster,rs_dict

def cs_merging(group1, group2):
    compressed_set[group1][0].extend(compressed_set[group2][0])
    compressed_set[group1][1] += compressed_set[group2][1]
    compressed_set[group1][2] += compressed_set[group2][2]
    compressed_set[group1][3] += compressed_set[group2][3]

    sd_merging = (compressed_set[group1][3] / compressed_set[group1][1]) - (compressed_set[group1][2] ** 2) / (compressed_set[group1][1] ** 2)
    compressed_set[group1][4], compressed_set[group1][5] = np.sqrt(sd_merging), compressed_set[group1][2] / compressed_set[group1][1]


def cs_ds_merging(group1_cs, group2_ds):
    discard_set[group2_ds][0].extend(compressed_set[group1_cs][0])
    discard_set[group2_ds][1] += compressed_set[group1_cs][1]
    discard_set[group2_ds][2] += compressed_set[group1_cs][2]
    discard_set[group2_ds][3] += compressed_set[group1_cs][3]

    sd_merging = (discard_set[group2_ds][3] / discard_set[group2_ds][1]) - (discard_set[group2_ds][2] ** 2) / (discard_set[group2_ds][1] ** 2)
    discard_set[group2_ds][4], discard_set[group2_ds][5] = np.sqrt(sd_merging), discard_set[group2_ds][2] / discard_set[group2_ds][1]


def noise_handling():
    discard_set = {}
    for index in modified_cluster:
        discard_set = discard_Set_creation(discard_set, index, modified_cluster[index], no_retained_set_arr)

    reduced_set_list = list(retained_set_dictionary.values())
    rs_points_array = np.array(reduced_set_list)
    cluster_total = int(len(reduced_set_list) * 0.5 + 1)
    kmeans = k_means(cluster_total)
    cs_clusters = create_cluster(kmeans.fit_predict(rs_points_array))

    cs_cluster_set = {}
    for index_1 in cs_clusters.keys():
        if len(cs_clusters[index_1]) > 1:
            cs_cluster_set = compressed_set_creation(cs_cluster_set,reduced_set_list,index_1, cs_clusters[index_1], rs_points_array)

    for k, v in cs_clusters.items():
        if len(v) > 1:
            for index_1 in v:
                deleting_key = next(key for key, value in retained_set_dictionary.items() if value == reduced_set_list[index_1])
                del retained_set_dictionary[deleting_key]


    reduced_set_list = []
    for key in retained_set_dictionary.keys():
        reduced_set_list.append(retained_set_dictionary[key])

    return discard_set,cs_cluster_set,reduced_set_list

def write_intermediate(intermediate_output_count,discard_points_count,compression_cluster_count,compression_point_count,retained_point_count):
      f.write("Round " + str(intermediate_output_count + 1) + ": " + str(discard_points_count) + "," + str(
        compression_cluster_count) + "," + str(
        compression_point_count) + "," + str(retained_point_count) + "\n")
      
def process_other_rounds(last_index,counter,reduced_set_list):
    last_round = 4
    for num_round in range(1, 5):
        beginning_index = last_index
        new_load = []
        last_index = len(input_data) if num_round == last_round else beginning_index + initial_load
        new_load = input_data[beginning_index:last_index]


        list_of_points = []
        last_ctr = counter
        for row in new_load:
            index, _, *pointer = row.split(",")
            list_of_points.append(pointer)
            indices[counter] = index
            other_features_dict[str(pointer)] = index
            counter += 1


        arr = np.array(list_of_points)


        for i in range(len(arr)):
            data_particular = arr[i]
            point = arr[i].astype(np.float)
            index_new = indices[last_ctr + i]
            closest_cluster = mahalanobis(point, discard_set)

            if closest_cluster > -1:
                ds_modify(discard_set, index_new, point, closest_cluster)
            else:
                closest_cluster = mahalanobis(point, compressed_set)
                if closest_cluster > -1:
                    ds_modify(compressed_set, index_new, point, closest_cluster)
                else:
                    retained_set_dictionary[index_new] = data_particular.tolist()
                    reduced_set_list.append(data_particular.tolist())

        arr = np.array(reduced_set_list)
        cluster_num = int(len(reduced_set_list) * 0.5 + 1)
        kmeans = KMeans(n_clusters=cluster_num, random_state=123)
        clusters_all = create_cluster(kmeans.fit_predict(arr))


        # Iterate over each CS cluster

        for cluster_cs in clusters_all.keys():
            if len(clusters_all[cluster_cs]) > 1:
                counting = 0
                if cluster_cs in compressed_set.keys():
                    while counting in compressed_set:
                        counting += 1
                else:
                    counting = cluster_cs
                compressed_set_creation(compressed_set,reduced_set_list,counting, clusters_all[cluster_cs], arr)

        for cluster in clusters_all.keys():
            if len(clusters_all[cluster]) > 1:
                for i in clusters_all[cluster]:
                    deleting_point = other_features_dict[str(reduced_set_list[i])]
                    if deleting_point in retained_set_dictionary.keys():
                        del retained_set_dictionary[deleting_point]

        reduced_set_list = []
        for key in retained_set_dictionary.keys():
            reduced_set_list.append(retained_set_dictionary[key])

        closest_cluster = compressed_cluster_merging(compressed_set, compressed_set)

        for cluster in closest_cluster.keys():
            if cluster != closest_cluster[cluster] and closest_cluster[
                cluster] in compressed_set.keys() and cluster in compressed_set.keys():
                cs_merging(cluster, closest_cluster[cluster])
                del compressed_set[closest_cluster[cluster]]

        if num_round == last_round:
            closest_cluster = compressed_cluster_merging(compressed_set, discard_set)
            for last_run in closest_cluster.keys():
                if closest_cluster[last_run] in discard_set.keys() and last_run in compressed_set.keys():
                    cs_ds_merging(last_run, closest_cluster[last_run])
                    del compressed_set[last_run]

        discard_points_count = 0
        compression_cluster_count = 0
        compression_point_count = 0
        for key in discard_set.keys():
            discard_points_count += discard_set[key][1]
        for key in compressed_set.keys():
            compression_cluster_count += 1
            compression_point_count += compressed_set[key][1]
        retained_point_count = len(reduced_set_list)

        write_intermediate(int_count,discard_points_count,compression_cluster_count,compression_point_count,retained_point_count)

    return discard_set

def write_results(final_point_index_cluster_dict):
    f.write("\n")
    f.write("The clustering results: ")
    for final_point in sorted(final_point_index_cluster_dict.keys(), key=int):
        f.write("\n")
        f.write(str(final_point) + "," + str(final_point_index_cluster_dict[final_point]))

def final_dict_creation(ds_group_set,cs_cluster_set,rs_dict):
    final_point_index_cluster_dict = {}
    for point_ds in ds_group_set:
        for single in ds_group_set[point_ds][0]:
            final_point_index_cluster_dict[single] = point_ds

    for point_cs in cs_cluster_set:
        for single_cs in cs_cluster_set[point_cs][0]:
            final_point_index_cluster_dict[single_cs] = point_cs

    for point_rs in rs_dict:
        final_point_index_cluster_dict[point_rs] = -1
    return final_point_index_cluster_dict

if __name__ == "__main__":

    start_time = time.time()
    input_file_path,clusters,output_file_path = load_data()
    file = open(input_file_path, "r")
    input_data = np.array(file.readlines())

    initial_load = make_partitions(input_data)
    initial_load_Arr,checking,first_arr,indices,other_features_dict,no_of_points,last_index,counter = processing_first_partition(initial_load)
    retained_sets_clusters,retained_set_dictionary = k_means_first_partition(clusters,first_arr)

    no_retained_set_arr = np.array(initial_load_Arr)
    modified_kmeans = k_means(clusters)

    # Perform K-means clustering on the data points without noise to form initial clusters
    modified_cluster = create_cluster(modified_kmeans.fit_predict(no_retained_set_arr))

    discard_set,compressed_set,retained_set_list=noise_handling()

    # Initialize variables to track various counts

    int_count,ds_count, compression_clusters,cs_count,retained_set_count = 0,0,0,0,0

    # Calculate the number of data points that were discarded in the initial clustering, and the number of clusters and data points in the noise clusters
    for i in discard_set.keys():
        ds_count += discard_set[i][1]
    for i in compressed_set.keys():
        compression_clusters == compression_clusters + 1
        cs_count += compressed_set[i][1]

    retained_set_count = len(retained_set_list)

    f = open(output_file_path, "w")
    f.write("The intermediate results:\n")
    write_intermediate(int_count,ds_count,compression_clusters,cs_count,retained_set_count)

    discard_set = process_other_rounds(last_index,counter,retained_set_list)
    final_dictionary = final_dict_creation(discard_set,compressed_set,retained_set_dictionary)

    write_results(final_dictionary)

    end_time = time.time() - start_time
    print("Duration : ", end_time)

