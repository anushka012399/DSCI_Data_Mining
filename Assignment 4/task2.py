from collections import defaultdict
from operator import add
from pyspark import SparkContext
from itertools import combinations
import random
import sys
import time

def load_data(spark, file_path):
    df = spark.textFile(file_path)
    header = df.first()
    ub_rdd = df.filter(lambda x: x != header).map(lambda x: (x.split(',')[0], x.split(',')[1])).groupByKey().map(lambda x: (x[0], list(set(x[1])))).collect()
    return ub_rdd

def make_vertices_edges_set(ub_rdd,threshold):
	vertices = set()
	edges = set()
	for l1 in ub_rdd:
		for l2 in ub_rdd:
			if l1[0] != l2[0]:
				if len(set(l1[1]) & set(l2[1])) >= threshold:
					vertices.add(l1[0])
					vertices.add(l2[0])
					edges.add(tuple((l1[0], l2[0])))
	return vertices,edges

def update_paths(node,parent,num_path):
	parent_nodes = parent[node]
	if len(parent_nodes) > 0:
		num_path[node] = sum([num_path[parent_node] for parent_node in parent_nodes])
	else:
		num_path[node] = 1

	return

def buildtree(root):
	tree,child,path_counts = dict(),dict(),dict()
	parent = defaultdict(set)

	tree[0] = root
	path_counts[root] = 1
	used_nodes = {root}
	first_level = adjacent_nodes[root]
	child[root] = first_level
	level = 1

	while first_level != set():
		tree[level] = first_level
		used_nodes = used_nodes.union(first_level)
		new_nodes = set()
		for node in first_level:
			adj_nodes = adjacent_nodes[node]
			child_nodes = adj_nodes - used_nodes
			child[node] = child_nodes
			for key, vals in child.items():
				for val in vals:
					parent[val].add(key)
			update_paths(node,parent,path_counts)
			new_nodes = new_nodes.union(adj_nodes)
		level_nodes = new_nodes - used_nodes
		first_level = level_nodes
		level += 1

	return tree,parent,path_counts,level

def calculate_edge_values(tree,level,find_parent,num_path,parent_value):
	edge_value = dict()

	while level != 1:
		for node in tree[level - 1]:
			parent_nodes = find_parent[node]
			
			for parent_node in parent_nodes:
				weight = num_path[parent_node] / num_path[node]
				edge_value[tuple(sorted((node, parent_node)))] = weight * parent_value[node]
				parent_value[parent_node] += edge_value[tuple(sorted((node, parent_node)))]
		level -= 1
	return edge_value

def Girvan_Newman(root):

	tree,find_parent,num_path,level = buildtree(root)

	parent_value = defaultdict(float)
	for node in vertices:
		if node != root:
			parent_value[node] = 1
	
	edge_value = calculate_edge_values(tree,level,find_parent,num_path,parent_value)

	return [(key, val) for key, val in edge_value.items()]


def community(node, adjacent_nodes):

	used_nodes = set()
	community = set()
	count = 0
	adj_nodes = adjacent_nodes[node]
	while True:
		used_nodes = used_nodes.union(adj_nodes)
		count += 1
		new_nodes = set()
		for n in adj_nodes:
			new_adj_nodes = adjacent_nodes[n]
			new_nodes = new_nodes.union(new_adj_nodes)
		new_used_nodes = used_nodes.union(new_nodes)
		if len(used_nodes) == len(new_used_nodes):
			break
		adj_nodes = new_nodes - used_nodes

	community = used_nodes
	if community == set():
		community = {node}

	return community


def find_communities(node, vertices, adjacent_nodes):

	communities = []
	used_nodes = community(node, adjacent_nodes)
	unused_nodes = vertices - used_nodes
	communities.append(used_nodes)
	while True:
		new_used_nodes = community(random.sample(unused_nodes, 1)[0], adjacent_nodes)
		communities.append(new_used_nodes)
		used_nodes = used_nodes.union(new_used_nodes)
		unused_nodes = vertices - used_nodes
		if len(unused_nodes) == 0:
			break

	return communities


def calculate_modularity(communities, m):

	modularity = 0
	for community in communities:
		partition_modularity = 0
		for i in community:
			for j in community:
				partition_modularity += A[(i, j)] - degree[i] * degree[j] / (2 * m)
		modularity += partition_modularity
	modularity = modularity / (2 * m)

	return modularity


def find_betweenness(spark,vertices):
	betweenness = spark.parallelize(vertices).map(lambda node: Girvan_Newman(node)).flatMap(lambda x: [pair for pair in x]) \
	.reduceByKey(add) \
	.map(lambda x: (x[0], x[1] / 2)) \
	.sortBy(lambda x: (-x[1], x[0])).collect()

	return betweenness

def write_betweenness_file(betweenness_output_file_path,betweenness):
	with open(betweenness_output_file_path, 'w+') as fout:
		for pair in betweenness:
			fout.write(str(pair)[1:-1] + '\n')
	return 

def iterative_search_for_communities(spark,betweenness,adjacent_nodes, vertices):
	m = len(edges) / 2

	current_edge_count = m
	max_modularity = -1

	while True:
		highest_betweenness = betweenness[0][1]
		for pair in betweenness:
			if pair[1] == highest_betweenness:

				adjacent_nodes[pair[0][0]].remove(pair[0][1])
				adjacent_nodes[pair[0][1]].remove(pair[0][0])
				current_edge_count -= 1


		temp_communities = find_communities(random.sample(vertices, 1)[0], vertices, adjacent_nodes)


		cur_modularity = calculate_modularity(temp_communities, m)

		if cur_modularity > max_modularity:
			max_modularity = cur_modularity
			communities = temp_communities


		if current_edge_count == 0:
			break

		betweenness = find_betweenness(spark,vertices)

	return communities

def write_community_file(community_output_file_path,sorted_communities):
	with open(community_output_file_path, 'w+') as fout:
		for community in sorted_communities:
			fout.write(str(community)[1:-1] + '\n')
	return

if __name__ == '__main__':
	start = time.time()
	filter_threshold = int(sys.argv[1])
	input_file_path = sys.argv[2]
	betweenness_output_file_path = sys.argv[3]
	community_output_file_path = sys.argv[4]

	spark = SparkContext.getOrCreate()
	spark.setLogLevel('WARN')

	ub_rdd = load_data(spark, input_file_path)

	vertices,edges = make_vertices_edges_set(ub_rdd,filter_threshold)

	adjacent_nodes = defaultdict(set)
	for pair in edges:
		adjacent_nodes[pair[0]].add(pair[1])

	betweenness = find_betweenness(spark,vertices)
	
	write_betweenness_file(betweenness_output_file_path,betweenness)

	# Calculate degree for each node
	degree = {node: len(adjacent_nodes[node]) for node in vertices}

	# Create adjacency matrix
	A = {(node1, node2): int((node1, node2) in edges) for node1 in vertices for node2 in vertices}

	communities = iterative_search_for_communities(spark,betweenness,adjacent_nodes, vertices)

	sorted_communities = spark.parallelize(communities) \
	.map(lambda x: sorted(x)) \
	.sortBy(lambda x: (len(x), x)).collect()

	write_community_file(community_output_file_path,sorted_communities)

	end = time.time()
	print('Duration: {}'.format(end - start))


