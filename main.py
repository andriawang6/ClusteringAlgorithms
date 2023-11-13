import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# reads easy.csv and converts to lists
easy = pd.read_csv("easy.csv", header=0)
easy_index = []
easy_value = []
for i in range(len(easy)):
    easy_index.append(easy["index"][i])
    easy_value.append(easy["Value"][i])

# reads harder.csv and converts to lists
harder = pd.read_csv("harder.csv", header=0)
harder_index = []
harder_value = []
for i in range(len(harder)):
    harder_index.append(harder["index"][i])
    harder_value.append(harder["Value"][i])

# reads students.csv and converts to lists
students = pd.read_csv("students.csv", header=0)
students_name = []
students_age = []
students_height = []
students_weight = []
students_reading_level = []
students_math_score = []
for i in range(len(students)):
    students_name.append(students["Name"][i])
    students_age.append(students["Age"][i])
    students_height.append(students["Height"][i])
    students_weight.append(students["Weight"][i])
    students_reading_level.append(students["Reading level"][i])
    students_math_score.append(students["Math score"][i])


# returns the maximum value in a list
def list_min(my_list):
    my_min = my_list[0]
    for i in my_list:
        if i < my_min:
            my_min = i
    return my_min


# returns the maximum value in a list
def list_max(my_list):
    my_max = my_list[0]
    for i in my_list:
        if i > my_max:
            my_max = i
    return my_max


# returns the index of the minimum value in a list
def min_index(my_list):
    my_min_index = 0
    for i in range(len(my_list)):
        if my_list[i] < my_list[my_min_index]:
            my_min_index = i
    return my_min_index


# returns the index of the maximum value in a list
def max_index(my_list):
    my_max_index = 0
    for i in range(len(my_list)):
        if my_list[i] > my_list[my_max_index]:
            my_max_index = i
    return my_max_index


# returns a list sorted in ascending order.
def sort_ascending(my_list):
    my_copy = my_list.copy()
    sorted_list = []
    for i in range(len(my_copy)):
        minimum_index = min_index(my_copy)
        sorted_list.append(my_copy[minimum_index])
        my_copy.pop(minimum_index)
    return sorted_list


# returns a list sorted in descending order.
def sort_descending(my_list):
    my_copy = my_list.copy()
    sorted_list = []
    for i in range(len(my_copy)):
        maximum_index = max_index(my_copy)
        sorted_list.append(my_copy[maximum_index])
        my_copy.pop(maximum_index)
    return sorted_list


# takes a list of data and returns the distance between consecutive points (sort first)
def distance_between(my_list):
    sorted_list = sort_ascending(my_list)
    dist_list = []
    for i in range(len(sorted_list) - 1):
        distance = sorted_list[i + 1] - sorted_list[i]
        dist_list.append(distance)
    return dist_list


# takes a list of data and returns the indexes of the two points which are the closest together
# only finds the first closest points
def index_of_closest_points(my_list):
    dist_list = distance_between(my_list)
    x = min_index(dist_list)
    return x, x + 1


# clusters hierarchically through average linkage
def hierarchical(my_list, num_clusters):
    sorted_list = sort_ascending(my_list)

    # creates a list of sorted_list separated into its separate cluster
    lists = [[sorted_list[i]] for i in range(len(sorted_list))]

    for i in range(len(sorted_list) - num_clusters):
        # find the index of the two points that are the closest together
        num1, num2 = index_of_closest_points(sorted_list)

        # combines the two lists that are closest
        lists[num1] = lists[num1] + lists[num2]

        # removes the latter element
        lists.remove(lists[num2])

        # replaces sorted_list[num1] and [num2] with the average of the values
        sorted_list[num1] = np.average(lists[num1])

        # removes the latter element
        sorted_list.pop(num2)
    return lists


# picks a random centroid from a random part in the list
def random_centroids(my_list, num_clusters):
    return np.array([[my_list[int(np.floor(random.random() * len(my_list)))]] for num in range(num_clusters)])


# assigns lists in my_list to centroids in centroids
def assign_clusters(my_list, centroids, k):
    # creates a multidimensional empty list
    cluster = [[] for i in range(len(centroids))]

    for num in my_list:
        dis = []

        # finds the distance between each element in my_list and centroids and adds them to a list
        for clus in centroids:
            dis.append(abs(num - clus))

        # determines which cluster to add the element num to and appends it
        cluster[min_index(dis)].append(num)
    return cluster


# finds the mean of all the values in a cluster and returns it; returns 0 if the cluster is empty
def find_cluster_mean(clusters):
    if len(clusters) == 0:
        return 0
    return np.mean(clusters)


# runs k-means clustering
def one_d_cluster(my_list, num_clusters):
    # creates random centroids and assign the values in my_list to each centroid
    centroids = random_centroids(my_list, num_clusters)
    clusters = assign_clusters(my_list, centroids, num_clusters)

    # creates a while loop that runs as long as a centroid has been changed
    moved = True
    while moved:
        # assumes nothing has been moved
        moved = False

        # runs through each of the clusters
        for i in range(num_clusters):
            temp_val = centroids[i]

            # reassigns the value of centroids based on the mean of the elements within it
            centroids[i] = find_cluster_mean(clusters[i])

            # checks if centroids have been moved; if so, allows the while loop to run again
            if temp_val != centroids[i]:
                moved = True

        # reassigns clusters based on the new position of the centroids
        clusters = assign_clusters(my_list, centroids, num_clusters)
    return centroids, clusters


# follows the same logic as assign_clusters 1D, but with two lists and a different distance formula
def assign_clusters_2d(list1, list2, centroids_x, centroids_y, k):
    cluster_x = [[] for i in range(k)]
    cluster_y = [[] for i in range(k)]
    for i in range(len(list1)):
        dis = []
        for j in range(len(centroids_x)):
            # calculates the distance between values using the distance formula
            dis.append(abs(np.sqrt(((list1[i] - centroids_x[j]) ** 2) + ((list2[i] - centroids_y[j]) ** 2))))
        cluster_x[min_index(dis)].append(list1[i])
        cluster_y[min_index(dis)].append(list2[i])
    return cluster_x, cluster_y


# follows same logic as 1D but with two lists for each variable
def two_d_cluster(list1, list2, num_clusters):
    centroids_x = random_centroids(list1, num_clusters)
    centroids_y = random_centroids(list2, num_clusters)
    clusters_x, clusters_y = assign_clusters_2d(list1, list2, centroids_x, centroids_y, num_clusters)
    moved = True
    while moved:
        moved = False
        for i in range(num_clusters):
            temp_val = centroids_x[i].copy()
            temp_val_2 = centroids_y[i].copy()
            centroids_x[i] = np.mean((clusters_x[i]))
            centroids_y[i] = np.mean(find_cluster_mean(clusters_y[i]))
            if (temp_val != centroids_x[i]) or (temp_val_2 != centroids_y[i]):
                moved = True
        clusters_x, clusters_y = assign_clusters_2d(list1, list2, centroids_x, centroids_y, num_clusters)
    return centroids_x, centroids_y, clusters_x, clusters_y


# Graphs 1D Clustering
one_d_cluster_num = 5 # number of clusters to group the data into
# cycles through colors so each cluster is a different color !!
color = iter(plt.cm.rainbow(np.linspace(0, 1, one_d_cluster_num)))
centroids, clusters = one_d_cluster(students_math_score, one_d_cluster_num)
for x in clusters:
    c = next(color)
    for i in x:
        plt.scatter(i, 0, color=c)
# plots centroids
plt.scatter(centroids, np.zeros(len(centroids)), color="black", alpha=0.5)
plt.show()

# Graphs 2D Clustering
two_2_cluster_num = 5  # number of clusters you want to group it into
centroids_x, centroids_y, result_x, result_y = two_d_cluster(students_reading_level, students_math_score, two_2_cluster_num)
n = two_2_cluster_num
color = iter(plt.cm.rainbow(np.linspace(0, 1, two_2_cluster_num)))
for i in range(len(result_x)):
    c = next(color)
    for j in range(len(result_x[i])):
        plt.scatter(result_x[i][j], result_y[i][j], color=c)
plt.scatter(centroids_x, centroids_y, color="black", alpha=0.5)
plt.show()
