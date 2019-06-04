import numpy as np
from scipy.spatial import distance
from sklearn.datasets import load_iris
import time
import matplotlib.pyplot as plt
start_time = time.time()


def create_data():
    all_data = load_iris()
    data = all_data.data
    data = data[:, 0:2]
    clusters = list()
    for z in range(len(data)):
        clusters.append(data[z])
    for ii in range(len(clusters)):
        clusters[ii] = clusters[ii].reshape(1, len(clusters[ii]))
    return clusters


def create_dis_matrix(data1):
    dis_matrix = np.zeros((len(data1), len(data1)))
    for t in range(len(data1)):
        for xx in range(len(data1)):
            if t is xx:
                continue
            else:
                dis = distance.euclidean(data1[t], data1[xx])
                dis_matrix[t, xx] = dis
    return dis_matrix


def if_core_return_neighbors(minpts, radios, dis_matx, data_set, data_index):
    neighbors = 0
    list_indexes = list()
    for p in range(len(data_set)):
        if dis_matx[data_index, p] < radios and data_index != p:
            neighbors += 1
            list_indexes.append(p)
    if neighbors > minpts:
        return list_indexes
    else:
        return None


def cluster(minpts1, radios1, matrix_dis, data_set1):
    clusters_indexes = list()
    core_neighbors = None
    core_neighbors1 = None
    for inx in range(len(data_set1)):
        if core_neighbors is None:
            core_neighbors = if_core_return_neighbors(minpts1, radios1, matrix_dis, data_set1, inx)
        else:
            clusters_indexes.append(inx)
            break
    if core_neighbors is None:
        return "noise", None
    for o in core_neighbors:
        if o not in clusters_indexes:
            clusters_indexes.append(o)
        if core_neighbors1 is None:
            core_neighbors1 = if_core_return_neighbors(minpts1, radios1, matrix_dis, data_set1, o)
        else:
            for y in core_neighbors1:
                if y not in core_neighbors:
                    core_neighbors.append(y)
            core_neighbors1 = None
    cluster0 = data_set1[clusters_indexes[0]]
    for z in clusters_indexes[1:]:
        cluster0 = np.concatenate((cluster0, data_set1[z]), axis=0)
    return clusters_indexes, cluster0


def main(minpts11, radios11):
    dat = create_data()
    clusterss = list()
    noise = dat.copy()
    for u in range(len(noise)):
       diss_matrix = create_dis_matrix(noise)
       clusters_indexes1, cluster1 = cluster(minpts11, radios11, diss_matrix, noise)
       if clusters_indexes1 is "noise":
           break
       else:
           clusters_indexes1.sort(reverse=True)
           for t in clusters_indexes1:
               del noise[t]
           clusterss.append(cluster1)
    return clusterss, noise


cluster_list, noise_list = main(3, 0.33)


for i in range(len(cluster_list)):
    plt.scatter(cluster_list[i][:, 0], cluster_list[i][:, 1])
for x in range(len(noise_list)):
    plt.scatter(noise_list[x][:, 0], noise_list[x][:, 1], c="k")
plt.axis([4, 8, 1.5, 5])
plt.show()
        
print(time.time() - start_time)




        
