import numpy as np
#import pandas as pd
#from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


def data_preparation(n_cluster, data):
    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred = kmeans.fit_predict(data)

    return kmeans



