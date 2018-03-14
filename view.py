import argparse
import argh
import os
import time
from contextlib import contextmanager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define variables
datafile_dir = "CSV"
positive_sample = "people.csv"
negative_sample = "robot.csv"

@contextmanager
def timer(message):
    tick = time.time()
    yield
    tock = time.time()
    print("%s: %.3f" % (message, (tock - tick)))

def read_CSV_files(dir="CSV"):
    data_dir = os.path.join(os.getcwd(), dir)
    positive_file = os.path.join(data_dir, positive_sample)
    negative_file = os.path.join(data_dir, negative_sample)

    positive_data = pd.read_csv(positive_file, header=None)
    negative_data = pd.read_csv(negative_file, header=None)
    return positive_data, negative_data

def produce_dataset(positive_data, negative_data):
    positive_label = pd.DataFrame([1 for _ in range(positive_data.shape[0])])
    negative_label = pd.DataFrame([0 for _ in range(negative_data.shape[0])])
    data = pd.concat([positive_data, negative_data], ignore_index=True)
    label = pd.concat([positive_label, negative_label], ignore_index=True).values.ravel()
    division = positive_data.shape[0]
    return data, label, division

# Data handling
# pos, neg = read_CSV_files()
# data, label, division = produce_dataset(pos, neg)
# train_x, test_x, train_y, test_y = train_test_split(data, label, random_state=0, test_size=0.25, stratify=label)

def PCA_dimensionality_reduction(positive_data, negative_data, d=2):
    data = pd.concat([positive_data, negative_data], ignore_index=True)
    data_new = PCA(n_components=d).fit_transform(data)
    positive_data_new = data_new[:positive_data.shape[0]]
    negative_data_new = data_new[positive_data.shape[0]+1:]
    return positive_data_new, negative_data_new

def TSNE_dimensionality_reduction(positive_data, negative_data, scale=1000, d=2):
    # Because of high counting cost, we use only a part of the data
    positive_data_sub = positive_data.sample(n=scale)
    negative_data_sub = negative_data.sample(n=scale)
    data_sub = pd.concat([positive_data_sub, negative_data_sub], ignore_index=True)
    data_new = TSNE(n_components=d).fit_transform(data_sub)
    positive_data_new = data_new[:scale]
    negative_data_new = data_new[scale+1:]
    return positive_data_new, negative_data_new

def draw_2d_image(p_data, n_data, name):
    plt.figure(name)
    plt.scatter(p_data[:, 0], p_data[:, 1], s=0.3, c="g")
    plt.scatter(n_data[:, 0], n_data[:, 1], s=0.3, c="r")
    plt.show()

# def draw_PCA_scatter(positive_data=pos, negative_data=neg, d=2):
#     pos, neg = PCA_dimensionality_reduction(positive_data, negative_data, d=d)
#     draw_2d_image(pos, neg, "PCA scatter")

# def draw_TSNE_scatter(positive_data=pos, negative_data=neg, scale=1000, d=2):
def draw_TSNE_scatter(positive_data, negative_data, scale=1000, d=2):
    pos, neg = TSNE_dimensionality_reduction(positive_data, negative_data, scale=scale, d=d)
    draw_2d_image(pos, neg, "TSNE scatter")

# Counting Kullback–Leibler divergence of the features
# def count_features_KL(data=data):
#     positive_data = data[:division]
#     negative_data = data[division+1:]
#     # Draw histogram picture for each feature, and count KL
#     for i in range(16):
#         plt.figure("feature"+str(i))
#         pos_statis, _ = np.histogram(positive_data[i], bins=50, range=(data[i].min(), data[i].max()), normed=False)
#         neg_statis, _ = np.histogram(negative_data[i], bins=50, range=(data[i].min(), data[i].max()), normed=False)
#         KL = entropy(neg_statis, pos_statis)
#         # Draw picture
#         index = np.arange(50)
#         bar_width = 0.35
#         plt.xlabel('bin')
#         plt.ylabel('number')
#         plt.title("Kullback–Leibler divergence on the "+str(i)+": "+str(KL))
#         plt.bar(index, pos_statis, bar_width, color='g', label='people')
#         plt.bar(index+bar_width, neg_statis, bar_width, color='r', label='robot')
#         plt.show()
#
# def construct_pipeline():
#     param_dist = {"C": [0.5, 0.75, 1.0],
#                   "kernel": ["linear", "rbf", "sigmoid"]
#                   }
#     n_iter_search = 8
#     clf = RandomizedSearchCV(svm.SVC(), param_distributions=param_dist, n_iter=n_iter_search)
#     pipe = Pipeline([('pca', PCA(n_components=2)),
#                      ('clf', clf),
#                     ])
#     return pipe
#
# def result_report(true_y, pred_y):
#     labels = [1, 0]
#     target_names = ["people", "robot"]
#     return confusion_matrix(true_y, pred_y), \
#            classification_report(true_y, pred_y, labels=labels, target_names=target_names, digits=3)

# def classify_by_pipeline(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y):
#     pipe = construct_pipeline()
#     print("Chosing parameter...")
#     with timer("Chosing parameter"):
#         pipe.fit(train_x, train_y)
#     print(pipe.named_steps['clf'].best_params_)
#     pred_y = pipe.predict(test_x)
#     cfs, clr = result_report(test_y, pred_y)
#     print("confussion_matrix:")
#     print(cfs)
#     print("classification_report:")
#     print(clr)
#
# parser = argparse.ArgumentParser()
# argh.add_commands(parser, [count_features_KL, draw_PCA_scatter, draw_TSNE_scatter, classify_by_pipeline])

# if __name__ == "__main__":
    # argh.dispatch(parser)
    # draw_TSNE_scatter