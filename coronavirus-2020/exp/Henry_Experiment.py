"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between 
countries. What country has the most similar trajectory
to a given country?
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
MIN_CASES = 1000
N_CLUSTERS = 2
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_US.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []

for val in np.unique(confirmed["Province_State"]):
    df = data.filter_by_attribute(
        confirmed, "Province_State", val)
    cases, labels = data.get_cases_chronologically(df)
    cases = np.sum(cases, axis=0)
    label = labels[0]
    label = label[6]
    features.append(cases)
    targets.append(label)

features = np.concatenate(features, axis=0)
features = np.reshape(features, (58, 119))
targets = np.array(targets)

#for val in [2, 3, 4, 5, 6, 7, 8]:
#    print("cluster number:", val)
#    kmeans = KMeans(n_clusters=val).fit(features)
#    fit_labels = kmeans.labels_
#    for value in range(0, val):
#        indices = np.nonzero(fit_labels == value)
#        indices = indices[0].astype(int)
#        print(value, targets[indices])

deaths = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_deaths_US.csv')
deaths = data.load_csv_data(deaths)
populations = []

for val in np.unique(deaths["Province_State"]):
    df = data.filter_by_attribute(
        deaths, "Province_State", val)
    cases2, labels2 = data.get_cases_chronologically2(df)
    population = np.sum(labels2[:, 11:], axis=0)
    populations.append(population[0])

print(populations)
new_features = []
counter = 0
for row in features:
    sum = np.sum(row)
    print(populations[counter])
    if populations[counter] != 0:
        y = sum / populations[counter]
    else:
        counter += 1
        continue
    days_till_first = np.argwhere(row == 0)
    combined = [1 - (len(days_till_first) / len(row)), y]
    new_features.append(combined)
    print(targets[counter], combined)
    counter += 1

new_features = np.concatenate(new_features, axis=0)
new_features = np.reshape(new_features, (56, -1))


cm = plt.get_cmap('jet')

for val in [2, 3, 4, 5, 6, 7, 8]:
    kmeans = KMeans(n_clusters=val).fit(new_features)
    fit_labels = kmeans.labels_
    colors = [cm(i) for i in np.linspace(0, 1, val)]
    for value in range(0, val):
        indices = np.nonzero(fit_labels == value)
        indices = indices[0].astype(int)
        selections = new_features[indices]
        x_values = np.transpose(selections[:, :1])
        y_values = np.transpose(selections[:, 1:])
        color = colors[value]
        plt.scatter(x_values, y_values, c=color)
    plt.title("Total confirmed cases compared to time of first occurrance")
    plt.xlabel("Days from first US positive test to first State positive test")
    plt.ylabel("Total Confirmed Cases / Population")
    plt.show()


#kmeans1 = KMeans()


#for val in np.unique(confirmed["Province_State"]):
#    df = data.filter_by_attribute(
#        confirmed, "Province_State", val)
#    cases, labels = data.get_cases_chronologically(df)
#    cases = cases.sum(axis=0)

#    if cases.sum() > 10000:
#        i = len(legend)
#        lines = ax.plot(cases, label=labels[0,1])
#        handles.append(lines[0])
#        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
#        lines[0].set_color(colors[i])
#        legend.append(labels[0, 6])

#ax.set_ylabel('# of confirmed cases')
#ax.set_xlabel("Time (days since Jan 22, 2020)")

#ax.set_yscale('log')
#ax.legend(handles, legend, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4)
#plt.tight_layout()
#plt.savefig('results/deaths_by_state.png')