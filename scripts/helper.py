import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import ast
import json
from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import ztest
from copy import deepcopy

# Wong 2011 color friendly palette
colors = ['#909090', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
results = pd.read_csv('../data/results.csv')
responses = pd.read_csv('../data/responses.csv')


def get_adj_dict():
    path = '../data/networks.json'
    with open(path, 'r') as f:
        return json.load(f)


def normalize(a):
    total = sum(a)
    if total == 0:
        return np.zeros(len(a))
    return np.array([x / total for x in a])


def softmax(x):
    y = np.exp(x)
    f_x = y / np.sum(np.exp(x))
    return f_x


# Measures of accuracy
def performance(d):
    count = 0
    total = 0
    for _, row in responses.iterrows():
        game = str(row['GAMEID'])
        #         c = np.argmax(d[game][row['SubjectID'] - 1])
        player = row['SubjectID'] - 1
        guesses = d[game][player]
        #         print(guesses)
        top = np.argwhere(guesses == np.amax(guesses))
        c = top[np.random.randint(len(top))][0]
        if c == 2:
            count += 1
        total += 1
    return count, total


def human_agent_match(d):
    count = 0
    total = 0
    sels = []
    for _, row in responses.iterrows():
        game = str(row['GAMEID'])
        player = row['SubjectID'] - 1
        guesses = d[game][player]
        #         print(guesses)
        top = np.argwhere(guesses == np.amax(guesses))
        selection = top[np.random.randint(len(top))][0]
        #         selection = np.argmax(d[game][player])
        if selection == row['Culprit'] - 1:
            count += 1
        total += 1
    return count, total


# Measures of accuracy
def correct_response_list(d):
    matches = [0] * len(responses)
    for i, row in responses.iterrows():
        game = str(row['GAMEID'])
        c = np.argmax(d[game][row['SubjectID'] - 1])
        if c == 2:
            matches[i] = 1
    return matches


def correct_human_response_list(d, responses):
    matches = [0] * len(responses)
    for i, row in responses.iterrows():
        game = str(row['GAMEID'])
        player = row['SubjectID'] - 1
        selection = np.argmax(d[game][player])
        if selection == row['Culprit'] - 1:
            matches[i] = 1
    return matches


def z_test(a, b):
    return ztest(a, b, np.mean(a) - np.mean(b))


def compare_models(a, b):
    agreement = 0
    total = 0
    for game in a.keys():
        for player in range(len(a[game])):
            total += 1
            if np.argmax(a[game][player]) == np.argmax(b[game][player]):
                agreement += 1
    return agreement, total


def combine_models(a, b, wa, wb):
    d = {}
    for game in a.keys():
        d[game] = []
        for player in range(len(a[game])):
            new = normalize(a[game][player] * wa + b[game][player] * wb)
            d[game].append(new)
    return d


def randmax(l):
    top = np.argwhere(l == np.amax(l))
    selection = top[np.random.randint(len(top))][0]
    return selection


def randmin(l):
    bot = np.argwhere(l == np.amin(l))
    selection = bot[np.random.randint(len(bot))][0]
    return selection


def logdiff(p1, p2):
    return (np.log(p1) - np.log(p2))


def KLv(p1, p2):
    return np.multiply(p1, logdiff(p1, p2))


def KL(p1, p2):
    """
    Kullback-Leibler divergence between densities 1 and 2.
    """
    return np.sum(KLv(p1, p2))
