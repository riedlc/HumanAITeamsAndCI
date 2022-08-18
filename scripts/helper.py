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
    '''
    Measure the performance of model
    :param d: Posterior distributions for every modeled team and player
    :return: Number of correct responses, total responses
    '''
    count = 0
    total = 0
    for _, row in responses.iterrows():
        game = str(row['GAMEID'])
        player = row['SubjectID'] - 1
        guesses = d[game][player]
        top = np.argwhere(guesses == np.amax(guesses))
        c = top[np.random.randint(len(top))][0]
        if c == 2:
            count += 1
        total += 1
    return count, total


def human_agent_match(d):
    '''
    Measure the alignment between a model and the actual human response
    :param d: Posterior distributions for every modeled team and player
    :return: Number of human-agent response matches responses, total responses
    '''
    count = 0
    total = 0
    for _, row in responses.iterrows():
        game = str(row['GAMEID'])
        player = row['SubjectID'] - 1
        guesses = d[game][player]
        top = np.argwhere(guesses == np.amax(guesses))
        selection = top[np.random.randint(len(top))][0]
        if selection == row['Culprit'] - 1:
            count += 1
        total += 1
    return count, total


def randmax(l):
    '''
    Get the maximum of a List.
    If multiple values are the max value, select a random one
    :param l: list to select from
    :return:
    '''
    top = np.argwhere(l == np.amax(l))
    selection = top[np.random.randint(len(top))][0]
    return selection


def logdiff(p1, p2):
    return (np.log(p1) - np.log(p2))


def KLv(p1, p2):
    return np.multiply(p1, logdiff(p1, p2))


def KL(p1, p2):
    '''
    Kullback-Leibler divergence between two densities.
    :param p1: density 1
    :param p2: density 2
    :return:
    '''
    return np.sum(KLv(p1, p2))
