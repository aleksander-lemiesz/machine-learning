import csv
import numpy as np
import pandas as pd
from collections import Counter
from data_utils import data_utils


def jaccard_similarity(a, b):
    # convert to set
    a = set(a)
    b = set(b)
    # calucate jaccard similarity
    return float(len(a.intersection(b))) / len(a.union(b))


def knn(k, train_x, train_y, task):
    # Calculate distances
    distances = []
    for x in train_x:
        distances.append(jaccard_similarity(task, x))

    # Get the closest k
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [train_y[i][0] for i in k_indices]

    # Find the most common label in k neighbors
    most_common = Counter(k_nearest_labels).most_common()
    min_dist = min(distances)

    return most_common[0][0], min_dist


def open_csv(filename):
    rows = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            rows.append(row)
    return rows


def write_submission(evals, sims):
    tasks_csv = open_csv("task.csv")
    submissions = []
    similarities = []
    for i, x in enumerate(tasks_csv):
        evaluation = evals.loc[evals['id'].values == int(x[0])]['evaluation'].to_numpy()[0]
        sim = sims.loc[sims['id'].values == int(x[0])]['dist'].to_numpy()[0]
        submissions.append([x[0], x[1], x[2], evaluation])
        similarities.append(sim)

    submission_file = "submission.csv"
    with open(submission_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerows(submissions)


def exe():
    train, tasks, movies = data_utils()

    es = []
    ss = []

    alt_tasks = pd.DataFrame(data=tasks, columns=['x_id', 'genres', 'user_id'])
    alt_train = pd.DataFrame(data=train, columns=['genres', 'user_id', 'evaluation'])

    grouped_tasks = alt_tasks.groupby('user_id')
    grouped_train = alt_train.groupby('user_id')

    for tasks_group in grouped_tasks:

        user_id = tasks_group[0]

        tasks_no_user = tasks_group[1].drop("user_id", axis='columns').to_numpy()
        train_group_no_user = grouped_train.get_group(user_id).drop(
            columns=["user_id", "evaluation"], axis='columns').to_numpy()
        train_features = [x[0] for x in train_group_no_user]
        evaluations = grouped_train.get_group(user_id)[['evaluation']].copy().to_numpy()

        for task in tasks_no_user:
            task_id = int(task[0])

            task_features = task[-(len(task) - 1):][0]

            evaluation, dist = knn(25, train_features, evaluations, task_features)

            es.append([task_id, evaluation])
            ss.append([task_id, dist])

    evals = pd.DataFrame(data=es, columns=["id", "evaluation"])
    sims = pd.DataFrame(data=ss, columns=["id", "dist"])
    write_submission(evals, sims)


if __name__ == '__main__':
    exe()
