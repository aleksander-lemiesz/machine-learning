import csv

import numpy as np
import pandas as pd

import load_data
from DecisionTree import DecisionTree


def open_csv(filename):
    rows = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            rows.append(row)
    return rows


def write_submission(evals):
    tasks_csv = open_csv("task.csv")
    submissions = []
    for i, x in enumerate(tasks_csv):
        evaluation = evals.loc[evals['id'].values == int(x[0])]['evaluation'].to_numpy()[0]
        submissions.append([x[0], x[1], x[2], evaluation])


def exe():
    # Load the dataset
    # train, tasks = load_data.exe()
    train = pd.read_csv("train_alt.csv", index_col=0)
    tasks = pd.read_csv("tasks_alt.csv", index_col=0)

    grouped_tasks = tasks.groupby('user_id')
    grouped_train = train.groupby('user_id')

    es = []

    for tasks_group in grouped_tasks:

        user_id = tasks_group[0]

        tasks_no_user = tasks_group[1].drop("user_id", axis='columns').to_numpy()
        train_group_no_user = grouped_train.get_group(user_id).drop(
            columns=["user_id", "evaluation"], axis='columns').to_numpy()
        evaluations = grouped_train.get_group(user_id)[['evaluation']].copy().to_numpy().flatten()

        # Train tree with train data for user
        clf = DecisionTree(max_depth=4)
        clf.fit(train_group_no_user, evaluations)

        for task in tasks_no_user:
            task_id = int(task[0])
            if task_id == 32221:
                print('Starting prediction for task: ' + str(task_id))

            task_features = task[-(len(task) - 1):]

            # Predict using test data
            predictions = clf.predict(task_features)
            # print(predictions)

            es.append([task_id, predictions])

    evals = pd.DataFrame(data=es, columns=["id", "evaluation"])
    write_submission(evals)


if __name__ == '__main__':
    exe()
