import csv

import numpy as np
import pandas as pd
import load_data
import collab.collab as col


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

    submission_file = "submission.csv"
    with open(submission_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerows(submissions)


def exe():
    # Load the dataset
    # train, tasks = load_data.exe()
    train = pd.read_csv("train_alt.csv", index_col=0)
    tasks = pd.read_csv("tasks_alt.csv", index_col=0)

    es = []

    # Collab vars
    n = 10
    M = 200
    R = 358
    # Fit
    movie_arr, user_id_dict = col.create_movie_arr(M, R, train)
    fitted = col.fit(movie_arr, M, R, n)

    for task in tasks.to_numpy():
        task_id = int(task[0])

        user_id = task[2]
        user_id_key = col.get_key_from_value(user_id_dict, user_id)

        movie_id = task[1] - 1

        # Predict using fitted matrices
        predictions = col.f(fitted[0], fitted[1], movie_id, user_id_key)

        es.append([task_id, int(predictions)])

    evals = pd.DataFrame(data=es, columns=["id", "evaluation"])
    write_submission(evals)


if __name__ == '__main__':
    exe()
