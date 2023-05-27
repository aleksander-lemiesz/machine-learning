import csv

import numpy as np
import pandas as pd

import load_data


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


def get_most_similar_user(m_id, user_id, grouped_train):
    orig_user = grouped_train.get_group(user_id)
    user_movies = orig_user[['t_id', 'evaluation']].to_numpy()
    best_sim = float('inf')
    best_sim_user = orig_user

    for train_user_id, train_user_group in grouped_train:
        cur_sim = 0
        if train_user_id == user_id or does_group_contain_movie(m_id, train_user_group) == False:
            continue
        movies = train_user_group[['t_id', 'evaluation']].to_numpy()
        for m1 in user_movies:
            for m2 in movies:
                if m1[0] == m2[0]:
                    cur_sim += (m1[1] - m2[1]) ** 2
        cur_sim /= len(user_movies)

        if cur_sim < best_sim:
            best_sim = cur_sim
            best_sim_user = train_user_group

    return best_sim_user


def get_eval_from_user(m_id, sim_user):
    series_eval = sim_user.loc[sim_user['t_id'] == m_id]['evaluation']
    return series_eval[series_eval.index[0]]


def does_group_contain_movie(m_id, user_group):
    return len(user_group.loc[user_group['t_id'] == m_id]) > 0


def exe():
    # Load the dataset
    # train, tasks = load_data.exe()
    train = pd.read_csv("train_alt.csv", index_col=0)
    tasks = pd.read_csv("tasks_alt.csv", index_col=0)

    grouped_tasks = tasks.groupby('user_id')
    grouped_train = train.groupby('user_id')

    es = []

    for i, tasks_group in enumerate(grouped_tasks):

        user_id = tasks_group[0]
        print('Processing user ' + str(user_id))
        # print(f'{(i / len(grouped_tasks) * 100):.2}% done...')
        print("{:.2f}".format(i / len(grouped_tasks) * 100) + '% done...')

        tasks_no_user = tasks_group[1].drop("user_id", axis='columns').to_numpy()

        for task in tasks_no_user:
            task_id = int(task[0])
            # print('Processing task ' + str(task_id) + '...')
            # print(str(i/len(tasks_no_user)*100) + '% done...')

            task_features = task[-(len(task) - 1):]
            m_id = task_features[0]

            # Predict using test data
            sim_user = get_most_similar_user(m_id, user_id, grouped_train)
            predictions = get_eval_from_user(m_id, sim_user)

            # print(predictions)

            es.append([task_id, predictions])

    evals = pd.DataFrame(data=es, columns=["id", "evaluation"])
    write_submission(evals)


if __name__ == '__main__':
    exe()
