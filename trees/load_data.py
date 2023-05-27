import pandas as pd
import tmdbsimple as tmdb
tmdb.API_KEY = 'c5aa9e739a77012982eda3690fd1dc43'


def handle_movies():
    print('Reading movies...')
    movies_csv = pd.read_csv("movie.csv", sep=';')
    movies_arr = movies_csv.to_numpy()
    movies_features = []

    print('Extracting features...')
    for m in movies_arr:
        t_id = m[0]
        tmdb_id = m[1]

        movie = tmdb.Movies(tmdb_id).info()
        # genres = set([g['name'] for g in movie['genres']])
        # adult = movie['adult']
        budget = movie['budget']
        # original_language = movie['original_language']
        popularity = movie['popularity']
        release_date = movie['release_date'][:4]
        revenue = movie['revenue']
        vote_average = movie['vote_average']
        vote_count = movie['vote_count']

        features = [
            t_id,
            # adult,
            budget,
            # original_language,
            popularity,
            release_date,
            revenue,
            vote_average,
            vote_count
            ]
        movies_features.append(features)
        # print(features)

    print('Saving movies with features...')
    alt_m = pd.DataFrame(data=movies_features,
                         columns=[
                             't_id',
                             'budget',
                             'popularity',
                             'release_date',
                             'revenue',
                             'vote_average',
                             'vote_count'
                         ])
    alt_m.to_csv("movies_alt.csv", index=False)
    return movies_features


def load_movies_alt():
    ms = pd.read_csv("movies_alt.csv")
    return ms


def handle_train(movies):
    print('Loading train data...')
    train_csv = pd.read_csv("train.csv", sep=';', header=None)
    train_arr = train_csv.to_numpy()
    train = []

    print('Extracting train data features...')
    for x in train_arr:
        user_id = x[1]
        t_id = int(x[2])
        evaluation = x[3]

        movie = movies.iloc[t_id - 1]

        features = [
            movie[1],
            movie[2],
            movie[3],
            movie[4],
            movie[5],
            movie[6],
            user_id,
            evaluation
        ]
        # print(features)
        train.append(features)

    print('Saving train data...')
    alt_t = pd.DataFrame(data=train, columns=[
        'budget',
        'popularity',
        'release_date',
        'revenue',
        'vote_average',
        'vote_count',
        'user_id',
        'evaluation'
    ])
    alt_t.to_csv("train_alt.csv")
    return train


def handle_tasks(movies):
    tasks_csv = pd.read_csv("task.csv", sep=';', header=None)
    tasks_arr = tasks_csv.to_numpy()
    tasks = []

    for x in tasks_arr:
        x_id = x[0]
        user_id = x[1]
        t_id = int(x[2])

        movie = movies.iloc[t_id - 1]

        features = [
            x_id,
            movie[1],
            movie[2],
            movie[3],
            movie[4],
            movie[5],
            movie[6],
            user_id
        ]
        # print(features)
        tasks.append(features)

    alt_t = pd.DataFrame(data=tasks, columns=[
        'x_id',
        'budget',
        'popularity',
        'release_date',
        'revenue',
        'vote_average',
        'vote_count',
        'user_id'
    ])
    alt_t.to_csv("tasks_alt.csv")
    return tasks


def exe():
    # handle_movies()
    movies = load_movies_alt()
    train = handle_train(movies)
    tasks = handle_tasks(movies)
    return train, tasks


if __name__ == '__main__':
    exe()
