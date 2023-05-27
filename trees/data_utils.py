import pandas as pd
import tmdbsimple as tmdb
tmdb.API_KEY = 'c5aa9e739a77012982eda3690fd1dc43'


def handle_movies():
    movies_csv = pd.read_csv("movie.csv", sep=';')
    movies_arr = movies_csv.to_numpy()
    movies_features = []

    for m in movies_arr:
        t_id = m[0]
        tmdb_id = m[1]

        movie = tmdb.Movies(tmdb_id).info()
        genres = set([g['name'] for g in movie['genres']])

        features = [t_id, genres]
        movies_features.append(features)
        # print(features)

    alt_m = pd.DataFrame(data=movies_features, columns=['t_id', 'genres'])
    alt_m.to_csv("movies_alt.csv", index=False)
    return movies_features


def handle_train(movies):
    train_csv = pd.read_csv("train.csv", sep=';', header=None)
    train_arr = train_csv.to_numpy()
    train_x = []

    for x in train_arr:
        user_id = x[1]
        t_id = int(x[2])
        evaluation = x[3]

        movie = movies[t_id - 1]

        features = [
            movie[1],
            user_id,
            evaluation
        ]
        # print(features)
        train_x.append(features)

    alt_t = pd.DataFrame(data=train_x, columns=['genres', 'user_id', 'evaluation'])
    alt_t.to_csv("train_alt.csv")
    return train_x


def handle_tasks(movies):
    tasks_csv = pd.read_csv("task.csv", sep=';', header=None)
    tasks_arr = tasks_csv.to_numpy()
    tasks = []

    for x in tasks_arr:
        x_id = x[0]
        user_id = x[1]
        t_id = int(x[2])

        movie = movies[t_id - 1]

        features = [
            x_id,
            movie[1],
            user_id
        ]
        # print(features)
        tasks.append(features)

    alt_t = pd.DataFrame(data=tasks, columns=['x_id', 'genres', 'user_id'])
    alt_t.to_csv("tasks_alt.csv")
    return tasks


def data_utils():
    movies = handle_movies()
    # train = handle_train(movies)
    # tasks = handle_tasks(movies)
    # return train, tasks, movies


if __name__ == '__main__':
    data_utils()
