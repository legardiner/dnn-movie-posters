import movies_dataset as movies
import movies_genre_model

min_year = 1980
max_year = 2017
epochs = 10000
genres = movies.list_genres(8)
ratings = ['G', 'PG', 'PG-13', 'R']

# select a smaller ratio (e.g. 40) for quicker training
for ratio in [70]:
#     we load the data once for each ratio, so we can use it for multiple versions, epochs, etc.
    x_train, x2_train, y_train = movies.load_genre_data(min_year, max_year, genres, ratings, ratio, 'train')
    x_validation, x2_validation, y_validation = movies.load_genre_data(min_year, max_year, genres, ratings, ratio, 'validation')
    for version in [1, 2, 3]:
        movies_genre_model.build(version, min_year, max_year, genres, ratings, ratio, epochs,
                                 x_train=x_train,
                                 x2_train=x2_train,
                                 y_train=y_train,
                                 x_validation=x_validation,
                                 x2_validation=x2_validation,
                                 y_validation=y_validation)
