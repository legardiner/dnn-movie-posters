"""
Manage movies data (extracted from /data/MovieGenre.csv).
"""

import io
import os.path
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

images_folder = 'data/images/'
test_data_ratio = 7  # 14.3%
validation_data_ratio = 6  # 14.3%
parsed_movies = []  # cache


class Movie:
    imdb_id = 0
    title = ''
    year = 0
    genres = []
    poster_url = ''
    rating = ''

    def poster_file_exists(self) -> bool:
        return os.path.isfile(self.poster_file_path())

    def download_poster(self):
        try:
            response = urllib.request.urlopen(self.poster_url)
            data = response.read()
            file = open(self.poster_file_path(), 'wb')
            file.write(bytearray(data))
            file.close()
            return data
        except:
            print('-> error')

    def poster_file_path(self, size=100) -> str:
        return images_folder + str(size) + "/" + self.poster_file_name()

    def poster_file_name(self):
        return str(self.imdb_id) + '.jpg'

    def is_valid(self) -> bool:
        return self.poster_url.startswith('https://') \
               and 1900 <= self.year <= 2018 \
               and len(self.title) > 1 \
               and len(self.genres) == 1 \
               and len(self.rating) >= 1

    def to_rgb_pixels(self, poster_size):
        data = open(images_folder + str(poster_size) + '/' + str(self.imdb_id) + '.jpg', "rb").read()
        image = Image.open(io.BytesIO(data))
        rgb_im = image.convert('RGB')
        pixels = []
        for x in range(image.size[0]):
            row = []
            for y in range(image.size[1]):
                r, g, b = rgb_im.getpixel((x, y))
                pixel = [r / 255, g / 255, b / 255]
                row.append(pixel)
            pixels.append(row)

        return pixels
    
    def to_rgb_pixels_flipped(self, poster_size):
        data = open(images_folder + str(poster_size) + '/' + str(self.imdb_id) + '.jpg', "rb").read()
        image = Image.open(io.BytesIO(data))
        rgb_im = image.convert('RGB')
        pixels = []
        for x in range(image.size[0]):
            row = []
            for y in range(image.size[1] - 1, -1, -1):
                r, g, b = rgb_im.getpixel((x, y))
                pixel = [r / 255, g / 255, b / 255]
                row.append(pixel)
            pixels.append(row)

        return pixels

    def get_genres_vector(self, genres):
        vector = []
        if self.has_any_genre(genres):
            for genre in genres:
                vector.append(int(self.has_genre(genre)))
        return vector
        
    def get_genres(self):
        return self.genres
    
    def get_rating_vector(self, ratings):
        vector = []
        if self.has_any_rating(ratings):
            for rating in ratings:
                vector.append(int(self.has_rating(rating)))
        return vector

    def short_title(self) -> str:
        max_size = 20
        return (self.title[:max_size] + '..') if len(self.title) > max_size else self.title

    def is_test_data(self) -> bool:
        return self.imdb_id % test_data_ratio == 0

    def has_any_genre(self, genres) -> bool:
        return len(set(self.genres).intersection(genres)) > 0

    def has_genre(self, genre) -> bool:
        return genre in self.genres
    
    def has_any_rating(self, ratings) -> bool:
        return len(set(self.rating).intersection(ratings)) > 0
    
    def has_rating(self, rating) -> bool:
        return rating in self.rating


    def __str__(self):
        return self.short_title() + ' (' + str(self.year) + ')'


def download_posters(min_year=0):
    for movie in list_movies():
        print(str(movie))
        if movie.year >= min_year:
            if not movie.poster_file_exists():
                movie.download_poster()
                if movie.poster_file_exists():
                    print('-> downloaded')
                else:
                    print('-> could not download')
            else:
                print('-> already downloaded')
        else:
            print('-> skip (too old)')


def load_genre_data(min_year, max_year, genres, ratings, ratio, data_type, verbose=True):
    xs = []
    x2s = []
    ys = []

    for year in reversed(range(min_year, max_year + 1)):
        if verbose:
            print('loading movies', data_type, 'data for', year, '...')

        xs_year, x2s_year, ys_year = _load_genre_data_per_year(year, genres, ratings, ratio, data_type)
        _add_to(xs_year, xs)
        _add_to(x2s_year, x2s)
        _add_to(ys_year, ys)

        if verbose:
            print('->', len(xs_year))

    return np.concatenate(xs), np.concatenate(x2s), np.concatenate(ys)


def _load_genre_data_per_year(year, genres, ratings, poster_ratio, data_type):
    xs = []
    x2s = []
    ys = []

    count = 1
    for movie in list_movies(year, genres, ratings):
        if movie.poster_file_exists():
            if (data_type == 'train' and not movie.is_test_data() and count % validation_data_ratio != 0) \
                    or (data_type == 'validation' and not movie.is_test_data() and count % validation_data_ratio == 0) \
                    or (data_type == 'test' and movie.is_test_data()):
                x = movie.to_rgb_pixels(poster_ratio)
                y = movie.get_genres_vector(genres)
                x2 = movie.get_rating_vector(ratings)
                xs.append(x)
                x2s.append(x2)
                ys.append(y)
                x = movie.to_rgb_pixels_flipped(poster_ratio)
                xs.append(x)
                x2s.append(x2)
                ys.append(y)
            count += 1

    xs = np.array(xs, dtype='float32')
    x2s = np.array(x2s, dtype='uint8')
    ys = np.array(ys, dtype='uint8')
    return xs, x2s, ys


def _add_to(array1d, array2d):
    if len(array1d) > 0:
        array2d.append(array1d)


def list_movies(year=None, genres=None, ratings=None):
    if len(parsed_movies) == 0:
        data = pd.read_csv('data/MovieGenre3.csv', encoding='ISO-8859-1')
        for index, row in data.iterrows():
            movie = _parse_movie_row(row)
            if movie.is_valid():
                parsed_movies.append(movie)

        parsed_movies.sort(key=lambda m: m.imdb_id)

    result = parsed_movies

    if year is not None:
        result = [movie for movie in result if movie.year == year]

    if genres is not None:
        result = [movie for movie in result if movie.has_any_genre(genres)]
    
    if ratings is not None:
        result = [movie for movie in result if movie.has_any_rating(ratings)]

    return result


def _parse_movie_row(row) -> Movie:
    movie = Movie()
    movie.imdb_id = int(row['imdbId'])
    movie.title = row['Title'][:-7]
    year = row['Title'][-5:-1]
    if year.isdigit() and len(year) == 4:
        movie.year = int(row['Title'][-5:-1])

    url = str(row['Poster'])
    if len(url) > 0:
        movie.poster_url = url.replace('"', '')

    genre_str = str(row['Genre2'])
    if len(genre_str) > 0:
        movie.genres = genre_str.split('|')
        
    rating_str = row['Rating']
    if len(rating_str) > 0:
        movie.rating = rating_str


    return movie


def search_movie(imdb_id=None, title=None) -> Movie:
    movies = list_movies()
    for movie in movies:
        if imdb_id is not None and movie.imdb_id == imdb_id:
            return movie
        if title is not None and movie.title == title:
            return movie


def list_genres(number):
    if number == 3:
        return ['Comedy', 'Drama', 'Action']
    if number == 4:
        return ['Adventure', 'Documentary', 'Horror', 'Romance']
    if number == 7:
        return list_genres(3) + ['Animation', 'Romance', 'Adventure', 'Horror']
    if number == 8:
        return ['Action', 'Animation', 'Comedy', 'Drama', 'Family', 'Horror', 'Rom-Com', 'Sci-Fi']
    if number == 14:
        return list_genres(7) + ['Sci-Fi', 'Crime', 'Mystery', 'Thriller', 'War', 'Family', 'Western']
