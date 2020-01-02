import numpy as np
import pandas as pd
from collections import Counter

from sklearn.decomposition import PCA, TruncatedSVD


def get_df_data():
    ifile_path = 'data/data.csv'
    df_data = pd.read_csv(ifile_path, sep=';')
    train_target = pd.read_csv('data/train.csv', sep=';')
    test_id = pd.read_csv('data/test.csv', sep=';')
    df_train = pd.merge(df_data, train_target, on='ID', how='inner')
    df_subm = pd.merge(df_data, test_id, on='ID', how='inner')
    return df_data, df_train, df_subm


class Expr:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __repr__(self):
        return ' '.join(['{}_{}'.format(k, v) for k, v in self.__dict__.items()])


def get_line_stats(line):
    counter = Counter(line)
    chars_set = set(line)
    eng = 'abcdefghijklmnopqrstuvwxyz'
    rus = 'абвгдеёжзийклмнопрстухфцчшщъыьэюя'
    words = line.replace('?', '').replace(',', '').replace('.', '').split()  # !
    words_uniq = set(words)
    words_len = [len(w) for w in words]
    chars_mistakes = sum((i+1 < len(line) and (line[i+1] == ' ')) or (i - 1 > 0 and line[i-1].isalpha())
                          for i in range(len(line)) if line[i] in {'.', ',', '!'})

    stats = dict(
                 first_valid=int(line[0].isupper()),
                 last_valid=int(line[-1] == '?'),
                 point=counter.get('.', 0),
                 comma=counter.get(',', 0),
                 excl=counter.get('!', 0),
                 q_point=counter.get('?', 0),
                 total_len=len(chars_set),
                 caps=len({c for c in chars_set if c.isupper()}),
                 chars=len({c for c in chars_set if c.isalpha()}),
                 another_chars_uniq=sum({line.count(c) for c in chars_set if not c.isalpha()}),
                 rus_letter=sum(c in rus for c in chars_set),
                 eng_letter=sum(c in eng for c in chars_set),
                 words=len(words),
                 wors_uniq=len(words_uniq),
                 min_word_len=np.min(words_len) if len(words_len) else 0,
                 mean_word_len=np.mean(words_len) if len(words_len) else 0,
                 median_word_len=np.median(words_len) if len(words_len) else 0,
                 max_word_len=np.max(words_len) if len(words_len) else 0,
                 num_uniq=sum({line.count(c) for c in chars_set if not c.isnumeric()}),
                 space_valid=int('  ' not in line),
                 chars_mistakes=chars_mistakes
    )
    return stats


def get_stats_features(lines):
    stats_list = []
    for line in lines:
        stats = get_line_stats(line)
        stats_list.append(stats)
    return pd.DataFrame(stats_list)


def get_pca_features(X_vec, n_components=20):
    # pca = PCA(n_components=n_components)
    pca = TruncatedSVD(n_components=n_components)
    return pca.fit_transform(X_vec)


if __name__ == '__main__':
    lines = [
             'В каком году была впервые опубликована «Книга рекордов Гиннесса»?',
             'Как звали первую жену Петра 1?',
             'Сколько было президентов при СССР?',
             'Какой фразы не было в песне ""Алладин""',
             'Какой маршал СССР командовал парадом Победы в Москве 24 июня 1945 года?',
             'Сколько сантиметров в одном километре?',
             'По одной из версий, этот термин получил новое название, потому что охрана порядка на некоторых футбольных матчах усиливалась за счёт полицейских-кавалеристов. Назовите этот термин.',
             'Как называется луч, который делит угол пополам?',
             'Что такое sketch book?',
             'Какая из песен группы nickelback принесла им их популярность?',
             'Как зовут основателя ВКонтакте?',
             'Как называется-боязнь длинных слов?',
             ]

    print(lines[0], get_line_stats(lines[0]))

    df_stats = get_stats_features(lines)
    print(df_stats.to_string())
