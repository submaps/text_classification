import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import lightgbm as lgb
from xgboost import XGBClassifier
from data_utils import get_df_data, Expr, get_stats_features, get_pca_features


start = pd.Timestamp.now()

df_data, df_train, df_subm = get_df_data()

train = df_train
test = df_subm

train_text = df_train['Question']
test_text = df_subm['Question']

expr = Expr(max_words_count=20000,
            max_chars_count=80000,
            max_charngram=7,
            lowercase=False,
            stack_words_and_chars=True,
            lgb_max_deph=2,
            n_estimators=200,
            num_leaves=10,
            max_wordngram=1,
            use_stats_features=True,
            use_count_features=True,
            use_pca=False,
            max_features_count=50000,
            n_components=300,
            model_name='lgb',
            start=start)

all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, expr.max_wordngram),
    max_features=expr.max_words_count,
    lowercase=expr.lowercase)

word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(1, expr.max_charngram),
    max_features=expr.max_chars_count,
    lowercase=expr.lowercase)

char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_stats_features = get_stats_features(train_text)
test_stats_features = get_stats_features(test_text)

count_char_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 7),
                                        max_features=expr.max_features_count, lowercase=False)
count_word_vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=expr.max_features_count, lowercase=False)

count_char_vectorizer.fit(all_text)
count_word_vectorizer.fit(all_text)

train_char_count_features = count_char_vectorizer.transform(train_text)
test_char_count_features = count_char_vectorizer.transform(test_text)

train_word_count_features = count_word_vectorizer.transform(train_text)
test_word_count_features = count_word_vectorizer.transform(test_text)

if expr.use_stats_features:
    # train_features = train_stats_features
    # test_features = test_stats_features
    if expr.use_count_features and expr.use_pca:
        train_features = hstack([train_char_features,
                                 train_word_features,
                                 train_stats_features,
                                 train_word_count_features,
                                 train_char_count_features,
                                 get_pca_features(train_word_count_features, n_components=expr.n_components),
                                 get_pca_features(train_char_count_features, n_components=expr.n_components),
                                 get_pca_features(train_char_features, n_components=expr.n_components),
                                 get_pca_features(train_word_features, n_components=expr.n_components)
                                 ])
        test_features = hstack([test_char_features,
                                test_word_features,
                                test_stats_features,
                                test_word_count_features,
                                test_char_count_features,
                                get_pca_features(test_word_count_features, n_components=expr.n_components),
                                get_pca_features(test_char_count_features, n_components=expr.n_components),
                                get_pca_features(test_char_features, n_components=expr.n_components),
                                get_pca_features(test_word_features, n_components=expr.n_components)
                                ])
    elif expr.use_count_features:
        train_features = hstack(
            [train_char_features, train_word_features, train_stats_features, train_word_count_features,
             train_char_count_features])
        test_features = hstack([test_char_features, test_word_features, test_stats_features, test_word_count_features,
                                test_char_count_features])
    else:
        train_features = hstack([train_char_features, train_word_features, train_stats_features])
        test_features = hstack([test_char_features, test_word_features, test_stats_features])
else:
    if expr.stack_words_and_chars:
        train_features = hstack([train_char_features, train_word_features])
        test_features = hstack([test_char_features, test_word_features])
    else:
        train_features = train_char_features
        test_features = test_char_features

expr.train_features_shape = train_features.shape
print('train features shape', expr.train_features_shape)

submission = pd.DataFrame.from_dict({'ID': test['ID']})

train_target = train['Answer']

if expr.model_name == 'logreg':
    classifier = LogisticRegression(C=0.1, solver='sag')
elif expr.model_name == 'xgb':
    classifier = XGBClassifier()
elif expr.model_name == 'lgb':
    classifier = lgb.LGBMClassifier(max_depth=expr.lgb_max_deph,
                                    metric="auc",
                                    n_estimators=expr.n_estimators,
                                    num_leaves=expr.num_leaves,
                                    boosting_type="gbdt",
                                    learning_rate=0.1,
                                    feature_fraction=0.45,
                                    colsample_bytree=0.45,
                                    bagging_fraction=0.8,
                                    bagging_freq=5,
                                    reg_lambda=0.2)

expr.cv_auc_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
print('CV score for class {} is {}'.format(1, expr.cv_auc_score))

classifier.fit(train_features, train_target, verbose=0)
y_pred = classifier.predict_proba(test_features)[:, 1]
y_pred_class = classifier.predict(test_features)

submission['Answer'] = y_pred
submission_info = {'1th_class_count': sum(y_pred_class),
                   'total': len(y_pred),
                   '1th_class_perc': sum(y_pred) / len(y_pred) * 100}

print('{} class balance: {} total: {} perc: {:0.2f} auc: {:0.5f}'.format(
    expr.model_name,
    submission_info['1th_class_count'],
    submission_info['total'],
    submission_info['1th_class_perc'],
    expr.cv_auc_score))

submission_file_path = f'output/{expr}.csv'.replace('000', 'K') \
    .replace('False', 'F') \
    .replace('features', 'fs') \
    .replace('count', 'c') \
    .replace('True', 'T')

submission.to_csv(submission_file_path, index=False, header=None)


expr.__dict__.update(submission_info)
print('saved:', submission_file_path)
expr.elapsed = pd.Timestamp.now() - expr.start
print('elapsed:', expr.elapsed)
expr.submission_file_path = submission_file_path
df_stats = pd.DataFrame(expr.__dict__)
df_stats.to_csv('stats/expr_stats.csv', mode='a')
