import pandas
import numpy as np
from tqdm import tqdm

class Data(object):
    pass

dot_product = True
movielens = False
include_genre_feature = False
implicit_feedback = True
negative_samples_per_positive_sample = 1

nrows_read = 10000
nrows_we_work_with = 2000
number_of_movies = 100
number_of_users = 100
batch_size = 200
number_of_extra_users = 2

data = Data()
if movielens:
    # data comes from https://www.kaggle.com/grouplens/movielens-20m-dataset
    number_of_embedding_dimensions = 50
    data.tag = pandas.read_csv('~/Downloads/movie_lens_kaggle/tag.csv', nrows=nrows_read)
    data.genome_tags = pandas.read_csv('~/Downloads/movie_lens_kaggle/genome_tags.csv', nrows=nrows_read)
    data.genome_scores = pandas.read_csv('~/Downloads/movie_lens_kaggle/genome_scores.csv', nrows=nrows_read)
    data.link = pandas.read_csv('~/Downloads/movie_lens_kaggle/link.csv', nrows=nrows_read)
    data.movie = pandas.read_csv('~/Downloads/movie_lens_kaggle/movie.csv', nrows=nrows_read)
    data.rating = pandas.read_csv('~/Downloads/movie_lens_kaggle/rating.csv', nrows=nrows_read)
else:
    number_of_embedding_dimensions = 2
    # users 0, 1, 2 are the same
    # users 3, 4, 5 are the same but opposite to user 0
    # user 1 has a rating for movies 2 and 5 put into validation
    # user 2 has no interaction for movies 2 and 5
    # similarly for users 4 and 5
    user_0 = [
        {'movieId': 0, 'userId': 0, 'rating': 5.0, 'train': True},
        {'movieId': 1, 'userId': 0, 'rating': 5.0, 'train': True},
        {'movieId': 2, 'userId': 0, 'rating': 5.0, 'train': True},
        {'movieId': 3, 'userId': 0, 'rating': 0.0, 'train': True},
        {'movieId': 4, 'userId': 0, 'rating': 0.0, 'train': True},
        {'movieId': 5, 'userId': 0, 'rating': 0.0, 'train': True},
    ]

    user_3 = [
        {'movieId': 0, 'userId': 3, 'rating': 0.0, 'train': True},
        {'movieId': 1, 'userId': 3, 'rating': 0.0, 'train': True},
        {'movieId': 2, 'userId': 3, 'rating': 0.0, 'train': True},
        {'movieId': 3, 'userId': 3, 'rating': 5.0, 'train': True},
        {'movieId': 4, 'userId': 3, 'rating': 5.0, 'train': True},
        {'movieId': 5, 'userId': 3, 'rating': 5.0, 'train': True},
    ]
    data.rating = pandas.DataFrame([
        {'movieId': 0, 'userId': 1, 'rating': 5.0, 'train': True},
        {'movieId': 1, 'userId': 1, 'rating': 5.0, 'train': True},
        {'movieId': 2, 'userId': 1, 'rating': 5.0, 'train': False},
        {'movieId': 3, 'userId': 1, 'rating': 0.0, 'train': True},
        {'movieId': 4, 'userId': 1, 'rating': 0.0, 'train': True},
        {'movieId': 5, 'userId': 1, 'rating': 0.0, 'train': False},

        {'movieId': 0, 'userId': 2, 'rating': 5.0, 'train': True},
        {'movieId': 1, 'userId': 2, 'rating': 5.0, 'train': False},
        {'movieId': 2, 'userId': 2, 'rating': 5.0, 'train': True},
        {'movieId': 3, 'userId': 2, 'rating': 0.0, 'train': True},
        {'movieId': 4, 'userId': 2, 'rating': 0.0, 'train': False},
        {'movieId': 5, 'userId': 2, 'rating': 0.0, 'train': True},
        
        {'movieId': 0, 'userId': 3, 'rating': 0.0, 'train': True},
        {'movieId': 1, 'userId': 3, 'rating': 0.0, 'train': True},
        {'movieId': 2, 'userId': 3, 'rating': 0.0, 'train': False},
        {'movieId': 3, 'userId': 3, 'rating': 5.0, 'train': True},
        {'movieId': 4, 'userId': 3, 'rating': 5.0, 'train': True},
        {'movieId': 5, 'userId': 3, 'rating': 5.0, 'train': False},

        {'movieId': 0, 'userId': 4, 'rating': 0.0, 'train': True},
        {'movieId': 1, 'userId': 4, 'rating': 0.0, 'train': False},
        {'movieId': 2, 'userId': 4, 'rating': 0.0, 'train': True},
        {'movieId': 3, 'userId': 4, 'rating': 5.0, 'train': True},
        {'movieId': 4, 'userId': 4, 'rating': 5.0, 'train': False},
        {'movieId': 5, 'userId': 4, 'rating': 5.0, 'train': True},
        
        {'movieId': 6, 'userId': -1, 'rating': 2.5, 'train': True},
        {'movieId': 7, 'userId': -1, 'rating': 2.5, 'train': True},
    ])
    list_of_dicts = user_0 + user_3
    for i in range(number_of_extra_users):
        list_of_dicts.extend([{**x, **{'userId': i + 6}} for x in user_0])
    for i in range(number_of_extra_users):
        list_of_dicts.extend([{**x, **{'userId': i + 6 + number_of_extra_users}} for x in user_3])
    data.rating = data.rating.append(pandas.DataFrame(list_of_dicts))
    data.movie = pandas.DataFrame([
        {'movieId': 0, 'genres': 'Adventure|Comedy'},
        {'movieId': 1, 'genres': 'Adventure|Comedy'},
        {'movieId': 2, 'genres': 'Adventure|Comedy'},
        {'movieId': 3, 'genres': 'Adventure|Comedy'},
        {'movieId': 4, 'genres': 'Adventure|Comedy'},
        {'movieId': 5, 'genres': 'Adventure|Comedy'},
    ])
# from https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
if movielens:
    data.rating = data.rating.sample(frac=1).reset_index(drop=True)

all_genres = set()

for row in range(len(data.movie)):
    dictionary = data.movie.iloc[row].to_dict()
    genres = dictionary['genres']
    for genre in genres.split('|'):
        all_genres.add(genre)

all_genres = sorted(all_genres)
print(f'# of genres: {len(all_genres)}')

map_movie_id_to_genre_feature_vector = dict()

for row in range(len(data.movie)):
    dictionary = data.movie.iloc[row].to_dict()
    genres = dictionary['genres']
    movie_id = dictionary['movieId']
    if movie_id not in map_movie_id_to_genre_feature_vector:
        map_movie_id_to_genre_feature_vector[movie_id] = [0] * len(all_genres)
    for genre in genres.split('|'):
        map_movie_id_to_genre_feature_vector[movie_id][all_genres.index(genre)] = 1

movies_counter = dict()
all_movies = set()
for row in range(len(data.rating)):
    dictionary = data.rating.iloc[row].to_dict()
    movie_id = dictionary['movieId']
    all_movies.add(movie_id)
    movies_counter[movie_id] = movies_counter.get(movie_id, 0) + 1

all_movies = sorted(movies_counter.keys(), key=lambda x: -movies_counter[x])[:number_of_movies]

users_counter = dict()
all_users = set()
for row in range(len(data.rating)):
    dictionary = data.rating.iloc[row].to_dict()
    user_id = dictionary['userId']
    movie_id = dictionary['movieId']
    if movie_id in all_movies:
        all_users.add(user_id)
        users_counter[user_id] = users_counter.get(user_id, 0) + 1

all_users = sorted(users_counter.keys(), key=lambda x: -users_counter[x])[:number_of_users]
nrows_we_work_with = min(len(data.rating), nrows_we_work_with)

map_user_id_to_movies_rated = dict()
for row in range(len(data.rating)):
    dictionary = data.rating.iloc[row].to_dict()
    user_id = dictionary['userId']
    movie_id = dictionary['movieId']
    if movie_id in all_movies:
        if user_id not in map_user_id_to_movies_rated:
            map_user_id_to_movies_rated[user_id] = set()
        map_user_id_to_movies_rated[user_id].add(movie_id)

all_movies_list = sorted(all_movies)
all_users_list = sorted(all_users)
max_dim = max(len(all_users), len(all_movies)) + 1
print(f'# of dimensions: {max_dim}')

from src.local_bandito import LocalBanditoSequence, LocalBanditoEnsemble, Strategy
from src.support_regressors import GradientDescentRegressor, DataPreProcessor

bandito_options = {
    'BayesLinearRegressor': {
        'model_type': LocalBanditoEnsemble.ModelType.BayesLinearRegressor,
        'laplace_approximation': False,
        'diagonal_covariance': False,
        'bayesian_model': True,
        'l2_regularization_constant': 1e-3,
        'data_pre_processor_options': {
            'use_interaction_features': True,
            'use_extended_interaction_features': True,
        }
    },
    'GradientDescentRegressor': {
        'model_type': LocalBanditoEnsemble.ModelType.GradientDescentRegressor,
        'hidden_layer_sizes': [],
        'incremental_model': True,
        'pass_sample_weight': False,
        'bayesian_model': False,
        'number_of_epochs': 200,  # 1 for policy_gradient
        'step_size': 0.05,
        'batch_normalization': False,
        'train_in_batches': True,
        'scale_loss_by_row_instead_of_batch': False,  # True for policy_gradient
        'batch_size': batch_size,
        'internal_batch_size': min(200, batch_size),
        'internal_model_type': GradientDescentRegressor.ModelType.dot_product if dot_product 
        else GradientDescentRegressor.ModelType.linear,
        'optimizer_type': GradientDescentRegressor.OptimizerType.adam,
        'final_activation_type': GradientDescentRegressor.ActivationType.sigmoid if implicit_feedback
        else GradientDescentRegressor.ActivationType.custom_range,
        'final_activation_custom_range': (0, 5),
        'use_numdifftools_instead_of_jax': False,
        'l2_regularization_constant': 0.001,
        'save_training_data_filename': None,  # 'GDR_training_data.pkl',
        'print_progress': True,
        'softmax_constant': 1.0,
        'determine_embedding_indices': True,  # this tells bandito to determine the embedding indices
        'number_of_embedding_dimensions': number_of_embedding_dimensions,
        'fifo_queue_size_for_embeddings': max_dim,
        'early_stopping_at_loss_fraction_per_epoch': 0.001,
        'total_time_to_train_in_seconds': 100,
        'data_pre_processor_options': {
            'use_interaction_features': True,
            'use_extended_interaction_features': True,
            'standard_scaling': True,
            'separated': True
        }
    }
}

# these have default values assigned within the function

action_feature_metadata = [{
    'name': 'item_id',
    'possible_values': DataPreProcessor.ProcessCategoricalType.embedding if dot_product else tuple(sorted(all_movies))
}]

context_feature_metadata = [{
    'name': 'user_id',
    'possible_values': DataPreProcessor.ProcessCategoricalType.embedding if dot_product else tuple(sorted(all_users))
}]

if include_genre_feature:
    for genre in all_genres:
        action_feature_metadata.append({'name': f'genre_{genre}', 'possible_values': [0, 1]})

agent = LocalBanditoEnsemble(

    action_feature_metadata=action_feature_metadata,
    context_feature_metadata=context_feature_metadata,

    fraction_of_choices_random=0.0,
    min_count_to_skip_unknown_score=5,
    min_global_count_to_skip_unknown_score=5,
    predict_on_all_models=False,
    prediction_distribution_sample_count=100,

    min_prior_counts_determined_at_top_level=True,
    action_feature_vectors_to_choose_from=None,
    acquisition_samples=1000,
    remove_previous_samples_during_acquisition=False,

    strategy=Strategy.thompson_sampling,
    # bayesian_model = True,
    softmax_constant=1.0,

    # for the trailing list used to remove duplicates during acquisition
    trailing_list_length=1000,
    coreset_trailing_list_length=100,

    # reinforcement learning options
    ignore_action_feature_vectors=False,
    # experience_replay_ratio = 0,
    # reservoir_sampler_size = 1000,
    # self_storage_filename = None,
    scipy_minimize_options=None,
    numdiff_gradient_options=None,
    numdiff_hessian_options=None,

    # Ensemble-specific options
    bandit_name='GradientDescentRegressor',
    training_model_names=None,
    efficient=True,
    number_of_outputs=1,
    bandit_options=bandito_options,
    validation_reservoir_size=0,
    number_of_cross_fold_validations=0,
    minimum_ratio_of_training_to_validation_data=5
)


if movielens:
    indices_train = [i for i in range(nrows_we_work_with) if np.random.uniform() < 0.7]
else:
    indices_train = [i for i in range(nrows_we_work_with) 
                     if not np.isnan(data.rating.iloc[i].to_dict()['train']) and data.rating.iloc[i].to_dict()['train']]
indices_valid = [i for i in range(nrows_we_work_with) if i not in indices_train]
index = {(data.rating.iloc[i].to_dict()['userId'], data.rating.iloc[i].to_dict()['movieId']): i for i in
         range(len(data.rating))}

if implicit_feedback:

    # originally I used pivot tables
    # from https://stackoverflow.com/questions/20104522/using-numpy-to-convert-user-item-ratings-into-2-d-array
    # pivot = data.rating.pivot(index='userId', columns='movieId', values='rating')
    # pivot = pivot.loc[all_users_list, all_movies_list]
    # pivot = pivot.clip(0, 1).fillna(0)

    list_of_dicts = list()
    for user_id in all_users:
        for movie_id in all_movies:
            interaction = False
            rating = None
            train = False
            valid = False
            if (user_id, movie_id) in index:
                interaction = True
                if index[(user_id, movie_id)] in indices_train:
                    train = True
                if index[(user_id, movie_id)] in indices_valid:
                    valid = True
                rating = data.rating.iloc[index[(user_id, movie_id)]].to_dict()['rating']
            list_of_dicts.append({'userId': user_id, 'movieId': movie_id,
                                  'interaction': interaction, 'rating': rating,
                                  'train': train, 'valid': valid})
    data.rating = pandas.DataFrame(list_of_dicts)
    indices_train = [i for i in range(len(data.rating)) if data.rating.iloc[i].to_dict()['train']]
    original_number_of_valid_indices = len(indices_valid)
    indices_valid = [i for i in range(len(data.rating)) if not data.rating.iloc[i].to_dict()['train']]
    np.random.shuffle(indices_valid)
    indices_valid = indices_valid[:original_number_of_valid_indices]


def get_action_feature_vector_from_dict(dictionary):
    if include_genre_feature:
        movie_id = dictionary['movieId']
        action_feature_vector = [movie_id]
        action_feature_vector.extend(map_movie_id_to_genre_feature_vector.get(movie_id, [0] * len(all_genres)))
        return action_feature_vector
    else:
        movie_id = dictionary['movieId']
        return [movie_id]


def get_context_feature_vector_from_dict(dictionary):
    user_id = dictionary['userId']
    return [user_id]


print('\ntraining...')
for row in tqdm(indices_train):
    dictionary = data.rating.iloc[row].to_dict()
    if implicit_feedback:
        output_value = dictionary['interaction']
        if output_value != 1:
            raise ValueError("whoa")
    else:
        output_value = dictionary['rating']
    if row in indices_train:
        agent.train(
            get_action_feature_vector_from_dict(dictionary),
            get_context_feature_vector_from_dict(dictionary),
            [output_value]
        )

agent.force_batch_training()

predictions_train = []
ground_truth_train = []
predictions_valid = []
ground_truth_valid = []
print('\nevaluating...')
for row in tqdm(indices_train + indices_valid):
    dictionary = data.rating.iloc[row].to_dict()
    prediction = agent.predict(
        get_action_feature_vector_from_dict(dictionary),
        get_context_feature_vector_from_dict(dictionary),
        force_return_score_even_with_few_updates=True
    )[0]
    if implicit_feedback:
        output_value = round(dictionary['interaction'])
    else:
        output_value = dictionary['rating']
    if row in indices_valid:
        if prediction != [None]:
            predictions_valid.append(prediction)
            ground_truth_valid.append(output_value)
    else:
        if prediction != [None]:
            predictions_train.append(prediction)
            ground_truth_train.append(output_value)

if implicit_feedback:
    from sklearn.metrics import average_precision_score

    print(f'Validation r2: {average_precision_score(ground_truth_valid, predictions_valid)}')
    print(f'Training r2: {average_precision_score(ground_truth_train, predictions_train)}')
    # print(
    #     f'Validation accuracy score: {accuracy_score(ground_truth_valid, [[round(y) for y in x] for x in predictions_valid])}')
    # print(
    #     f'Training accuracy: {accuracy_score(ground_truth_train, [[round(y) for y in x] for x in predictions_train])}')
else:
    from sklearn.metrics import r2_score

    print(f'Validation r2: {r2_score(ground_truth_valid, predictions_valid)}')
    print(f'Training r2: {r2_score(ground_truth_train, predictions_train)}')

if not movielens:
    print(f'data.rating:')
    for row in range(len(data.rating)):
        dictionary = data.rating.iloc[row].to_dict()
        prediction = float(agent.predict(
            get_action_feature_vector_from_dict(dictionary),
            get_context_feature_vector_from_dict(dictionary),
            force_return_score_even_with_few_updates=True
        )[0])
        dictionary.update({'prediction': prediction})
        dictionary.pop("train")
        print(f' {"T" if row in indices_train else " "} {dictionary}')
        

# Run this after predictions to load up the embeddings
print('Model parameters: ')
for key, val in agent.bandit_to_use.model.deterministic_model.model_parameters.items():
    print(f'{key}: {val}')
for key, val in agent.bandit_to_use.model.deterministic_model.model_parameters_FIFO_parallel.items():
    print(f'{key}')
    for sub_key, sub_val in val.items():
        print(f'  {sub_key}:')
        dictionary = sub_val.get()
        for sub_sub_key in sorted(dictionary.keys()):
            print(f'    {sub_sub_key}: {dictionary[sub_sub_key]}')
