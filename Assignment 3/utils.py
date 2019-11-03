import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
import keras.backend as K
from keras import Input, Model, Sequential
from keras.layers import Dense, Lambda, BatchNormalization, Dropout, Concatenate
from keras.losses import mse


def normalize(x):
    return (x - x.mean()) / x.std()


def import_wine(n_samples=40000, n_test_samples=5000, y_transform=None):
    dataset1 = pd.read_csv('../wine-reviews/winemag-data-130k-v2.csv')[:n_samples + n_test_samples]
    dataset2 = pd.read_csv('../wine-reviews/winemag-data_first150k.csv')[:n_samples + n_test_samples]

    x = pd.concat([dataset1, dataset2], ignore_index=True, sort=False)

    del dataset1
    del dataset2

    x = x[x['country'] != '']
    x = x[x['province'] != '']
    x = x[x['variety'] != '']
    x = x[x['description'] != '']

    x = x[['country', 'price', 'province', 'variety', 'description', 'points']]
    x = x.dropna()
    y = x[['points']]
    x = x[['country', 'price', 'province', 'variety', 'description']]

    x = pd.get_dummies(x, columns=['country', 'province', 'variety'])
    x['description'] = x.apply(lambda s: len(s['description']), axis=1)
    x['description'] = (x['description'] - x['description'].min()) / (x['description'].max() - x['description'].min())

    x['price'] = normalize(x['price'])
    x['description'] = normalize(x['description'])

    y['points'] = pd.cut(y['points'], 5, labels=[k for k in range(5)])

    if y_transform == 'to_categorical':
        from keras.utils import to_categorical
        y = to_categorical(y)
    elif y_transform == 'get_dummies':
        y = pd.get_dummies(y, columns=['points'])

    x_train = x[:n_samples]
    x_test = x[n_samples:n_samples + n_test_samples]
    y_train = np.reshape(y.values[:n_samples], n_samples)
    y_test = np.reshape(y.values[n_samples:n_samples + n_test_samples], n_test_samples)

    return x_train, y_train, x_test, y_test


def import_adult(use_to_categorical=False, n_samples=40000, n_test_samples=5000):
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '>50k']

    train = pd.read_csv('../adult/adult-data.csv', names=names)[:n_samples + n_test_samples]
    test = pd.read_csv('../adult/adult-test.csv', names=names)[:n_samples + n_test_samples]
    test = test.drop([0])

    whole = train.append(test)
    for key in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'native-country']:
        whole = whole[~whole[key].str.contains('[?]')]

    x = whole.drop('>50k', 1)
    y = whole[['>50k']]

    x = pd.get_dummies(
        x,
        columns=[
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'native-country'
        ]
    ).astype('float32')

    tmp = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None

    values = []
    for value in y.values:
        if value not in values:
            values.append(value)

    y['>50k'] = pd.factorize(y['>50k'])[0]

    pd.options.mode.chained_assignment = tmp

    if use_to_categorical:
        from keras.utils import to_categorical
        y = to_categorical(y)

    # for key in x.keys():
    #     x[key] = normalize(x[key])

    x['age'] = normalize(x['age'])
    x['fnlwgt'] = normalize(x['fnlwgt'])
    x['education-num'] = normalize(x['education-num'])
    x['capital-gain'] = normalize(x['capital-gain'])
    x['capital-loss'] = normalize(x['capital-loss'])
    x['hours-per-week'] = normalize(x['hours-per-week'])

    x_train = x[:n_samples]
    x_test = x[n_samples:n_samples + n_test_samples]
    y_train = np.reshape(y.values[:n_samples], n_samples)
    y_test = np.reshape(y.values[n_samples:n_samples + n_test_samples], n_test_samples)

    return x_train, y_train, x_test, y_test


def display_cm(matrix, title, xlabel, ylabel, xparamval, yparamval):
    matrix = np.array(matrix).transpose()

    plot.style.use('default')
    fig, ax = plot.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap='magma_r')
    ax.figure.colorbar(im, ax=ax)

    matrix = matrix.transpose()

    ax.set(xticks=np.arange(matrix.shape[0]),
           yticks=np.arange(matrix.shape[1]),
           xticklabels=xparamval, yticklabels=yparamval,
           title=title,
           ylabel=ylabel,
           xlabel=xlabel)

    plot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(i,
                    j,
                    format(int(matrix[i, j]), fmt),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > thresh else "black"
                    )
    fig.tight_layout()

    np.set_printoptions(precision=2)

    plot.show()


intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


INTERMEDIATE_DIM = 128


def create_vae(original_dim, latent_dim=2):
    input_shape = (original_dim,)

    inputs = Input(shape=input_shape)
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model(inputs, z)

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(INTERMEDIATE_DIM, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    decoder = Model(latent_inputs, outputs)

    outputs = decoder(encoder(inputs))
    vae = Model(inputs, outputs)

    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return encoder, vae


def create_wine_model(input_dim, output_dim):
    model = Sequential()

    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def create_mi_wine_model(input_dims, output_dim):
    classic_inputs = Input(shape=(input_dims[0],))
    cluster_inputs = Input(shape=(input_dims[1],))

    concatenated = Concatenate()([classic_inputs, cluster_inputs])

    x = Dense(10, activation='relu')(concatenated)
    x = Dense(15, activation='relu')(x)
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model([classic_inputs, cluster_inputs], outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def create_adult_model(input_dim, output_dim):
    model = Sequential()

    model.add(Dense(15, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(15, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def create_mi_adult_model(input_dims, output_dim):
    classic_inputs = Input(shape=(input_dims[0],))
    cluster_inputs = Input(shape=(input_dims[1],))

    concatenated = Concatenate()([classic_inputs, cluster_inputs])

    x = Dense(15, activation='relu')(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(15, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    outputs = Dense(output_dim, activation='softmax')(x)

    model = Model([classic_inputs, cluster_inputs], outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model
