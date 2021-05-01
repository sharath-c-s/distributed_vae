import numpy as np
import tensorflow as tf
from IPython import display

from VAE import VAE


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
batch_size = 32
test_size = 10000
SPLIT_SIZE = 10

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size))
train_size, test_size = int(train_size / SPLIT_SIZE), int(test_size / SPLIT_SIZE)
X_train, X_test = list(), list()
for i in range(0, SPLIT_SIZE):
    X_train.append(
        train_images[int((i * len(train_images) / SPLIT_SIZE)):(int((i + 1) * len(train_images) / SPLIT_SIZE))])
    X_test.append(
        test_images[int((i * len(test_images) / SPLIT_SIZE)):(int((i + 1) * len(test_images) / SPLIT_SIZE))])
for i in range(0, SPLIT_SIZE):
    X_train[i] = (tf.data.Dataset.from_tensor_slices(X_train[i]).shuffle(train_size).batch(batch_size))
    X_test[i] = (tf.data.Dataset.from_tensor_slices(X_test[i]).shuffle(test_size).batch(batch_size))


def FLVAE(epochs, latent_dim, X_train, X_test, train_dataset):
    model = VAE(latent_dim)
    for i in range(0, epochs):
        models = []
        elbo = 0
        for j in range(0, SPLIT_SIZE):
            print('Running data on node ', j + 1)
            current_model = model

            for train_x in X_train[j]:
                model.train_step(current_model, train_x, model.optimizer)
            loss = tf.keras.metrics.Mean()
            for test_x in X_test[j]:
                loss(model.compute_loss(current_model, test_x))
            elbo = -loss.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, clien-num: {}'.format(i, elbo, j))
            # generate_and_save_images(current_model, i, j, test_sample)

            models.append(current_model)

        weights = [model.get_weights() for model in models]
        new_weights = list()
        for weights_list_tuple in zip(*weights):
            new_weights.append(np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))

        model.set_weights(new_weights)
        # write code for accuracy

    loss = tf.keras.metrics.Mean()
    for train_x in train_dataset:
        loss(model.compute_loss(model, train_x))
    print(loss.result())


def DFLVAE(epochs, latent_dim, X_train, X_test, train_dataset):
    model = VAE(latent_dim)
    models = []
    for j in range(0, SPLIT_SIZE):
        print('Running data on node ', j + 1)
        current_model = model
        for train_x in X_train[j]:
            model.train_step(current_model, train_x, model.optimizer)
        loss = tf.keras.metrics.Mean()
        for test_x in X_test[j]:
            loss(model.compute_loss(current_model, test_x))
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, clien-num: {}'.format(0, elbo, j))
        # generate_and_save_images(current_model, i, j, test_sample)
        models.append(current_model)

    for i in range(1, epochs):
        elbo = 0
        weights = [model.get_weights() for model in models]
        print("dddddd", len(weights))
        models = []
        for j in range(0, SPLIT_SIZE):
            weights_here = weights[:j] + weights[j + 1:]
            print("fffffffff", len(weights))
            new_weights = []
            for weights_list_tuple in zip(*weights_here):
                new_weights.append(np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))
            print('Running data on node ', j + 1)
            current_model = model
            current_model.set_weights(new_weights)
            for train_x in X_train[j]:
                model.train_step(current_model, train_x, model.optimizer)
            loss = tf.keras.metrics.Mean()
            for test_x in X_test[j]:
                loss(model.compute_loss(current_model, test_x))
            elbo = -loss.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, clien-num: {}'.format(i, elbo, j))
            # generate_and_save_images(current_model, i, j, test_sample)
            models.append(current_model)

        # write code for accuracy

    # loss = tf.keras.metrics.Mean()
    # for train_x in train_dataset:
    #     loss(model.compute_loss(model, train_x))
    # print(loss.result())


def GFLVAE(epochs, latent_dim, X_train, X_test, train_dataset):
    model = VAE(latent_dim)
    models = []
    for j in range(0, SPLIT_SIZE):
        print('Running data on node ', j + 1)
        current_model = model
        for train_x in X_train[j]:
            model.train_step(current_model, train_x, model.optimizer)
        loss = tf.keras.metrics.Mean()
        for test_x in X_test[j]:
            loss(model.compute_loss(current_model, test_x))
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, clien-num: {}'.format(0, elbo, j))
        # generate_and_save_images(current_model, i, j, test_sample)
        models.append(current_model)

    for i in range(1, epochs):
        elbo = 0
        weights = [model.get_weights() for model in models]
        models = []
        for j in range(0, SPLIT_SIZE):
            weights_here = [weights[j], weights[(j + 1) % 10]]
            new_weights = []
            for weights_list_tuple in zip(*weights_here):
                new_weights.append(np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)]))
            print('Running data on node ', j + 1)
            current_model = model
            current_model.set_weights(new_weights)
            for train_x in X_train[j]:
                model.train_step(current_model, train_x, model.optimizer)
            loss = tf.keras.metrics.Mean()
            for test_x in X_test[j]:
                loss(model.compute_loss(current_model, test_x))
            elbo = -loss.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, clien-num: {}'.format(i, elbo, j))
            # generate_and_save_images(current_model, i, j, test_sample)
            models.append(current_model)


epochs = 10
latent_dim = 2

GFLVAE(epochs, latent_dim, X_train, X_test, train_dataset)
