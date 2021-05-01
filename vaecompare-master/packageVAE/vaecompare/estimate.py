# ----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------

from __future__ import division

import itertools
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch.utils import data

from .utils import (loss_function, Collate)


class VAE():
    """
    Estimate univariate density using Bayesian Fourier Series.
    This estimator only works with data the lives in
    [0, 1], however, the class implements estimators to automatically
    transform user inputted data to [0, 1]. See parameter `transform`
    below.

    Parameters
    ----------
    ncomponents : integer
        Maximum number of components of the Fourier series
        expansion.

    nn_weight_decay : object
        Mulplier for penalizaing the size of neural network weights. This penalization occurs for training only (does not affect score estimator nor validation of early stopping).

    num_layers : integer
        Number of hidden layers for the neural network. If set to 0, then it degenerates to linear regression.
    hidden_size : integer
        Multiplier for the size of the hidden layers of the neural network. If set to 1, then each of them will have ncomponents components. If set to 2, then 2 * ncomponents components, and so on.

    es : bool
        If true, then will split the training set into training and validation and calculate the validation internally on each epoch and check if the validation loss increases or not.
    es_validation_set_size : float, int
        Size of the validation set if es == True, given as proportion of train set or as absolute number. If None, then `round(min(x_train.shape[0] * 0.10, 5000))` will be used.
n_train = x_train.shape[0] - n_test
    es_give_up_after_nepochs : float
        Amount of epochs to try to decrease the validation loss before giving up and stoping training.
    es_splitter_random_state : float
        Random state to split the dataset into training and validation.

    nepoch : integer
        Number of epochs to run. Ignored if es == True.

    batch_initial : integer
        Initial batch size.
    batch_step_multiplier : float
        See batch_inital.
    batch_step_epoch_expon : float
        See batch_inital.
    batch_max_size : float
        See batch_inital.

    batch_test_size : integer
        Size of the batch for validation and score estimators.
        Does not affect training efficiency, usefull when there's
        little GPU memory.
    gpu : bool
        If true, will use gpu for computation, if available.
    verbose : integer
        Level verbosity. Set to 0 for silent mode.
    """

    def __init__(self,

                 num_layers_decoder=10,
                 hidden_size_decoder=100,
                 dropout_decoder=0.5,

                 num_layers_encoder=10,
                 hidden_size_encoder=100,
                 dropout_encoder=0.5,

                 latent_dim=20,

                 batch_initial=100,
                 batch_step_multiplier=1.4,
                 batch_step_epoch_expon=2.0,
                 batch_max_size=1000,
                 dataloader_workers=1,
                 initial_lr=0.01,

                 nn_weight_decay=0,
                 es=True,
                 es_validation_set_size=None,
                 es_give_up_after_nepochs=15,
                 es_splitter_random_state=0,
                 float_type="float",

                 distribution="gaussian",

                 batch_test_size=2000,
                 gpu=True,
                 verbose=1,
                 ):

        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

    def fit(self, y_train, nepoch=100):
        assert (y_train is not None)

        self.lr = self.initial_lr

        if len(y_train.shape) == 1:
            y_train = y_train[:, None]

        self.gpu = self.gpu and torch.cuda.is_available()

        self.y_dim = y_train.shape[1]

        self._construct_neural_net()
        self.epoch_count = 0

        self.move(self.gpu, self.float_type)

        return self.improve_fit(y_train=y_train, nepoch=nepoch)

    def improve_fit(self, y_train=None, nepoch=100):
        assert (y_train is not None)

        if len(y_train.shape) == 1:
            y_train = y_train[:, None]

        dataset = y_train

        criterion = loss_function

        assert (self.batch_initial >= 1)
        assert (self.batch_step_multiplier > 0)
        assert (self.batch_step_epoch_expon > 0)
        assert (self.batch_max_size >= 1)
        assert (self.batch_test_size >= 1)

        assert (self.num_layers_decoder >= 0)
        assert (self.hidden_size_decoder > 0)
        assert (self.num_layers_encoder >= 0)
        assert (self.hidden_size_encoder > 0)

        ninstances = len(dataset)

        range_epoch = range(nepoch)
        if self.es:
            es_validation_set_size = self.es_validation_set_size
            if es_validation_set_size is None:
                es_validation_set_size = round(
                    min(ninstances * 0.10, 5000))
            permutation = np.random.permutation(range(ninstances))
            index_val = permutation[:es_validation_set_size]
            index_train = permutation[es_validation_set_size:]
            dataset_val = data.Subset(dataset, index_val)
            dataset_train = data.Subset(dataset, index_train)

            self.best_loss_val = np.infty
            es_tries = 0
            range_epoch = itertools.count()  # infty iterator

            batch_test_size = min(self.batch_test_size,
                                  len(dataset_val))
            self.loss_history_validation = []
        else:
            dataset_train = dataset

        batch_max_size = min(self.batch_max_size, len(dataset_train))
        self.loss_history_train = []

        start_time = time.time()

        optimizer = optim.Adamax(self.neural_net.parameters(),
                                 lr=self.lr,
                                 weight_decay=self.nn_weight_decay)
        es_penal_tries = 0
        err_count = 0
        for epoch_id in range_epoch:
            batch_size = int(min(batch_max_size,
                                 self.batch_initial +
                                 self.batch_step_multiplier *
                                 self.epoch_count ** self.batch_step_epoch_expon))

            try:
                self.neural_net.train()
                self._one_epoch(True, batch_size, dataset_train,
                                optimizer, criterion)

                if self.es:
                    self.neural_net.eval()
                    avloss = self._one_epoch(False, batch_test_size,
                                             dataset_val, optimizer, criterion)
                    self.loss_history_validation.append(avloss)
                    if avloss < self.best_loss_val:
                        self.best_loss_val = avloss
                        best_state_dict = self.neural_net.state_dict()
                        best_state_dict = deepcopy(best_state_dict)
                        es_tries = 0
                        if self.verbose >= 2:
                            print("This is the lowest validation loss",
                                  "so far.", flush=True)
                    else:
                        es_tries += 1

                    if (es_tries == self.es_give_up_after_nepochs // 3
                            or
                            es_tries == self.es_give_up_after_nepochs // 3
                            * 2):
                        if self.verbose >= 2:
                            print("Decreasing learning rate by half.",
                                  flush=True)
                        optimizer.param_groups[0]['lr'] *= 0.5
                        self.neural_net.load_state_dict(best_state_dict)
                    elif es_tries >= self.es_give_up_after_nepochs:
                        self.neural_net.load_state_dict(best_state_dict)
                        if self.verbose >= 1:
                            print("Validation loss did not improve after",
                                  self.es_give_up_after_nepochs, "tries.",
                                  "Stopping", flush=True)
                        break

                self.epoch_count += 1
            except RuntimeError as err:
                err_count += 1
                if err_count >= 100:
                    raise err
                print(err)
                print("Runtime error problem probably due to",
                      "high learning rate.")
                print("Decreasing learning rate by half.",
                      flush=True)

                self._construct_neural_net()
                self.move(self.gpu, self.float_type)
                self.lr /= 2
                optimizer = optim.Adamax(self.neural_net.parameters(),
                                         lr=self.lr, weight_decay=self.nn_weight_decay)
                self.epoch_count = 0
                self.best_loss_val = np.infty

                continue
            except KeyboardInterrupt:
                if self.epoch_count > 0 and self.es:
                    print("Keyboard interrupt detected.",
                          "Switching weights to lowest validation loss",
                          "and exiting", flush=True)
                    self.neural_net.load_state_dict(best_state_dict)
                break

        self.elapsed_time = time.time() - start_time
        if self.verbose >= 1:
            print("Elapsed time:", self.elapsed_time, flush=True)

        return self

    def _one_epoch(self, is_train, batch_size, dataset, optimizer,
                   criterion):
        with torch.set_grad_enabled(is_train):
            loss_vals = []
            batch_sizes = []

            collate_fn = Collate(float_type=self.float_type)

            data_loader = data.DataLoader(dataset,
                                          batch_size=batch_size, shuffle=True, drop_last=is_train,
                                          pin_memory=self.gpu,
                                          num_workers=self.dataloader_workers,
                                          collate_fn=collate_fn)

            for inputv in data_loader:
                if self.gpu:
                    inputv = inputv.cuda(non_blocking=True)

                batch_actual_size = inputv.shape[0]
                optimizer.zero_grad()
                output = self.neural_net(inputv)
                loss = criterion(output, inputv, self.distribution)

                np_loss = loss.data.item()
                if np.isnan(np_loss):
                    raise RuntimeError("Loss is NaN")

                loss_vals.append(np_loss)
                batch_sizes.append(batch_actual_size)

                if is_train:
                    loss.backward()
                    optimizer.step()

            avgloss = np.average(loss_vals, weights=batch_sizes)
            if self.verbose >= 2:
                print("Finished epoch", self.epoch_count,
                      "with batch size", batch_size, "and",
                      ("train" if is_train else "validation"),
                      "loss", avgloss, flush=True)

            return avgloss

    def reconstruction_loss(self, y_test):
        assert (y_test is not None)

        if len(y_test.shape) == 1:
            y_test = y_test[:, None]

        with torch.no_grad():
            dataset = y_test
            self.neural_net.eval()

            batch_size = min(self.batch_test_size, len(dataset))

            loss_vals = []
            batch_sizes = []

            collate_fn = Collate(float_type=self.float_type,
                                 variable_length=self.variable_length,
                                 attmec=self.attmec)

            data_loader = data.DataLoader(dataset,
                                          batch_size=batch_size, shuffle=True, drop_last=False,
                                          pin_memory=self.gpu, collate_fn=collate_fn,
                                          num_workers=self.dataloader_workers, )

            for inputv in data_loader:
                if self.gpu:
                    inputv = inputv.cuda(non_blocking=True)

                batch_actual_size = inputv.shape[0]
                output = self.neural_net(inputv)
                loss = criterion(output, inputv, self.distribution)
                loss = loss.cpu().numpy().mean()
                loss_vals.append(loss.item())
                batch_sizes.append(batch_actual_size)

            if self.nclasses is None:
                return -1 * np.average(loss_vals, weights=batch_sizes)
            else:
                return 1 - np.average(loss_vals, weights=batch_sizes)

    def sample_parameters(self, size=1):
        with torch.no_grad():
            self.neural_net.eval()

            z_sample = np.random.normal(size=(size, self.latent_dim))

            z_sample = torch.FloatTensor(z_sample)
            if self.gpu:
                z_sample = z_sample.cuda()

            res = self.neural_net.decode(z_sample)
            if self.distribution == "gaussian":
                return res[0].cpu().numpy(), res[1].cpu().numpy()
            elif self.distribution == "bernoulli":
                return res.cpu().numpy()

    def sample_y(self, size=1):
        with torch.no_grad():
            res = self.sample_parameters(size)
            if self.distribution == "gaussian":
                mu, logvar = res
                return stats.norm.rvs(mu, np.exp(0.5 * logvar))
            elif self.distribution == "bernoulli":
                return stats.bernoulli.rvs(res)

    def _construct_neural_net(self):
        class NeuralNet(nn.Module):
            def __init__(self, y_dim, lantent_size,
                         num_layers_decoder, hidden_size_decoder,
                         dropout_decoder,

                         num_layers_encoder, hidden_size_encoder,
                         dropout_encoder,

                         distribution,
                         ):
                super(NeuralNet, self).__init__()

                self.distribution = distribution

                # Encoder
                self.dropl = nn.Dropout(p=dropout_encoder)
                llayers = []
                out_size = hidden_size_encoder
                in_size = y_dim
                for i in range(num_layers_encoder):
                    llayer = nn.Linear(in_size, out_size)
                    self._initialize_layer_linear(llayer)
                    llayers.append(llayer)
                    llayers.append(nn.ELU())
                    llayers.append(nn.BatchNorm1d(out_size))
                    llayers.append(self.dropl)
                    in_size = hidden_size_encoder
                self.layers_enc = nn.Sequential(*llayers)

                self.lenc_mean = nn.Linear(in_size, lantent_size)
                self.lenc_logvar = nn.Linear(in_size, lantent_size)

                # Decoder
                self.dropl = nn.Dropout(p=dropout_decoder)
                llayers = []
                out_size = hidden_size_decoder
                in_size = lantent_size
                for i in range(num_layers_decoder):
                    llayer = nn.Linear(in_size, out_size)
                    self._initialize_layer_linear(llayer)
                    llayers.append(llayer)
                    llayers.append(nn.ELU())
                    llayers.append(nn.BatchNorm1d(out_size))
                    llayers.append(self.dropl)
                    in_size = hidden_size_decoder
                self.layers_dec = nn.Sequential(*llayers)

                self.ldec_mean = nn.Linear(in_size, y_dim)
                if self.distribution == "gaussian":
                    self.ldec_logvar = nn.Linear(in_size, y_dim)

            def encode(self, y):
                h1 = self.layers_enc(y)
                return self.lenc_mean(h1), self.lenc_logvar(h1)

            def reparameterize(self, mu, logvar):
                std = logvar.mul(0.5).exp_()
                eps = std.data.new(std.size()).normal_()
                return eps.mul(std).add_(mu)

            def decode(self, z):
                h3 = self.layers_dec(z)
                if self.distribution == "gaussian":
                    return self.ldec_mean(h3), self.ldec_logvar(h3)
                elif self.distribution == "bernoulli":
                    return torch.sigmoid(self.ldec_mean(h3))

            def forward(self, y):
                mu_enc, logvar_enc = self.encode(y)
                z = self.reparameterize(mu_enc, logvar_enc)
                if self.distribution == "gaussian":
                    mu_dec, logvar_dec = self.decode(z)
                    return mu_dec, logvar_dec, mu_enc, logvar_enc
                elif self.distribution == "bernoulli":
                    theta_dec = self.decode(z)
                    return theta_dec, mu_enc, logvar_enc

            def _initialize_layer_linear(self, layer):
                nn.init.constant_(layer.bias, 0)
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_normal_(layer.weight, gain=gain)

        self.neural_net = NeuralNet(self.y_dim,
                                    self.latent_dim,

                                    self.num_layers_decoder,
                                    self.hidden_size_decoder,
                                    self.dropout_decoder,

                                    self.num_layers_encoder,
                                    self.hidden_size_encoder,
                                    self.dropout_encoder,

                                    self.distribution,
                                    )

    def move(self, gpu=None, float_type=None):
        if gpu is None:
            pass
        elif gpu:
            self.neural_net.cuda()
            self.gpu = True
        else:
            self.neural_net.cpu()
            self.gpu = False

        if float_type is None:
            pass
        elif float_type == "half":
            self.neural_net.half()
        elif float_type == "float":
            self.neural_net.float()
        elif float_type == "double":
            self.neural_net.double()
        else:
            raise ValueError("Invalid float_type.")

        return self

    def __getstate__(self):
        d = self.__dict__.copy()
        if hasattr(self, "neural_net"):
            state_dict = self.neural_net.state_dict()
            for k in state_dict:
                state_dict[k] = state_dict[k].cpu()
            d["neural_net_params"] = state_dict
            del (d["neural_net"])

        return d

    def __setstate__(self, d):
        self.__dict__ = d

        if "neural_net_params" in d.keys():
            self._construct_neural_net()
            self.neural_net.load_state_dict(self.neural_net_params)
            del (self.neural_net_params)
            self.move(float_type=self.float_type)
            if self.gpu:
                if torch.cuda.is_available():
                    self.move(gpu=True)
                else:
                    self.gpu = False
                    print("Warning: GPU was used to train this model, "
                          "but is not currently available and will "
                          "be disabled "
                          "(renable with estimator move_to_gpu)",
                          flush=True)
