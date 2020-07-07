import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU
from tensorflow.keras.utils import to_categorical

from cond_rnn import ConditionalRNN

TIME_STEPS = 50
NUM_CELLS = 256


class MySimpleModel(tf.keras.Model):
    def __init__(self, use_conditions):
        super(MySimpleModel, self).__init__()
        if use_conditions:
            self.cond = ConditionalRNN(NUM_CELLS, cell='GRU', dtype=tf.float32)
        else:
            # simple GRU.
            self.cond = GRU(NUM_CELLS)
        self.out = tf.keras.layers.Dense(units=1, activation='linear')

    def call(self, inputs, **kwargs):
        o = self.cond(inputs)
        o = self.out(o)
        return o


class Categorizer:

    def __init__(self, lst: list):
        self.uniq = sorted(set(lst))
        self.map = dict(zip(self.uniq, to_categorical(range(len(set(self.uniq)))).tolist()))

    def hello(self):
        pass


class Batcher:

    def __init__(self, x, y, cities, regions, city_conditions, region_conditions, batch_size, use_conditions):
        # shuffle.
        random.seed(123)
        indexes = list(range(x.shape[0]))
        random.shuffle(indexes)
        self.x = x[indexes]
        self.y = y[indexes]
        self.cities = cities[indexes]
        self.regions = regions[indexes]
        assert len(self.x) == len(self.y) == len(self.cities) == len(self.regions)
        self.city_conditions = city_conditions
        self.region_conditions = region_conditions
        self.batch_size = batch_size
        self.steps_per_epoch = self.x.shape[0] // self.batch_size
        self.use_conditions = use_conditions
        self.i = 0

    def generator(self):
        while True:
            if self.i == self.steps_per_epoch - 1:
                self.i = 0
            s = slice(self.i * self.batch_size, (self.i + 1) * self.batch_size)
            batch_x = self.x[s]
            batch_y = self.y[s]
            cities = self.cities[s]
            regions = self.regions[s]
            batch_cities = np.array([self.city_conditions.map[r] for r in cities])
            batch_regions = np.array([self.region_conditions.map[r] for r in regions])
            if self.use_conditions:
                x = batch_x, batch_cities, batch_regions
            else:
                x = batch_x
            y = batch_y
            self.i += 1
            yield x, y


def main():
    use_conditions = True
    data = np.load('city_temperature.npz')
    cities = data['cities']
    regions = data['regions']
    city_conditions = Categorizer(cities)
    region_conditions = Categorizer(regions)
    x = data['temp']  # (time, ts)
    temp = []
    for i in range(TIME_STEPS, x.shape[0]):
        temp.append(x[i - TIME_STEPS:i])
    temp = np.array(temp)  # (batch_size, time_steps, num_ts)

    temp = np.transpose(temp, axes=(2, 0, 1))

    cities = np.tile(np.expand_dims(cities, axis=-1), (1, temp.shape[1]))
    cities = cities.flatten()  # (NUM_SAMPLES,)

    regions = np.tile(np.expand_dims(regions, axis=-1), (1, temp.shape[1]))
    regions = regions.flatten()  # (NUM_SAMPLES,)

    temp = np.reshape(temp, (-1, TIME_STEPS, 1))  # (NUM_SAMPLES, TIME_STEPS, 1)

    x = temp[:, :-1, :]
    y = temp[:, -1:, :]

    model = MySimpleModel(use_conditions)
    batch_size = 512
    validation_split = 0.9
    vs = int(validation_split * x.shape[0])
    x_train = x[:vs]
    y_train = y[:vs]
    cities_train = cities[:vs]
    regions_train = regions[:vs]

    x_test = x[vs:]
    y_test = y[vs:]
    cities_test = cities[vs:]
    regions_test = regions[vs:]

    batcher_train = Batcher(x_train, y_train, cities_train, regions_train, city_conditions, region_conditions,
                            batch_size, use_conditions)
    batcher_test = Batcher(x_test, y_test, cities_test, regions_test, city_conditions, region_conditions, batch_size,
                           use_conditions)

    print(f'mean value loss test = {np.mean(np.abs(np.mean(x_test, axis=1).squeeze() - y_test.squeeze()))}')
    print(f'last value loss test = {np.mean(np.abs(x_test[:, -1, :].squeeze() - y_test.squeeze()))}')

    model.compile(optimizer='adam', loss='mae')
    model.fit(
        x=batcher_train.generator(),
        epochs=10,
        steps_per_epoch=batcher_train.steps_per_epoch,
        validation_steps=batcher_test.steps_per_epoch,
        validation_data=batcher_test.generator()
    )


if __name__ == '__main__':
    main()
