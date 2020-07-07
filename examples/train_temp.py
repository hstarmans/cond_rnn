import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random
from cond_rnn import ConditionalRNN

TIME_STEPS = 50
NUM_CELLS = 24


class MySimpleModel(tf.keras.Model):
    def __init__(self):
        super(MySimpleModel, self).__init__()
        self.cond = ConditionalRNN(NUM_CELLS, cell='GRU', dtype=tf.float32)
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

    def __init__(self, x, y, cities, regions, city_conditions, region_conditions, batch_size):
        # shuffle.
        random.seed(123)
        indexes = list(range(x.shape[0]))
        random.shuffle(indexes)

        self.x = x[indexes]
        self.y = y[indexes]
        self.cities = cities[indexes]
        self.regions = regions[indexes]
        self.city_conditions = city_conditions
        self.region_conditions = region_conditions
        self.batch_size = batch_size
        self.steps_per_epoch = self.x.shape[0] // self.batch_size

    def generator(self):
        for i in range(self.steps_per_epoch):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            batch_x = self.x[s]
            batch_y = self.y[s]
            cities = self.cities[s]
            regions = self.regions[s]
            batch_cities = np.array([self.city_conditions.map[r] for r in cities])
            batch_regions = np.array([self.region_conditions.map[r] for r in regions])
            x = batch_x, batch_cities, batch_regions
            y = batch_y
            i += 1
            yield x, y


def main():
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

    model = MySimpleModel()

    batcher = Batcher(x, y, cities, regions, city_conditions, region_conditions, 32)
    generator = batcher.generator()

    model.compile(optimizer='adam', loss='mae')
    model.fit(
        x=generator,
        epochs=10,
        steps_per_epoch=batcher.steps_per_epoch
    )


if __name__ == '__main__':
    main()
