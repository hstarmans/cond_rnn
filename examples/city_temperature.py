import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm as tqdm

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

d = pd.read_csv('city_temperature.csv', dtype=str)

city_to_region_map = dict(zip(d['City'].values, d['Region'].values))

d['AvgTemperature'] = d['AvgTemperature'].astype(float)
d = d[d['Year'].apply(len) == 4]
d = d[d['AvgTemperature'] != -99.0]
d = d[d['Day'].apply(lambda x: x.zfill(2)) != '00']

# to celsius
d['AvgTemperature'] = (d['AvgTemperature'] - 32) / 1.8

d['Date'] = d['Year'] + '-' + d['Month'].apply(lambda x: x.zfill(2)) + '-' + d['Day'].apply(lambda x: x.zfill(2))
d['Date'] = pd.to_datetime(d['Date'])
d.set_index('Date', inplace=True)
d.sort_index(inplace=True, ascending=True)
d = d[['Region', 'City', 'AvgTemperature']]
d.sort_values(['Region', 'City'], ascending=True, inplace=True)
print(d.head())

d_per_city = []
for city in tqdm.tqdm(sorted(set(d['City']))):
    db = d[d['City'] == city]['AvgTemperature']
    db.rename(city, inplace=True)
    db = db[~db.index.duplicated(keep='first')]
    d_per_city.append(db)

a = pd.concat(d_per_city, axis=1)
a.fillna(method='ffill', inplace=True)
a.fillna(method='bfill', inplace=True)

print(a.head())
print(a.tail())
print(len(a))

# subtract with the values 1 year ago.
# gross normalization. In practice separate it with train/test set first.
b = a.values[365:, :] - a.values[:-365, :]
b = (b - np.mean(b, axis=0)) / (b.std(axis=0) + 0.001)

plt.plot(b)
plt.show()

cities = np.array(a.columns, dtype=str)
regions = np.array([city_to_region_map[c] for c in cities.tolist()], dtype=str)

np.savez_compressed(
    file='city_temperature.npz',
    temp=np.array(b, dtype=np.float32),  # half the size and still precise.
    cities=cities,
    regions=regions,
)

# print(np.load('city_temperature.npz')['cities'])
# print(np.load('city_temperature.npz')['regions'])
# print(np.load('city_temperature.npz')['temp'].shape)
