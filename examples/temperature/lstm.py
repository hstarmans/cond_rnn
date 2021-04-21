#!/usr/bin/env python
# coding: utf-8
#
# **Abstract:**
# A tensorflow LSTM network is used to predict the daily temperature
#  of Amsterdam.
#  Without exogoneous components the best MAE is 1.46
#  degrees on the test set. Cond_rnn is able to get a MAE of 0.87
#  using the temperature in 30 neighbouring.
#  cities. A GPU is used to speed up calculations,
#  here [p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/).

# +
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from cond_rnn import ConditionalRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.python.framework.random_seed import set_seed
import numpy as np
import shap

import temp

# settings
cells = 200
epochs = 40
test_size = 0.2
pca_comp = 25
validation_split = 0  # we set to 0 for fair comparison with armax
window = 100
cities = 30         # neighbouring cities to include in cond_rnn vector
random_state = 123  # random state  fixed for similar result, ideal is average
df = temp.read_data(option='daily')
# -

# Again, I mainly look at the temperature in Amsterdam.

df_city = (df.droplevel(level=['region', 'country'])
             .unstack(level='date').T.sort_index()
             .dropna()
           )
df_city.Amsterdam.head()

# See the notes on ARMA. Here the 30 most correlating
# temperatures are used as exogenous component.

df_cor = df_city.corr()
df_cor.head()
# One more is grabbed as the most correlating city is Amsterdam itself
top_cities = (df_cor[
    df_cor.index == 'Amsterdam'].T
                                .nlargest(cities+1, ['Amsterdam'])
                                .index[0:cities+1].to_list()
)
df_data = (df_city[top_cities[1:]].shift(1)
                                  .assign(Amsterdam=df_city.Amsterdam)
                                  .dropna()
           )
df_data.columns = df_data.columns.astype(str)

# The dataset is transformed for machine learning.
# Temperature is standard scaled and an input x is generated which contains
# the previous 100 values for the city of Amsterdam.
# For the other cities, only the previous daily temperature is used.
# Note, the code should also support multiple labels.

LABELS = ['Amsterdam']
FEATURES = top_cities[1:]

# Earlier version of this notebook did not use PCA scaling.
# PCA increases the test mae of the model from 0.87 to 1.0 (not good) but
# improves the interpretation provided by Shapley values (good).

pca = PCA().fit(df_city[FEATURES].values)
plt.figure('PCA explained variance')
plt.grid()
plt.plot(list(range(1, len(df_city[FEATURES].columns)+1)),
         np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()
print(f"{pca_comp} components explain\
      {np.cumsum(pca.explained_variance_ratio_)[6]: .2f} of variance.")

# First, the PCA transformation is applied and hereafter the standard scaling.

pca_trafo = PCA(pca_comp).fit(df_city[FEATURES].values)
FEATURES_pca = [f'a{i}' for i in range(0, pca_comp)]
df_reduced = pd.DataFrame(pca_trafo.transform(df_city[FEATURES]),
                          index=df_city.index, columns=FEATURES_pca)
df_reduced[LABELS] = df_city[LABELS]


# +
def get_loc(lst):
    '''converts names to indices for columntransformer'''
    return [df_reduced.columns.get_loc(x) for x in lst]


ct = ColumnTransformer([
        ('LABELS', StandardScaler(), get_loc(LABELS)),
        ('FEATURES', StandardScaler(), get_loc(FEATURES_pca))
    ], remainder='drop')
df_data = pd.DataFrame(ct.fit_transform(df_reduced),
                       columns=LABELS+FEATURES_pca).dropna().sort_index()
for lag in range(window):
    for label in LABELS:
        df_data.loc[:, f'x-{label}{lag+1}'] = df_data[label].shift(lag+1)
df_data = df_data.dropna().sort_index()
# -

# The data is split in a train and test set. Shuffle is disabled to enable
#  comparison with ARMAX.

train, test = train_test_split(df_data, test_size=test_size, shuffle=False)

# The data is reshaped so it suited for tensorflow and cond_rnn.


def create_xy(data):
    'helper function to create x, c and y with proper shapes'
    x = data.filter(like='x-', axis=1).values[..., np.newaxis]
    c = data[FEATURES_pca].to_numpy()
    y = data[LABELS].to_numpy()
    return x, c, y


# create correct shapes for tensorflow
x_train, c_train, y_train = create_xy(train)
x_test, c_test, y_test = create_xy(test)
# deterministic
set_seed(random_state)

#  As before, I start out by a pure autoregressive model.

model = Sequential(layers=[GRU(cells),
                   Dense(units=len(LABELS), activation='linear')])
model.compile(optimizer='adam', loss='mae')
history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=None,
                    shuffle=True,
                    validation_split=validation_split)

# The final test loss is;


def inverseLABEL(data):
    return (ct.named_transformers_['LABELS']
              .inverse_transform(data)
            )


modelmae = mean_absolute_error(inverseLABEL(model.predict(x_test)),
                               inverseLABEL(y_test))
print(f"The MAE is {modelmae:.2f}")

# The above test loss is very similar to ARMA. Let's try to improve on this
#  estimate with an exogenous model.

print("WARNING: Install latest version of cond_rnn via git and not pip!")
model_exog = Sequential(layers=[ConditionalRNN(cells, cell='GRU'),
                                Dense(units=len(LABELS), activation='linear')])
model_exog.compile(optimizer='adam', loss='mae')

# Let's fit a model;

history = model_exog.fit(x=[x_train, c_train], y=y_train, epochs=28,
                         batch_size=None, shuffle=True,
                         validation_split=validation_split)

# The test loss for the exogenous model is;

exomae1 = mean_absolute_error(inverseLABEL(model_exog.predict([x_train,
                                                              c_train])),
                              inverseLABEL(y_train))
exomae2 = mean_absolute_error(inverseLABEL(model_exog.predict([x_test,
                                                              c_test])),
                              inverseLABEL(y_test))

print(f"The train MAE is {exomae1:.2f}")
print(f"The test MAE is {exomae2:.2f}")

# ARMAX also give insight in the relative importance of FEATURES.
# To provide, this with cond_rnn I use Shapley plots.
# The autoregressive component is fixed at a median value.
# The condition vector is varied

# load JS visualization code to notebook
shap.initjs()


def predict(data):
    'the autoregressive values are fixed and the condition vector is varied '
    pca_data = pd.DataFrame(pca_trafo.transform(data), columns=FEATURES_pca)
    pca_data[LABELS] = 1
    # NOTE: this line makes reverse complicated
    std_data = pd.DataFrame(ct.transform(pca_data),
                            columns=LABELS+FEATURES_pca).dropna().sort_index()
    c = std_data[FEATURES_pca].to_numpy()
    x_new = np.median(x_train)*np.ones((c.shape[0],
                                       x_train.shape[1], x_train.shape[2]))
    prediction = inverseLABEL(model_exog.predict([x_new, c]))
    return prediction[:, 0]


predict(df_city[FEATURES])

# Shapley provides a deep explainer for neural networks and a kernel explainer.
# I use a kernel explainer as it can be used for all problems
# and the deep explainer doesn't work with the latest version of of Tensorflow.
# The kernel explainer is initialized using kmeans in the train set.
# The Shap values are obtained with the test set.
# This procedure has simply been copied from website.

data = shap.kmeans(df_city[FEATURES].to_numpy(), 30)

explainer = shap.KernelExplainer(predict, data=data)

# Calculating values can be slow.

test_data = shap.kmeans(df_city[FEATURES].to_numpy(), 30)
shap_values = explainer.shap_values(test_data.data)

shap.force_plot(explainer.expected_value, shap_values[0],
                feature_names=FEATURES)  # link="logit")

# The following code, can be used to get the Shap values
# in a convenient dataframe.

df_shap = pd.Series(shap_values[0], index=FEATURES)
df_shap.sort_values(ascending=False)
