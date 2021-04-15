#!/usr/bin/env python
# coding: utf-8
#
# **Abstract:**
# A tensorflow LSTM network is used to predict the daily temperature
#  of Amsterdam.  
#  Without exogoneous components the best MAE is 1.46
#  degrees on the test set. Cond_rnn is able to get a MAE of 0.87
#  using the temperature in 30 neighbouring   
#  cities. A GPU is used to speed up calculations,
#  here [p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/).

# +
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
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
validation_split = 0  # we set to 0 for fair comparison with armax
window = 100
cities = 30         # neighbouring cities to include in cond_rnn vector
random_state = 123  # random state  fixed for similar result, ideal is average
df = temp.read_data(option='daily')
# -

# Again, I mainly look at the temperature in Amsterdam.

df_city = (df.droplevel(level=['region', 'country'])
             .unstack(level='date').T.sort_index()
           )
df_city.Amsterdam.head()

# See the notes on ARMA. Here the 30 most correlating
# temperatures are used as exogenous component. In my opinion,
# tensorflow should be better with multicollinearity.

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
#  the previous 100 values for the city of Amsterdam.
# For the other cities, only the previous daily temperature is used.
# Note, the code should also support multiple labels.

# +
LABELS = ['Amsterdam']
FEATURES = top_cities[1:]

def get_loc(lst):
    '''converts names to indices for columntransformer'''
    return [df_city.columns.get_loc(x) for x in lst]

ct = ColumnTransformer([
        ('LABELS', StandardScaler(), get_loc(LABELS)),
        ('FEATURES', StandardScaler(), get_loc(FEATURES))
    ], remainder='drop')
df_data = pd.DataFrame(ct.fit_transform(df_city),
                       columns=LABELS+FEATURES)
for lag in range(window):
    for label in LABELS:
        df_data.loc[:, f'x-{label}{lag+1}'] = df_data.Amsterdam.shift(lag+1)
df_data = df_data.dropna().sort_index()
# -

# The data is split in a train and test set. Shuffle is disabled to enable
#  comparison with ARMAX.

train, test = train_test_split(df_data, test_size=test_size, shuffle=False)

# Libraries are loaded and the data is reshaped.


def create_xy(data):
    'helper function to create x, c and y with proper shapes'
    x = data.filter(like='x-', axis=1).values[..., np.newaxis]
    c = data[FEATURES].to_numpy()
    y = data[LABELS].to_numpy()
    return x, c, y


# create correct shapes for tensorflow
x_train, c_train, y_train = create_xy(train)
x_test, c_test, y_test = create_xy(test)
# deterministic
set_seed(random_state)

#  As before, I start out by a pure autoregressive model.

model = Sequential(layers=[GRU(cells), Dense(units=len(LABELS), activation='linear')])
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

history = model_exog.fit(x=[x_train, c_train], y=y_train, epochs=epochs,
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
# The autoregressive component is fixed at a median value. The condition vector is varied  

# load JS visualization code to notebook
shap.initjs()
def predict(c):
    'the autoregressive values are fixed and the condition vector is varied '
    x_new = np.median(x_train)*np.ones((c.shape[0], x_train.shape[1], x_train.shape[2]))
    prediction = model_exog.predict([x_new, c])
    return prediction[:, 0]
predict(c_train)

# Shapley provides a deep explainer for neural networks and a kernel explainer.  
# I use a kernel explainer as it can be used for all problems and the deep explainer doesn't work with  
# the latest version of of Tensorflow.
# The kernel explainer is initialized using kmeans in the train set. The Shap values are obtained with  
# the test set. This procedure has simply been copied from website.

data = shap.kmeans(c_train, 30)

explainer = shap.KernelExplainer(predict, data=data)

# Calculating values can be slow.

test_data = shap.kmeans(c_test, 30)
shap_values = explainer.shap_values(test_data.data)

shap.force_plot(explainer.expected_value, shap_values[0], feature_names=FEATURES, link="logit")

# The following code, can be used to get the Shap values in a convenient dataframe.

df_shap = pd.Series(shap_values[0], index=FEATURES)
df_shap.sort_values(ascending=False)

# The model suffers from multi-collinairity. This result does not show that actually Brussel and Paris are very similar.
# The model is able to create a good prediction, but the insight is not good. Brussels and Paris should have the same impact.

df_cor[FEATURES][df_cor.index == LABELS[0]]


