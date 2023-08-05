import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization
import statistics
import numpy as np
import scipy.stats
import statsmodels.api as sm
from scipy.stats import shapiro
import pandas as pd
import jupyterthemes as jt
import researchpy as rp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from collections import defaultdict
from matplotlib.cm import ScalarMappable
from scipy.stats import linregress
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import GammaRegressor
from sklearn._loss.glm_distribution import (
    TweedieDistribution,
    NormalDistribution, PoissonDistribution,
    GammaDistribution, InverseGaussianDistribution,
)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from numpy import arange
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from yellowbrick.regressor import cooks_distance
from scipy.stats import binom
sns.set(style="ticks", context="talk")
plt.style.use('classic')
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import chi2
from tensorflow.keras import regularizers
from keras import backend as K
from keras.losses import Loss
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from numpy.random import seed
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.ensemble import IsolationForest

# ==================== #
#   Database loading   #
# ==================== #

# Read the CSV file into a DataFrame (data_preprocessed is the data file that has been filtered with correlation analysis and cleaned by interpolation)
data = pd.read_csv('data_preprocessed.csv') 

# Drop the 'Unnamed: 0' column
data.drop('Unnamed: 0', axis=1, inplace=True)

# Create a new DataFrame without the text columns ('title', 'emp_title', 'desc')
data_without_text  = data.loc[:, ~data.columns.isin(['title', 'emp_title', 'desc'])]

# Map 'loan_status' values to binary labels: 'Charged Off' -> 1, 'Fully Paid' -> 0
data_without_text['loan_status'] = data_without_text['loan_status'].map({'Charged Off':1, 'Fully Paid':0})

# Identify columns containing 'month' or 'year' in their names
cols_to_drop = data_without_text.filter(like='month').columns.tolist() + \
              data_without_text.filter(like='year').columns.tolist()
              
# Use drop() to drop the selected columns
data_without_text = data_without_text.drop(columns=cols_to_drop)

# =============================================== #
#   One-Hot encoding, no dummies for localGLMnet  #
# =============================================== #

# Define the columns to be one-hot encoded
to_convert = ['emp_length','term', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'pub_rec_bankruptcies']

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse = False)

# Fit the encoder on the specified columns in the DataFrame
encoder.fit(data_without_text[to_convert])
encoded_columns = encoder.transform(data_without_text[to_convert])

# Create a DataFrame from the encoded columns with appropriate column names
encoded_df = pd.DataFrame(encoded_columns, columns = encoder.get_feature_names_out(to_convert))

data_without_text = data_without_text.drop(columns=to_convert)

# Concatenate the original DataFrame with the encoded DataFrame
data_without_text = pd.concat([data_without_text, encoded_df], axis=1)

# ============================================= #
#   Outlier analysis with the IsolationForest   #
# ============================================= #

# Remove recoveries from the analysis as it's clear that a defaulting contract will have recoveries > 0
data_outliers = data_without_text.loc[:, ~data_without_text.columns.isin(['recoveries'])]

num_features = data_outliers.columns[:12].tolist()

# Grid search for contamination parameter (but naive approach as it's not based on the true anomaly rate, see further)
param_grid = {'contamination': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]}

data_gs = data_outliers.loc[:,num_features].drop("loan_status", axis = 1)

######## ------------------- COMPUTE THE CONTAMINATION RATE ------------------- ########


# Contamination rate estimation based on the average outliers rate in the dataset with IQR computation
iqr_dict = {}
for col in data_gs.columns:
    q1 = data_gs[col].quantile(0.25)
    q3 = data_gs[col].quantile(0.75)
    iqr = q3 - q1
    iqr_dict[col] = iqr
    
# Define a function to count outliers in a column
def count_outliers(col):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    outlier_range = (q1 - 1.5*iqr, q3 + 1.5*iqr)
    outliers = col[(col < outlier_range[0]) | (col > outlier_range[1])]
    return outliers.count()

# Apply the count_outliers function to each numeric column and compute the total number of outliers
num_rows = data_gs.shape[0]
total_outliers = 0
for col in data_gs.columns:
    col_outliers = count_outliers(data_gs[col])
    total_outliers += col_outliers

# Compute the mean percentage of outliers in the dataset
mean_outlier_rate = total_outliers / (len(data_gs.columns) * num_rows) * 100
print('Mean percentage of outliers: {:.2f}%'.format(mean_outlier_rate))


######## ------------------- ISOLATION FOREST ------------------- ########

# As it was naive to perform GS, use the IQR contamination estimator (0.046)
clf = IsolationForest(contamination=0.046)

# Create a dictionary to store the percentages of anomalies for each feature and loan_status level
anomalies_dict = {}

# Loop through each feature in the dataset (except for the 'loan_status' feature)
for feature in num_features:
    if feature != 'loan_status':
        # Fit the isolation forest model to the feature data
        clf.fit(data_outliers[[feature]])
        # Predict outliers using the model and create a contingency table
        y_pred = clf.predict(data_outliers[[feature]])
        contingency_table = pd.crosstab(index=y_pred, columns=data_outliers['loan_status'], normalize='columns')
        # Store the percentages of anomalies for each feature and loan_status level in the dictionary
        anomalies_dict[feature] = contingency_table.loc[-1].tolist()

# Create a list of labels for the x-axis
x_labels = [feature + '\n(0, 1)' for feature in anomalies_dict.keys()]

# Create a list of percentages of anomalies for each loan_status level
level0 = [anomalies_dict[feature][0] for feature in anomalies_dict.keys()]
level1 = [anomalies_dict[feature][1] for feature in anomalies_dict.keys()]

# Define the width of the bars
bar_width = 0.4

sns.set_style('whitegrid')
# Create the bar chart
fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(np.arange(len(x_labels))*2, level0, width=bar_width, label='Loan Status 0', color='lightblue')
ax.bar(np.arange(len(x_labels))*2+bar_width, level1, width=bar_width, label='Loan Status 1', color='darkblue')
ax.set_xticks(np.arange(len(x_labels))*2+bar_width/2)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_title('Percentage of Anomalies Detected by Isolation Forest')
ax.set_xlabel('Feature (Loan Status)')
ax.set_ylabel('Percentage of Anomalies')
ax.legend(loc='upper left')
ax.grid(True, axis='y')
plt.show()


# ========================================== #
#   Standardization fot continuous features  #
# ========================================== #

### -------------- MIN - MAX scaling -------------- ###

# ! If you prefer Standardization scaling, then skip this part, else run it
# Create a function to scale the features and store the initial ranges of features in a dict as we will need to unscale later
def var_standardization(data, variable, min_max_dict):
    y = "data." + variable
    x = eval(y)
    x_scaled = (x - min(x)) / (max(x) - min(x))
    name = "Scaled_" + variable
    data = data.loc[:, ~data.columns.isin([variable])]
    data[name] = x_scaled
    min_max_dict[variable] = {"min": min(x), "max": max(x)}
    return data, min_max_dict

# min-max scale all the continuous feautres and store in dic
min_max_dict = {}
data_without_text, min_max_dict = var_standardization(data_without_text, "loan_amnt", min_max_dict)
data_without_text, min_max_dict = var_standardization(data_without_text, "int_rate", min_max_dict)
data_without_text, min_max_dict = var_standardization(data_without_text, "annual_inc", min_max_dict)
data_without_text, min_max_dict = var_standardization(data_without_text, "dti", min_max_dict)
data_without_text, min_max_dict = var_standardization(data_without_text, "delinq_2yrs", min_max_dict)
data_without_text, min_max_dict = var_standardization(data_without_text, "last_pymnt_amnt", min_max_dict)
data_without_text, min_max_dict = var_standardization(data_without_text, "inq_last_6mths", min_max_dict)
data_without_text, min_max_dict = var_standardization(data_without_text, "open_acc", min_max_dict)
data_without_text, min_max_dict = var_standardization(data_without_text, "revol_bal", min_max_dict)
data_without_text, min_max_dict = var_standardization(data_without_text, "revol_util", min_max_dict)
data_without_text, min_max_dict = var_standardization(data_without_text, "recoveries", min_max_dict)
data_without_text, min_max_dict = var_standardization(data_without_text, "total_rec_late_fee", min_max_dict)
num_names = ["loan_amnt","int_rate","annual_inc","dti","delinq_2yrs","last_pymnt_amnt","recoveries","inq_last_6mths","open_acc","revol_bal","revol_util","total_rec_late_fee", "random_uniform", "random_gaussian"]


### -------------- Standard Scaler scaling -------------- ###

# ! If you prefer min-max scaling, then skip this part, else run it
# Create random features for noise introduction, bad fit for the LocalGLMnet, but follows the procedure of the W & R paper
rand_u = np.random.uniform(size=data_without_text.shape[0])
data_without_text["random_uniform"] = rand_u
rand_g = np.random.normal(size=data_without_text.shape[0])
data_without_text["random_gaussian"] = rand_g

scaler = StandardScaler()
scaler.fit(data_without_text[num_names])
scaled_data_standard = scaler.transform(data_without_text[num_names])
data_without_text[num_names] = scaled_data_standard
unscaled_data = scaler.inverse_transform(scaled_data_standard)
unscaled_df = pd.DataFrame(unscaled_data, columns=num_names)

# ============================= #
#   Binomial Deviance recoding  #
# ============================= #

class Deviance(keras.losses.Loss):
    def __init__(self, eps=1e-16, w=1, **kwargs):
        self.eps = eps
        self.w = w
        super(Deviance, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true = float(y_true)
        return -2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))

# ============================================================================================================================================== #
#   Keras model construction without month|year (feel free to keep these features in the previous pre-processing if you want to use panel data)  #
# ============================================================================================================================================== #

np.random.seed(87031800)

### -------------- Keras model -------------- ###


### -------------- create random control variables -------------- ###

data_without_my = data_without_text.copy()

# Exemple of the way you can introduce random features for noise introduction (skip it if min max scaling)
rand_u = np.random.uniform(size=data_without_my.shape[0])
data_without_my["random_uniform"] = rand_u
data_without_my, min_max_dict = var_standardization(data_without_my, "random_uniform", min_max_dict)

# Train test split
X = data_without_my.drop("loan_status", axis= 1)
X = X.drop('Scaled_recoveries', axis=1)
Y = data_without_my["loan_status"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=87031800)

# Calibration matrix for the HW test
calib_test = pd.concat([Xtest, Ytest], axis = 1)
calib_train = pd.concat([Xtrain, Ytrain], axis = 1)

### -------------- Keras model with GS & CV for different depth -------------- ###

# In this section we will try different configurations of LocalGLMnets to analyze convergences
import random
import tensorflow as tf

# initialization of different depth and number of neurons
depths = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
neurons = [5, 10, 15, 20, 30, 45, 50, 100, 128]

# Create a figure for the plots
fig1, ax1 = plt.subplots(figsize=(10, 8), facecolor="white")
fig2, ax2 = plt.subplots(figsize=(10, 8), facecolor="white")

# Loop through each combination of depth and neuron count
for i, depth in enumerate(depths):
    print(i, depth)
    # Build the model with the specified depth and neuron count
    inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name='Input_var')
    dense = inp_num
    chosen_neurons = []
    for k in range(depth):
        n = random.choice(neurons)
        dense = keras.layers.Dense(n, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.01), name=f'dense_{k}')(dense)
        chosen_neurons.append(n)
    attention = keras.layers.Dense(Xtrain.shape[1], activation='linear', name='Attention')(dense)
    dot = keras.layers.Dot(axes=1)([inp_num, attention])
    response = keras.layers.Dense(1, activation='sigmoid', name='Response')(dot)
    model = keras.Model(inputs=inp_num, outputs=response)
    model.compile(loss=Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.001))

    # Fit the model and record the history
    history = model.fit(Xtrain, Ytrain, epochs=200, batch_size=len(Xtrain), validation_split=0.2, verbose=2)

    # Plot the train and validation loss on the corresponding plot with the same color
    color = f'#{random.randint(0, 0xFFFFFF):06x}'
    label = f'Depth={depth}, Neurons={chosen_neurons}'
    ax1.plot(history.history['loss'], color=color, linestyle='-')
    ax2.plot(history.history['val_loss'], color=color, linestyle='-', label=label)

# Add a legend to the plots
ax1.legend()
ax2.legend()
ax1.grid()
ax2.grid()
# Set the axis labels and title of the plots
ax1.set_title('Training Deviance for Different Architectures')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Deviance Train')

ax2.set_title('Validation Deviance for Different Architectures')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Deviance Test')

# Show the plots
plt.show()


### -------------- Keras model with GS & CV for fixed size -------------- ###

# Naive Grid Search with the RandomizedSearchCV, first step before trying the stat of art Hyperband algorithm
# Feel free to change the LocalGLmnet parameters to check the hyperparameters retained
# In what follows, we will fine tune the parameters of the LocalGLMnet as well as number of neurons, depth
# We will choose to perform Full batch training in the next sections (read Master Thesis of Warnauts Aymeric to understand to advantages of this fixed parameters)
# But for instance we can check different batch sizes if you prefer stochastic batch learning

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer


# Create a function that return the LocalGlmnet architecture
def create_model(learning_rate=0.01, batch_size=100, epochs=200, activation1='relu', activation2='relu', activation3='relu', activation4='relu', activation5='relu', l2=0.01):
    inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name='Input_var')

    dense1 = keras.layers.Dense(45, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(l2), name='first')(inp_num)
    dense2 = keras.layers.Dense(20, activation=tf.nn.relu, name='second')(dense1)
    dense3 = keras.layers.Dense(10, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(l2), name='third')(dense2)
    dense4 = keras.layers.Dense(20, activation=tf.nn.relu, name='fourth')(dense3)
    dense5 = keras.layers.Dense(45, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(l2), name='fifth')(dense4)

    Attention = keras.layers.Dense(Xtrain.shape[1], activation="linear", name='Attention')(dense5)
    Dot = keras.layers.Dot(axes=1)([inp_num, Attention])
    Response = keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='Response')(Dot)

    model = keras.Model(inputs=inp_num, outputs=Response)
    model.compile(loss=Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"])
    return model

# Usefull to discriminate btw diff arch.
model = KerasClassifier(build_fn=create_model, verbose=2)

# Initialization of hyperparameter space
param_dist = {
    'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
    'batch_size': [20, 50, 100, 150],
    'epochs': [50, 100, 150, 200, 300],
    'activation1': ['relu'],
    'activation2': ['relu'],
    'activation3': ['relu'],
    'activation4': ['relu'],
    'activation5': ['relu'],
    'l2': [0.01, 0.001, 0.0001, 0.00001]
}

# Introduce the Deviance loss function
custom_scorer = make_scorer(Deviance(eps=1e-5, w=1), greater_is_better=False)

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    cv=10,
    random_state=87031800,
    scoring=custom_scorer
)

# train the architectures
random_search_result = random_search.fit(Xtrain, Ytrain)

# Return the parameters that lead to the lowest Deviance score 
print("Best: %f using %s" % (random_search_result.best_score_, random_search_result.best_params_))

### -------------- State of Art Hyperband hyperparameter tuner, tuning all parameters -------------- ###

import keras_tuner
from kerastuner import HyperParameters
from tabulate import tabulate
from tensorflow.keras import layers

def build_model(hp):
    inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name='Input_var')
    x = inp_num
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 6)):
        x = layers.Dense(
            units=hp.Int(f"units_{i}", min_value=5, max_value=65, step=1),
            activation=hp.Choice("activation", ["relu"]),
            kernel_regularizer=regularizers.l2(hp.Float(f"l2_reg_{i}", 0.0, 0.1, step=0.01)),
        )(x)
    Attention = layers.Dense(Xtrain.shape[1], activation="linear", name='Attention')(x)
    Dot = layers.Dot(axes=1)([inp_num, Attention])
    Response = layers.Dense(1, activation=tf.keras.activations.sigmoid, name='Response')(Dot)
    # Tune the learning rate
    learning_rate = hp.Choice("learning_rate", values=[1e-5, 1e-4, 1e-3])
    model = keras.Model(inputs=inp_num, outputs=Response)

    # Compile the model with the custom loss function
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=Deviance(eps=1e-05, w=1)
    )
    
    
    return model
  
# Define the hyperparameter search space (see the Hyperband doc to understand the choice of hyperparameters 
# as it allows to compute the max number of models and thus the approx computation time)
tuner = keras_tuner.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=700,
    factor=3,
    seed = 87031800,
    directory="tuning_dir", # you can retrieve all the trials in this directory
    project_name="my_model_months", # and in this file 
    hyperband_iterations=3
)

# Start the hyperparameter search with full batch size and quite large early stopping, you can modify these depending on you computation power and time available
tuner.search(Xtrain, Ytrain, epochs=700, validation_split=0.2, batch_size=len(Xtrain), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=20)])

# return the best hyperparameters in a latex tabel
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0].values
print(f"Number of Dense Layers: {best_hps.get('num_layers')}")
print(f"Number of Units in Dense Layers: {[best_hps.get(f'units_{i}') for i in range(best_hps.get('num_layers'))]}")
print(f"Activation Function: {best_hps.get('activation')}")
print(f"L2 Regularization: {[best_hps.get(f'l2_reg_{i}') for i in range(best_hps.get('num_layers'))]}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")
print(f"epochs number: {best_hps.get('epochs')}")
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
best_model_epochs = best_trial.epochs
# Get the best model and evaluate it on the test set
best_model = tuner.get_best_models(num_models=1)[0]

table = []
for key, value in best_hps.items():
    table.append([key, value])
    
latex_tabular = tabulate(table, headers=['Hyperparameter', 'Value'], tablefmt='latex')

print(latex_tabular)




### -------------- Final Keras model for standardized -------------- ### Run the following section if you decided to standardize, in other case skip it 

# set random seed to the same of the hyperband one if you want reproductible results
tf.keras.utils.set_random_seed(87031800)

# fill the architecture with your optimal parameters 
inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name = 'Input_var')
dense1 = keras.layers.Dense(65, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.07), name = '1')(inp_num)
dense2 = keras.layers.Dense(23, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.08), name = '2')(dense1)
dense3 = keras.layers.Dense(34, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.04), name = '3')(dense2)
dense4 = keras.layers.Dense(31, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.07), name = '4')(dense3)
dense5 = keras.layers.Dense(59, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.07), name = '5')(dense4)
Attention = keras.layers.Dense(Xtrain.shape[1], activation="linear", name = 'Attention')(dense5)

Dot = keras.layers.Dot(axes=1)([inp_num, Attention])

Response = keras.layers.Dense(1, activation= tf.keras.activations.sigmoid, name = 'Response')(Dot)
model = keras.Model(inputs=inp_num, outputs=Response)
print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
model.compile(loss= Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.001))
history = model.fit(Xtrain, Ytrain, epochs=700, batch_size = len(Xtrain), callbacks = [callback], validation_split=0.2, verbose=2)

# Plot to check the smoothness of the convergence in Deviance loss for Train and Validation sets
plt.figure(figsize=(7,6), facecolor="white")
plt.plot(history.history['loss'], color='blue', linestyle='-')
plt.plot(history.history['val_loss'], color='red', linestyle='-')
plt.title('Deviance convergences with optimal hyperparameters')
plt.ylabel('Deviance')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.grid()
plt.show()


### -------------- Final Keras model for min max_scaled -------------- ###

tf.keras.utils.set_random_seed(87031800)

inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name = 'Input_var')
dense1 = keras.layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.1), name = '1')(inp_num)
dense2 = keras.layers.Dense(23, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.08), name = '2')(dense1)
dense3 = keras.layers.Dense(59, activation=tf.nn.relu,  name = '3')(dense2)
dense4 = keras.layers.Dense(19, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.04), name = '4')(dense3)
dense5 = keras.layers.Dense(39, activation=tf.nn.relu,  name = '5')(dense4)
dense6 = keras.layers.Dense(51, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01), name = '6')(dense5)
Attention = keras.layers.Dense(Xtrain.shape[1], activation="linear", name = 'Attention')(dense6)

Dot = keras.layers.Dot(axes=1)([inp_num, Attention])

Response = keras.layers.Dense(1, activation= tf.keras.activations.sigmoid, name = 'Response')(Dot)
model = keras.Model(inputs=inp_num, outputs=Response)
print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
model.compile(loss= Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.001))
history = model.fit(Xtrain, Ytrain, epochs=1000, batch_size = len(Xtrain), callbacks = [callback], validation_split=0.2, verbose=2)

plt.figure(figsize=(7,6), facecolor="white")
plt.plot(history.history['loss'], color='blue', linestyle='-')
plt.plot(history.history['val_loss'], color='red', linestyle='-')
plt.title('Deviance convergences with optimal hyperparameters')
plt.ylabel('Deviance')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.grid()
plt.show()

### -------------- model Calibration test -------------- ###

# In this section we will build the statistics and plots for the Calibration
to_see = model.predict(Xtest)
pred_calib = pd.DataFrame(to_see, columns=["pred"])
to_see = np.column_stack((to_see, Ytest))
to_mean = to_see[to_see[:, 1] == 1]
mean_saturated = np.mean(to_see[:, 0]) # compute mu Test set to be compared with the observed one

y_pred = to_see[:,0]
y_true = to_see[:,1]
deviance = -2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))
deviance_saturated = deviance.numpy() # compute Deviance for the Test set 

# Pay attention that each time you run the following lines you will have to recreate the calib Test and Calib train such as in our Train test split
calib_test = calib_test.reset_index(drop=True)
pred_calib = pred_calib.reset_index(drop=True)
calib_test = pd.concat([calib_test, pred_calib], axis = 1)


def my_fun(x):
    return np.mean(x)

# groups for the calibration, feel free to change the number of groups determined by Categorical binary features
groups = calib_test.columns[:14].tolist()

to_comp1 = calib_test.groupby(groups)['loan_status'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)
to_comp2 = calib_test.groupby(groups)['pred'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)

# HW calibration statistics
O = to_comp1['my_fun']*to_comp1['count']
E = to_comp2['my_fun']*to_comp2['count']
n = to_comp1['count']
HLT = np.sum((O - E)**2/(E*(1-to_comp2['my_fun']) + 1e-10))
pval = 1 - chi2.cdf(HLT, df=(len(to_comp1)-2))

plt.figure(figsize=(7,6), facecolor="white")
plt.scatter(to_comp2['my_fun'], to_comp1['my_fun'], c='blue')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Predicted Frequency')
plt.ylabel('True Frequency')
plt.title('Group Calibration (LocalGLMnet)')
plt.legend([r'$\hat{{C}}$ = {:.3f}'.format(HLT), 'p-value = {:.3f}'.format(pval)], loc='upper left')
plt.grid()

plt.savefig('calibrationtestglmnet.png')

### -------------- model Calibration train -------------- ###

# idem train set, rerun the calibration set init

to_see2 = model.predict(Xtrain)
pred_calib_train = pd.DataFrame(to_see2, columns=["pred"])
to_see2 = np.column_stack((to_see2, Ytrain))
to_mean = to_see2[to_see2[:, 1] == 1]
np.mean(to_mean[:, 0])

calib_train = calib_train.reset_index(drop=True)
pred_calib_train = pred_calib_train.reset_index(drop=True)
calib_train = pd.concat([calib_train, pred_calib_train], axis = 1)

to_comp1 = calib_train.groupby(groups)['loan_status'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)
to_comp2 = calib_train.groupby(groups)['pred'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)

O = to_comp1['my_fun']*to_comp1['count']
E = to_comp2['my_fun']*to_comp2['count']
n = to_comp1['count']
HLT = np.sum((O - E)**2/(E*(1-to_comp2['my_fun'])))
pval = 1 - chi2.cdf(HLT, df=(len(to_comp1)-2))

plt.figure(figsize=(7,6), facecolor="white")
plt.scatter(to_comp2['my_fun'], to_comp1['my_fun'], c='blue')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Predicted Frequency')
plt.ylabel('True Frequency')
plt.title('Group Calibration (LocalGLMnet)')
plt.legend([r'$\hat{{C}}$ = {:.3f}'.format(HLT), 'p-value = {:.3f}'.format(pval)], loc='upper left')
plt.grid()

plt.savefig('calibrationtrainglmnet.png')

### -------------- Beta Coefficient extraction -------------- ###

# surrogate model to get the trained regression attentions
zz = keras.Model(inputs = model.inputs, outputs = model.get_layer("Attention").output)
print(zz.summary())
b0 = float(model.get_layer("Response").get_weights()[0]) 
beta_x = zz.predict(Xtrain)
beta_x = beta_x * b0  
beta_x_df = pd.DataFrame(beta_x)
beta_x_df.columns = ['Beta_' + col_name for col_name in Xtrain.columns]
beta_x_df.to_csv('beta_x.csv', index=False) # stock the beta if you want to analyze the regression attentions in R

# concat coeff and plots
Xtrain = Xtrain.reset_index(drop=True)

coef_plot_train = pd.concat([Xtrain, beta_x_df], axis=1)

# sample to plot  (feel free to get less points)
nsample = len(Xtrain)
np.random.seed(87031800)
idx = np.random.choice(coef_plot_train.shape[0], nsample, replace=False)

beta_to_plot = coef_plot_train.iloc[idx]
train_to_plot = Xtrain.iloc[idx]

# define the line size and quant_rand values
line_size = 1
quant_rand = 0.5

### -------------- Beta Coefficient extraction for standarscaled -------------- ###

# If you choose min-max scaling then skip this part 
num_names.append("random")
name_train = [col for col in Xtrain.columns if any(name in col for name in num_names)]
name_beta = [col for col in beta_x_df.columns if any(name in col for name in num_names)]
ranges = unscaled_df.max() - unscaled_df.min()
ranges = ranges.drop("recoveries")
unscaled_beta = beta_x_df[name_beta].copy()
index_list = ranges.index.tolist()
unscaled_beta.columns = index_list

for col in unscaled_beta.columns:
    unscaled_beta[col] = unscaled_beta[col]/(ranges[col])
    
unscaled_df = unscaled_df.drop("recoveries", axis=1)
unscaled_df_train, unscaled_df_test = train_test_split(unscaled_df, test_size=0.2, random_state=87031800)

for col_name, col_data in unscaled_df_train.items():
    if col_name in unscaled_beta.columns:
        plt.figure(figsize=(8, 6), facecolor = "white")
        plt.scatter(col_data, unscaled_beta[col_name], color="white", edgecolor="black")
        plt.xlabel(col_name)
        plt.ylabel('Regression attention')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.show()
        

np.std(unscaled_beta["random_uniform"])


### -------------- Beta Coefficient extraction for min max -------------- ###

# get the list of feature names and beta feature names
name_train = [col.replace('Scaled_', '') for col in Xtrain.columns if col.startswith('Scaled_')]
name_beta = [col for col in beta_x_df.columns if any(name in col for name in num_names)]

# loop over each feature and beta feature pair
for feature, beta_feature in zip(name_train, name_beta):
    # unscale the data
    x_unscaled = Xtrain["Scaled_" + feature] * (min_max_dict[feature]["max"] - min_max_dict[feature]["min"]) + min_max_dict[feature]["min"]
    beta_unscaled = beta_x_df[beta_feature] / (min_max_dict[feature]["max"] - min_max_dict[feature]["min"])
    # create a data frame with the data to plot
    dat_plt = pd.DataFrame({'var': x_unscaled,
                            'bx': beta_unscaled,
                            'col': ['black'] * nsample})
    # create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 6), facecolor = "white")
    sns.scatterplot(data=dat_plt, x='var', y='bx', color="white", edgecolor="black",  sizes=[1], ax=ax)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.axvspan(xmin=min(dat_plt['var']), xmax=max(dat_plt['var']), ymin=-quant_rand, ymax=quant_rand, alpha=0.002, facecolor='green')
    ax.set(xlabel=feature, 
           ylabel="regression attention")
    
    # add a smooth line using UnivariateSpline
    from scipy.interpolate import UnivariateSpline
    sorted_data = dat_plt.sort_values('var')
    spl = UnivariateSpline(sorted_data['var'], sorted_data['bx'], k=5)
    x_new = np.linspace(sorted_data['var'].min(), sorted_data['var'].max(), 300)
    smooth_bx = spl(x_new)
    ax.plot(x_new, smooth_bx, color='purple', linewidth=2.5, linestyle=(0, (3, 1, 1, 1, 1, 1)))
    
    # show the plot
    fig.show()
    
# Create a list of the categorical columns
categorical_columns = ['emp_length', 'home_ownership', "term", 'verification_status', 'purpose', 'addr_state', 'pub_rec_bankruptcies']

for col in categorical_columns:
    # Create a list of the beta columns that correspond to the current categorical column
    beta_cols = [c for c in beta_x_df.columns if col in c]
    
    # Pivot the table to create a DataFrame with 2 columns: the beta values and the corresponding label
    beta_values = beta_x_df[beta_cols]
    beta_values = beta_values.melt(var_name='Category', value_name='Regression attention')
    
    # Modify the x-axis labels
    beta_values['Category'] = beta_values['Category'].apply(lambda x: x.split('_')[-1])
    
    # Create the boxplot
    plt.figure(figsize=(6,5), facecolor="white")
    ax = sns.boxplot(x='Category', y='Regression attention', data=beta_values, palette="bright", showfliers=False, width=0.3)
    plt.title(f'Beta values for {col}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.show()


### -------------- Deviance test-------------- ###

# The first algortihm will only remove one feature for all features and check which one as to be removed based on the LRT

X = data_without_my.drop("loan_status", axis= 1)
X = X.drop('Scaled_recoveries', axis=1)
Y = data_without_my["loan_status"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=87031800)
calib_test = pd.concat([Xtest, Ytest], axis = 1)
calib_train = pd.concat([Xtrain, Ytrain], axis = 1)

tf.keras.utils.set_random_seed(87031800)

# Create a list of non-binary column names to remove
non_binary_cols = [col for col in Xtrain.columns]

# Create an empty dictionary to store the deviance values for each removed feature
deviance_dict = {}

from scipy.stats import chi2

# Add the deviance of the saturated model to the dictionary
deviance_dict['saturated'] = (deviance_saturated, mean_saturated, "-", "-", "-")
deviance_current = 100000 # set suffisently high for the starting val
deviances = []
rapport = min(deviance_dict.values())[0] - deviance_current

# Loop over the non-binary columns and remove one at a time
for col in non_binary_cols:
    # Remove the current column from the input matrix
    reduced_Xtrain = Xtrain.drop(col, axis=1)
    reduced_Xtest = Xtest.drop(col, axis=1)
    tf.keras.utils.set_random_seed(87031800)
    # Train the optimal model on the reduced input matrix
    inp_num = keras.layers.Input(shape=(reduced_Xtrain.shape[1],), name='Input_var')
    dense1 = keras.layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.1), name='1')(inp_num)
    dense2 = keras.layers.Dense(23, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.08), name='2')(dense1)
    dense3 = keras.layers.Dense(59, activation=tf.nn.relu, name='3')(dense2)
    dense4 = keras.layers.Dense(19, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.04), name='4')(dense3)
    dense5 = keras.layers.Dense(39, activation=tf.nn.relu, name='5')(dense4)
    dense6 = keras.layers.Dense(51, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01), name='6')(dense5)
    attention = keras.layers.Dense(reduced_Xtrain.shape[1], activation='linear', name='Attention')(dense6)
    dot = keras.layers.Dot(axes=1)([inp_num, attention])
    response = keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='Response')(dot)
    model = keras.Model(inputs=inp_num, outputs=response)
    
    # Compile and fit the model with Deviance loss
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    model.compile(loss=Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.001))
    history = model.fit(reduced_Xtrain, Ytrain, epochs=1000, batch_size=len(reduced_Xtrain), callbacks=[callback], validation_split=0.2, verbose=2)
    
    # Compute the deviance of the reduced model on the test set
    to_see = model.predict(reduced_Xtest)
    pred_calib = pd.DataFrame(to_see, columns=["pred"])
    to_see = np.column_stack((to_see, Ytest))
    to_mean = to_see[to_see[:, 1] == 1]
    mean = np.mean(to_see[:, 0])

    y_pred = to_see[:,0]
    y_true = to_see[:,1]
    deviance = -2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))
    deviance_value = deviance.numpy()
    
    # Compute the test statistic and p-value
    test_statistic = 2 * (deviance_saturated - deviance_value)
    p_value = chi2.cdf(test_statistic, 1)
    if p_value < 0.05:
        indic = 'to include'
    else: 
        indic = 'not to include'
    
    # Store the results in the dictionary with the removed feature as key
    deviance_dict[col] = (deviance_value, mean, test_statistic, p_value, indic)

# stock directely the result values in a latex table
table = [[key, value[0], value[1], value[2], value[3], value[4] if len(value) >= 5 else None] for key, value in deviance_dict.items()]
print(tabulate(table, headers=['Variable', 'Deviance', 'default frequaency', " LRT", "p-value at 0.05", " "], tablefmt='latex_booktabs'))

# This improved algortihm will check all combinasion of features to be removed and will return the optimal subset of features 
# Read the Master Thesis of Warnauts Aymeric if you want to understand the procedure better, idem as Stepwise selection proc
# initialize variables
deviances = []
remaining_cols = non_binary_cols.copy()
saturated_deviance = deviance_saturated
current_deviance = 100000
best_deviance = 100000
best_removed_col = None
num_vars_saturated = len(Xtrain.columns)
num_vars_reduced = num_vars_saturated

# start the while loop
while len(remaining_cols) > 0:
    num_vars_reduced = num_vars_reduced - 1
    print(f"Number of features: {num_vars_reduced}")
    print(f"best deviance: {best_deviance}")
    # iterate through all remaining columns
    for col in remaining_cols:
        print(f"Column removed inner loop: {col}")
        # remove the current column from the input matrix
        reduced_Xtrain = Xtrain.drop(col, axis=1)
        reduced_Xtest = Xtest.drop(col, axis=1)
        
        tf.keras.utils.set_random_seed(87031800)
        # Train a model on the reduced input matrix
        inp_num = keras.layers.Input(shape=(reduced_Xtrain.shape[1],), name='Input_var')
        dense1 = keras.layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.1), name='1')(inp_num)
        dense2 = keras.layers.Dense(23, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.08), name='2')(dense1)
        dense3 = keras.layers.Dense(59, activation=tf.nn.relu, name='3')(dense2)
        dense4 = keras.layers.Dense(19, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.04), name='4')(dense3)
        dense5 = keras.layers.Dense(39, activation=tf.nn.relu, name='5')(dense4)
        dense6 = keras.layers.Dense(51, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01), name='6')(dense5)
        attention = keras.layers.Dense(reduced_Xtrain.shape[1], activation='linear', name='Attention')(dense6)
        dot = keras.layers.Dot(axes=1)([inp_num, attention])
        response = keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='Response')(dot)
        model = keras.Model(inputs=inp_num, outputs=response)
        
        # Compile and fit the model with Deviance loss
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        model.compile(loss=Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.001))
        history = model.fit(reduced_Xtrain, Ytrain, epochs=1000, batch_size=len(reduced_Xtrain), callbacks=[callback], validation_split=0.2, verbose=0)
        
        # Compute the deviance of the reduced model on the test set
        to_see = model.predict(reduced_Xtest)
        pred_calib = pd.DataFrame(to_see, columns=["pred"])
        to_see = np.column_stack((to_see, Ytest))
        to_mean = to_see[to_see[:, 1] == 1]
        mean = np.mean(to_see[:, 0])

        y_pred = to_see[:,0]
        y_true = to_see[:,1]
        deviance = -2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))
        deviance_value = deviance.numpy()
        print(f"col already removed: {deviances}")
        print(f"deviance value when removing: {col, deviance_value}")
        # Compute the test statistic and p-value
        test_statistic = 2 * (deviance_saturated - deviance_value)
        p_value = chi2.cdf(test_statistic, num_vars_saturated - num_vars_reduced)
        if p_value < 0.05:
            indic = 'to include'
        else: 
            indic = 'not to include'
        
        # store the results in the dictionary with the removed feature as key
        deviance_dict[col] = (deviance_value, mean, test_statistic, p_value, indic)
        
        # check if the deviance of the reduced model is better than the current best deviance
        if deviance_value < best_deviance:
            best_deviance = deviance_value
            best_removed_col = col
    
    # check if the deviance of the best reduced model is better than the current model
    if best_deviance < current_deviance:
        # update the current deviance and remove the best feature from remaining_cols
        current_deviance = best_deviance
        deviances.append((best_removed_col, current_deviance))
        remaining_cols.remove(best_removed_col)
        Xtrain = Xtrain.drop(best_removed_col, axis=1)
        Xtest = Xtest.drop(best_removed_col, axis=1)
    else:
        # if the deviance of the best reduced model is not better, break the while loop
        break

## ------------------------------------------------------ GRADIENT EXTRACTION/INTERACTIONS ----------------------------------------------------- ##
from tensorflow.keras import Model
import tensorflow.keras.backend as K


# In this section we will compute the gardients of the regression attentions with resepect to all input features
# As it will be easier for the implementation to call the Gradient Layer keras Tool in a specific class to avoid disenabling eager.Execution
# If you run the beta_extraction, re-run the architecture 

class GradientLayer(tf.keras.layers.Layer):
    def __init__(self, target_tensor_idx, **kwargs):
        self.target_tensor_idx = target_tensor_idx
        super(GradientLayer, self).__init__(**kwargs)

    def call(self, inputs):
        target_tensor = inputs[self.target_tensor_idx]
        grad_fn = K.gradients(target_tensor, inputs)
        return grad_fn

    def compute_output_shape(self, input_shape):
        return input_shape

for j in range(Xtrain.shape[1]):
    beta_j = tf.keras.layers.Lambda(lambda x: x[:, j])(Attention)
    grad_layer = GradientLayer(target_tensor_idx=0)
    grad = grad_layer([beta_j, inp_num])
    model_grad = keras.Model(inputs=inp_num, outputs=grad)
    grad_beta = model_grad.predict(Xtrain)
    grad_computed = pd.DataFrame(grad_beta[1])
    varname = Xtrain.columns[j]
    grad_computed.to_csv(f"grad_compute_{varname}.csv", index=False)

X.to_csv('X.csv', index=False)

# The csv created contains the respective gradients for each regression attentions (1 csv/reg attention with 33 columns each one for the partial derivative wrt the spec input)
# The next step of local Regression has been done in R to take advantage of the specific function available but feel free to do it in python 

## ------------------------------- This section is fancy as it just allows to compare the weights attribuated to test set ------------------------------------------ ##

beta_x_test = zz.predict(Xtest)
beta_x_test = beta_x_test * b0
beta_x_test_df = pd.DataFrame(beta_x_test)
beta_x_test_df.columns = ['Beta_' + col_name for col_name in Xtest.columns]

# concat coeff and plots
Xtest = Xtest.reset_index(drop=True)

coef_plot_test = pd.concat([Xtest, beta_x_test_df], axis=1)

### sample to plot

nsample = 4000
seed = 87031800

np.random.seed(seed)
idx = np.random.choice(coef_plot_test.shape[0], nsample, replace=False)

beta_to_plot = coef_plot_test.iloc[idx]
test_to_plot = Xtest.iloc[idx]

# define the line size and quant_rand values
line_size = 1
quant_rand = 0.5

# get the list of feature names and beta feature names
name_test = Xtest.columns
name_beta = beta_x_test_df.columns

# loop over each feature and beta feature pair
for feature, beta_feature in zip(name_test, name_beta):
    # create a data frame with the data to plot
    dat_plt = pd.DataFrame({'var': beta_to_plot[feature],
                            'bx': beta_to_plot[beta_feature],
                            'col': ['green'] * nsample})
    # create the scatter plot
    fig = plt.figure()
    ax = sns.scatterplot(data=dat_plt, x='var', y='bx')
    ax.axhline(y=0, color='red', linewidth=line_size)
    ax.axhline(y=-0.5, color='green', linewidth=line_size)
    ax.axhline(y=0.5, color='green', linewidth=line_size)
    ax.axhline(y=-1/4, color='orange', linewidth=line_size, linestyle='dashed')
    ax.axhline(y=1/4, color='orange', linewidth=line_size, linestyle='dashed')
    ax.axvspan(xmin=min(dat_plt['var']), xmax=max(dat_plt['var']), ymin=-quant_rand, ymax=quant_rand, alpha=0.002, facecolor='green')
    ax.set(title="Regression attention", 
           xlabel=feature, 
           ylabel="regression attention beta({})".format(beta_feature))
    # show the plot
    fig.show()

# ================================================================================================================================================== #
#   Keras model construction with month|year / Panel data has been introducted in a naive way, feel free to improve this part of the implementation  #
# ================================================================================================================================================== #

X = data_without_text.drop("loan_status", axis= 1)
X = X.drop('Scaled_recoveries', axis=1)
Y = data_without_text["loan_status"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=87031800)
calib_test = pd.concat([Xtest, Ytest], axis = 1)
calib_train = pd.concat([Xtrain, Ytrain], axis = 1)


seed = 87031800
np.random.seed(seed)

inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name = 'Input_var')

dense1 = keras.layers.Dense(45, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.01), name = 'first')(inp_num)
dense2 = keras.layers.Dense(20, activation=tf.nn.relu, name = 'second')(dense1)
dense3 = keras.layers.Dense(10, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.01), name = 'third')(dense2)
dense4 = keras.layers.Dense(20, activation=tf.nn.relu, name = 'fourth')(dense3)
dense5 = keras.layers.Dense(45, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.01), name = 'fifth')(dense4)
Attention = keras.layers.Dense(Xtrain.shape[1], activation="linear", name = 'Attention')(dense5)

Dot = keras.layers.Dot(axes=1)([inp_num, Attention])

Response = keras.layers.Dense(1, activation= tf.keras.activations.sigmoid, name = 'Response')(Dot)
model = keras.Model(inputs=inp_num, outputs=Response)
print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
model.compile(loss= Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.001))
history = model.fit(Xtrain, Ytrain, epochs=200, batch_size = len(Xtrain), callbacks = [callback], validation_split=0.2, verbose=2)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

to_see = model.predict(Xtest)
pred_calib = pd.DataFrame(to_see, columns=["pred"])
to_see = np.column_stack((to_see, Ytest))
to_mean = to_see[to_see[:, 1] == 1]
np.mean(to_see[:, 0])

y_pred = to_see[:,0]
y_true = to_see[:,1]
-2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))

### -------------- Calibration-------------- ###
calib_test = calib_test.reset_index(drop=True)
pred_calib = pred_calib.reset_index(drop=True)
calib_test = pd.concat([calib_test, pred_calib], axis = 1)


def my_fun(x):
    return np.mean(x)

groups = calib_test.columns[:14].tolist()

to_comp1 = calib_test.groupby(groups)['loan_status'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)
to_comp2 = calib_test.groupby(groups)['pred'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)

O = to_comp1['my_fun']*to_comp1['count']
E = to_comp2['my_fun']*to_comp2['count']
n = to_comp1['count']
HLT = np.sum((O - E)**2/(E*(1-to_comp2['my_fun']) + 1e-10))
pval = 1 - chi2.cdf(HLT, df=(len(to_comp1)-2))

print((O - E)**2/(E*(1-to_comp2['my_fun'])))
print(E*(1-to_comp2['my_fun']))

plt.figure(figsize=(7,6))
plt.scatter(to_comp2['my_fun'], to_comp1['my_fun'], c='blue')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Predicted Frequency')
plt.ylabel('True Frequency')
plt.title('Group Calibration (LocalGLMnet)')
plt.legend([r'$\hat{{C}}$ = {:.3f}'.format(HLT), 'p-value = {:.3f}'.format(pval)], loc='upper left')
plt.grid()

to_see2 = model.predict(Xtrain)
pred_calib_train = pd.DataFrame(to_see2, columns=["pred"])
to_see2 = np.column_stack((to_see2, Ytrain))
to_mean = to_see2[to_see2[:, 1] == 1]
np.mean(to_mean[:, 0])

calib_train = calib_train.reset_index(drop=True)
pred_calib_train = pred_calib_train.reset_index(drop=True)
calib_train = pd.concat([calib_train, pred_calib_train], axis = 1)

to_comp1 = calib_train.groupby(groups)['loan_status'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)
to_comp2 = calib_train.groupby(groups)['pred'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)

O = to_comp1['my_fun']*to_comp1['count']
E = to_comp2['my_fun']*to_comp2['count']
n = to_comp1['count']
HLT = np.sum((O - E)**2/(E*(1-to_comp2['my_fun'])))
pval = 1 - chi2.cdf(HLT, df=(len(to_comp1)-2))

print((O - E)**2/(E*(1-to_comp2['my_fun'])))
print(E*(1-to_comp2['my_fun']))

plt.figure(figsize=(7,6))
plt.scatter(to_comp2['my_fun'], to_comp1['my_fun'], c='blue')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Predicted Frequency')
plt.ylabel('True Frequency')
plt.title('Group Calibration (LocalGLMnet)')
plt.legend([r'$\hat{{C}}$ = {:.3f}'.format(HLT), 'p-value = {:.3f}'.format(pval)], loc='upper left')
plt.grid()

y_pred = to_see2[:,0]
y_true = to_see2[:,1]
-2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))

# =============================================================== #
#   Single hd layer computation used for the results comparision  #
# =============================================================== #

# The optimal hyperparemeters of this model are not optimal, see the Rcode if you want to understand the way we can fin an optimal SL NN with
# GS and CV and including regularization for different number of neurons
data_nn = pd.read_csv('nn.csv')
data_nn = data_nn.drop('Scaled_recoveries', axis=1)
X = data_nn.drop("loan_status", axis= 1)
Y = data_nn["loan_status"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=87031800)
sum(Ytest)/len(Ytest) # mu in test set
sum(Ytrain)/len(Ytrain) # mu in train set
calib_test = pd.concat([Xtest, Ytest], axis = 1)
calib_train = pd.concat([Xtrain, Ytrain], axis = 1)

inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name = 'Input_var')
dense1 = keras.layers.Dense(100, activation=tf.nn.relu, name = 'first')(inp_num)
Response = keras.layers.Dense(1, activation= tf.keras.activations.sigmoid, name = 'Response')(dense1)
model = keras.Model(inputs=inp_num, outputs=Response)
print(model.summary())


model.compile(loss= Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.00001))
history = model.fit(Xtrain, Ytrain, epochs=200, batch_size = len(Xtrain), validation_split=0.2, verbose=2)


to_see = model.predict(Xtest)
pred_calib = pd.DataFrame(to_see, columns=["pred"])
to_see = np.column_stack((to_see, Ytest))
to_mean = to_see[to_see[:, 1] == 1]
np.mean(to_see[:, 0])

y_pred = to_see[:,0]
y_true = to_see[:,1]
-2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))


calib_test = calib_test.reset_index(drop=True)
pred_calib = pred_calib.reset_index(drop=True)
calib_test = pd.concat([calib_test, pred_calib], axis = 1)


def my_fun(x):
    return np.mean(x)

groups = calib_test.columns[:14].tolist()

to_comp1 = calib_test.groupby(groups)['loan_status'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)
to_comp2 = calib_test.groupby(groups)['pred'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)

O = to_comp1['my_fun']*to_comp1['count']
E = to_comp2['my_fun']*to_comp2['count']
n = to_comp1['count']
HLT = np.sum((O - E)**2/(E*(1-to_comp2['my_fun']) + 1e-10))
pval = 1 - chi2.cdf(HLT, df=(len(to_comp1)-2))

print((O - E)**2/(E*(1-to_comp2['my_fun'])))
print(E*(1-to_comp2['my_fun']))

plt.figure(figsize=(7,6))
plt.scatter(to_comp2['my_fun'], to_comp1['my_fun'], c='blue')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Predicted Frequency')
plt.ylabel('True Frequency')
plt.title('Group Calibration (LocalGLMnet)')
plt.legend([r'$\hat{{C}}$ = {:.3f}'.format(HLT), 'p-value = {:.3f}'.format(pval)], loc='upper left')
plt.grid()

# =============================================================== #
#   Improced LocalGLMnet with embedding layers for text features  #
# =============================================================== #


cols_to_drop = data.filter(like='month').columns.tolist() + \
               data.filter(like='year').columns.tolist()

# Use drop() to drop the selected columns
data_with_text = data.drop(columns=cols_to_drop)

data_with_text['loan_status'] = data_with_text['loan_status'].map({'Charged Off':1, 'Fully Paid':0})

to_convert = ['emp_length','term', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'pub_rec_bankruptcies']
encoder = OneHotEncoder(sparse = False)
encoder.fit(data_with_text[to_convert])
encoded_columns = encoder.transform(data_with_text[to_convert])
encoded_df = pd.DataFrame(encoded_columns, columns = encoder.get_feature_names_out(to_convert))

data_with_text = data_with_text.drop(columns=to_convert)
data_with_text = pd.concat([data_with_text, encoded_df], axis=1)


num_names = ["loan_amnt","int_rate","annual_inc","dti","delinq_2yrs","last_pymnt_amnt","inq_last_6mths","open_acc","revol_bal","revol_util","recoveries","total_rec_late_fee"]

min_max_dict = {}
data_with_text, min_max_dict = var_standardization(data_with_text, "loan_amnt", min_max_dict)
data_with_text, min_max_dict = var_standardization(data_with_text, "int_rate", min_max_dict)
data_with_text, min_max_dict = var_standardization(data_with_text, "annual_inc", min_max_dict)
data_with_text, min_max_dict = var_standardization(data_with_text, "dti", min_max_dict)
data_with_text, min_max_dict = var_standardization(data_with_text, "delinq_2yrs", min_max_dict)
data_with_text, min_max_dict = var_standardization(data_with_text, "last_pymnt_amnt", min_max_dict)
data_with_text, min_max_dict = var_standardization(data_with_text, "inq_last_6mths", min_max_dict)
data_with_text, min_max_dict = var_standardization(data_with_text, "open_acc", min_max_dict)
data_with_text, min_max_dict = var_standardization(data_with_text, "revol_bal", min_max_dict)
data_with_text, min_max_dict = var_standardization(data_with_text, "revol_util", min_max_dict)
data_with_text, min_max_dict = var_standardization(data_with_text, "recoveries", min_max_dict)
data_with_text, min_max_dict = var_standardization(data_with_text, "total_rec_late_fee", min_max_dict)

# This is done to clean the text features (even if it has been done in our preprocessing in R, if you start doing this from scratch, run these lines)
data_with_text['emp_title'].fillna('no information', inplace=True)
data_with_text['desc'].fillna('no information', inplace=True)
data_with_text['title'].fillna('no information', inplace=True)

#################################################
#   implementation with word2vec pre trained    #
#################################################

# https://radimrehurek.com/gensim/models/word2vec.html
from gensim.models import Word2Vec

# ------------------------- create word index and tokenization ----------------------- #

# Tokenization
data_with_text['tokens_emp_title'] = data_with_text['emp_title'].apply(word_tokenize)
data_with_text['tokens_desc'] = data_with_text['desc'].apply(word_tokenize)
data_with_text['tokens_title'] = data_with_text['title'].apply(word_tokenize)
# Remove text columns
data_with_text  = data_with_text.loc[:, ~data_with_text.columns.isin(['emp_title', 'desc', 'title'])]

# Word2Vec training for text features
mymodel1 = Word2Vec(data_with_text['tokens_emp_title'], min_count=1)
mymodel2 = Word2Vec(data_with_text['tokens_desc'], min_count=1)
mymodel3 = Word2Vec(data_with_text['tokens_title'], min_count=1)

# Get the vocabulary for each Word2Vec model
words1 = list(mymodel1.wv.key_to_index)
words2 = list(mymodel2.wv.key_to_index)
words3 = list(mymodel3.wv.key_to_index)

# Create embedding dictionaries 
embedding_dict_et = {}
for word in mymodel1.wv.key_to_index:
    embedding_dict_et[word] = mymodel1.wv.get_vector(word)
    
embedding_dict_d = {}
for word in mymodel2.wv.key_to_index:
    embedding_dict_d[word] = mymodel2.wv.get_vector(word)
    
embedding_dict_t = {}
for word in mymodel3.wv.key_to_index:
    embedding_dict_t[word] = mymodel3.wv.get_vector(word)
    
# Create embedding matrices 
embedding_dim1 = mymodel1.vector_size
word_index1 = mymodel1.wv.index_to_key
num_words1 = len(word_index1)
embedding_matrix1 = np.zeros((num_words1, embedding_dim1))

for i, word in enumerate(word_index1):
    embedding_matrix1[i] = mymodel1.wv[word]

embedding_dim2 = mymodel2.vector_size
word_index2 = mymodel2.wv.index_to_key
num_words2 = len(word_index2)
embedding_matrix2 = np.zeros((num_words2, embedding_dim2))

for i, word in enumerate(word_index2):
    embedding_matrix2[i] = mymodel2.wv[word]
    
embedding_dim3 = mymodel3.vector_size
word_index3 = mymodel3.wv.index_to_key
num_words3 = len(word_index3)
embedding_matrix3 = np.zeros((num_words3, embedding_dim3))

for i, word in enumerate(word_index3):
    embedding_matrix3[i] = mymodel3.wv[word]
    
# Function to convert tokens to indices based on word index    
def convert_tokens_to_indices(tokens, word_index):
    indices = []
    for token in tokens:
        if token in word_index:
            indices.append(word_index[token])
        else:
            indices.append(0) # use index 0 for unknown words
    return indices

# Convert tokens to indices 
data_with_text['indices_emp_title'] = data_with_text['tokens_emp_title'].apply(convert_tokens_to_indices, args=[mymodel1.wv.key_to_index])
data_with_text['indices_desc'] = data_with_text['tokens_desc'].apply(convert_tokens_to_indices, args=[mymodel2.wv.key_to_index])
data_with_text['indices_title'] = data_with_text['tokens_title'].apply(convert_tokens_to_indices, args=[mymodel3.wv.key_to_index])

# padding to get fixed length text features sequences
max_len1 = 10
padded_sequences1 = []
for seq in data_with_text['indices_emp_title']:
    if len(seq) >= max_len1:
        padded_sequences1.append(seq[:max_len1])
    else:
        padded_sequences1.append(seq + [0] * (max_len1 - len(seq)))
        
max_len2 = 30
padded_sequences2 = []
for seq in data_with_text['indices_desc']:
    if len(seq) >= max_len2:
        padded_sequences2.append(seq[:max_len2])
    else:
        padded_sequences2.append(seq + [0] * (max_len2 - len(seq)))
        
max_len3 = 10
padded_sequences3 = []
for seq in data_with_text['indices_title']:
    if len(seq) >= max_len3:
        padded_sequences3.append(seq[:max_len3])
    else:
        padded_sequences3.append(seq + [0] * (max_len3 - len(seq)))

        
X = data_with_text.drop(["loan_status", "tokens_emp_title", "tokens_desc", "tokens_title", "indices_emp_title", "indices_desc", "indices_title", "Scaled_recoveries"], axis= 1)
Y = data_with_text["loan_status"]
Xtrain, Xtest, Ytrain, Ytest, seq1_train, seq1_test, seq2_train, seq2_test, seq3_train, seq3_test = train_test_split(X, Y, padded_sequences1, padded_sequences2, padded_sequences3, test_size=0.2, random_state=87031800)
calib_test = pd.concat([Xtest, Ytest], axis = 1)
calib_train = pd.concat([Xtrain, Ytrain], axis = 1)

# convert padded_sequences1 to Numpy array
seq1_train = np.array(seq1_train)
seq1_test = np.array(seq1_test)
seq2_train = np.array(seq2_train)
seq2_test = np.array(seq2_test)
seq3_train = np.array(seq3_train)
seq3_test = np.array(seq3_test)

##########################
#   hyperband embeddings #
##########################

# In what follows we decided to test the performances obtained by including 1,2 or 3 text features thus, If you find some architectures with only 1 or 2 text features instead of 3, it's normal
# but feel free to add or remove the features and thus to modify the architecture (pooled/flatten sequences, concatenations and number of embedding layers)
# This hyperband will thus be modified each time we train an embedding ( 3! pre-trained Word2vec + 3! trained embedding from scratch + 3! trained pre-trained embeddings)

import keras_tuner
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# When training embedding (3! trained embedding from scratch + 3! trained pre-trained embeddings) set trainable = True
def build_model(hp):
    inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name='Input_var')
    inp_seq1 = keras.layers.Input(shape=(seq1_train.shape[1],), name='Input_seq1')
    inp_seq2 = keras.layers.Input(shape=(seq2_train.shape[1],), name='Input_seq2')
    inp_seq3 = keras.layers.Input(shape=(seq3_train.shape[1],), name='Input_seq3')
    # embedding layers for the sequences
    emb_seq1 = keras.layers.Embedding(input_dim=len(embedding_dict_et), output_dim=embedding_dim1, weights=[embedding_matrix1], input_length=max_len1, trainable=False, name='embedding_et')(inp_seq1)
    emb_seq2 = keras.layers.Embedding(input_dim=len(embedding_dict_d), output_dim=embedding_dim2, weights=[embedding_matrix2], input_length=max_len2, trainable=False, name='embedding_d')(inp_seq2)
    emb_seq3 = keras.layers.Embedding(input_dim=len(embedding_dict_t), output_dim=embedding_dim3, weights=[embedding_matrix3], input_length=max_len3, trainable=False, name='embedding_t')(inp_seq3)

    # global max pooling of the embedded sequences
    pooled_seq1 =  keras.layers.GlobalMaxPooling1D()(emb_seq1)
    pooled_seq2 =  keras.layers.GlobalMaxPooling1D()(emb_seq2)
    pooled_seq3 =  keras.layers.GlobalMaxPooling1D()(emb_seq3)

    # concatenate the pooled sequences and numerical input
    concat = keras.layers.Concatenate()([pooled_seq1,pooled_seq2, pooled_seq3, inp_num])
    
    # Tune the number of layers
    x = concat
    for i in range(hp.Int("num_layers", 4, 10)):
        x = Dense(
            units=hp.Int(f"units_{i}", min_value=25, max_value=100, step=2),
            activation=hp.Choice("activation", ["relu"]),
            kernel_regularizer=regularizers.l2(hp.Float(f"l2_reg_{i}", 0.0, 0.1, step=0.01)),
        )(x)
    Attention = layers.Dense(Xtrain.shape[1], activation="linear", name='Attention')(x)
    Dot = layers.Dot(axes=1)([inp_num, Attention])
    response = Dense(1, activation=tf.keras.activations.sigmoid, name='Response')(Dot)
    
    # Tune the learning rate
    learning_rate = hp.Choice("learning_rate", values=[1e-5, 1e-4, 1e-3])
    model = Model(inputs=[inp_num, inp_seq1, inp_seq2, inp_seq3], outputs=response)

    # Compile the model with the custom loss function
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=Deviance(eps=1e-05, w=1)
    )
    return model

    
# Define the hyperparameter search space
tuner = keras_tuner.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=1000,
    factor=3,
    seed=87031800,
    directory="tuning_dir",
    project_name="model_embeddings_pt30",
    hyperband_iterations=3
)

# Start the hyperparameter search
tuner.search([Xtrain, seq1_train, seq2_train, seq3_train], Ytrain, epochs=1000, validation_split=0.2, batch_size=len(Xtrain), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=20)])


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0].values
print(f"Number of Dense Layers: {best_hps.get('num_layers')}")
print(f"Number of Units in Dense Layers: {[best_hps.get(f'units_{i}') for i in range(best_hps.get('num_layers'))]}")
print(f"Activation Function: {best_hps.get('activation')}")
print(f"L2 Regularization: {[best_hps.get(f'l2_reg_{i}') for i in range(best_hps.get('num_layers'))]}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")
print(f"epochs number: {best_hps.get('epochs')}")
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
# Get the best model and evaluate it on the test set
best_model = tuner.get_best_models(num_models=1)[0]

from tabulate import tabulate
table = []
for key, value in best_hps.items():
    table.append([key, value])
    
latex_tabular = tabulate(table, headers=['Hyperparameter', 'Value'], tablefmt='latex')

print(latex_tabular)


# Here there is an exemple with one text feature, feel free to add the other (we don't write 3! combinations of hyperband and optimal architecture as the file will be too heavy)
# We prefer to suppress the optimal architecture each time it has been computed :)
# Tune a hyperband for each configuration !

tf.keras.utils.set_random_seed(87031800)

inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name='Input_var')
inp_emb_data1 = keras.layers.Input(shape=(len(seq1_train[0]),), name='Input_embedding_et')
emb_emp_title = keras.layers.Embedding(input_dim=len(embedding_dict_et), output_dim=embedding_dim1, weights=[embedding_matrix1], input_length=max_len1, trainable=True, name='embedding_et')(inp_emb_data1)
#flat_emp_title = keras.layers.Flatten()(emb_emp_title)
#flat_desc = keras.layers.Flatten()(emb_desc)
#flat_title = keras.layers.Flatten()(emb_title)
flat_emp_title = keras.layers.GlobalMaxPooling1D()(emb_emp_title)
concat = keras.layers.Concatenate()([flat_emp_title, inp_num])

dense1 = keras.layers.Dense(91, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.02), name = '1')(concat)
dense2 = keras.layers.Dense(73, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.09), name = '2')(dense1)
dense3 = keras.layers.Dense(63, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.02),  name = '3')(dense2)
dense4 = keras.layers.Dense(61, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.05), name = '4')(dense3)
dense5 = keras.layers.Dense(83, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.08), name = '5')(dense4)
dense6 = keras.layers.Dense(99, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.02), name = '6')(dense5)

Attention = keras.layers.Dense(Xtrain.shape[1], activation="linear", name = 'Attention')(dense6)

Dot = keras.layers.Dot(axes=1)([inp_num, Attention])

Response = keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='Response')(Dot)
model = keras.Model(inputs=[inp_num, inp_emb_data1], outputs=Response)
print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
model.compile(loss=Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.0001))

history = model.fit([Xtrain, seq1_train], Ytrain, epochs=185, batch_size = len(Xtrain),validation_split=0.2, callbacks=[callback], verbose=2)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

to_see = model.predict([Xtest, seq1_test])
pred_calib = pd.DataFrame(to_see, columns=["pred"])
to_see = np.column_stack((to_see, Ytest))
to_mean = to_see[to_see[:, 1] == 1]
mean_saturated = np.mean(to_see[:, 0])

y_pred = to_see[:,0]
y_true = to_see[:,1]
deviance = -2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))
deviance_saturated = deviance.numpy()

calib_test = calib_test.reset_index(drop=True)
pred_calib = pred_calib.reset_index(drop=True)
calib_test = pd.concat([calib_test, pred_calib], axis = 1)


def my_fun(x):
    return np.mean(x)

groups = calib_test.columns[:14].tolist()

to_comp1 = calib_test.groupby(groups)['loan_status'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)
to_comp2 = calib_test.groupby(groups)['pred'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)

O = to_comp1['my_fun']*to_comp1['count']
E = to_comp2['my_fun']*to_comp2['count']
n = to_comp1['count']
HLT = np.sum((O - E)**2/(E*(1-to_comp2['my_fun']) + 1e-10))
pval = 1 - chi2.cdf(HLT, df=(len(to_comp1)-2))

plt.figure(figsize=(7,6), facecolor="white")
plt.scatter(to_comp2['my_fun'], to_comp1['my_fun'], c='blue')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Predicted Frequency')
plt.ylabel('True Frequency')
plt.title('Group Calibration (LocalGLMnet)')
plt.legend([r'$\hat{{C}}$ = {:.3f}'.format(HLT), 'p-value = {:.3f}'.format(pval)], loc='upper left')
plt.grid()


from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


##########################################################
#   implementation with trained embeddings from scratch  #
##########################################################

from keras.preprocessing.text import Tokenizer

# Initialize Tokenizer (set the max number of words in it)
tokenizer = Tokenizer(num_words=10000)

# Convert to sequences of tokens
# Get vocabulary size (number of unique words)
# Get word index (mapping of words to indices)
tokenizer.fit_on_texts(data_with_text['emp_title'])
data_with_text['tokens_emp_title'] = tokenizer.texts_to_sequences(data_with_text['emp_title'])
vocab_size1 = len(tokenizer.word_index) + 1
word_index1 = tokenizer.word_index

tokenizer.fit_on_texts(data_with_text['desc'])
data_with_text['tokens_desc'] = tokenizer.texts_to_sequences(data_with_text['desc'])
vocab_size2 = len(tokenizer.word_index) + 1
word_index2 = tokenizer.word_index

tokenizer.fit_on_texts(data_with_text['title'])
data_with_text['tokens_title'] = tokenizer.texts_to_sequences(data_with_text['title'])
vocab_size3 = len(tokenizer.word_index) + 1
word_index3 = tokenizer.word_index

# Remove emp_title, desc, and title columns from the dataframe
data_with_text  = data_with_text.loc[:, ~data_with_text.columns.isin(['emp_title', 'desc', 'title'])]

# padding 
max_len1 = 10
padded_sequences1 = []
for seq in data_with_text['tokens_emp_title']:
    if len(seq) >= max_len1:
        padded_sequences1.append(seq[:max_len1])
    else:
        padded_sequences1.append(seq + [0] * (max_len1 - len(seq)))
        
max_len2 = 20
padded_sequences2 = []
for seq in data_with_text['tokens_desc']:
    if len(seq) >= max_len2:
        padded_sequences2.append(seq[:max_len2])
    else:
        padded_sequences2.append(seq + [0] * (max_len2 - len(seq)))
        
max_len3 = 10
padded_sequences3 = []
for seq in data_with_text['tokens_title']:
    if len(seq) >= max_len3:
        padded_sequences3.append(seq[:max_len3])
    else:
        padded_sequences3.append(seq + [0] * (max_len3 - len(seq)))
        
# Set the embedding representation of word dimension to 100 (convenient to capture patterns in the text features)    
embedding_dim = 100        
        
X = data_with_text.drop(["loan_status", "tokens_emp_title", "tokens_desc", "tokens_title", "Scaled_recoveries"], axis= 1)
Y = data_with_text["loan_status"]
Xtrain, Xtest, Ytrain, Ytest, seq1_train, seq1_test, seq2_train, seq2_test, seq3_train, seq3_test = train_test_split(X, Y, padded_sequences1, padded_sequences2, padded_sequences3, test_size=0.2, random_state=87031800)
calib_test = pd.concat([Xtest, Ytest], axis = 1)
calib_train = pd.concat([Xtrain, Ytrain], axis = 1)

# convert padded_sequences1 to Numpy array
seq1_train = np.array(seq1_train)
seq1_test = np.array(seq1_test)
seq2_train = np.array(seq2_train)
seq2_test = np.array(seq2_test)
seq3_train = np.array(seq3_train)
seq3_test = np.array(seq3_test)


# This is the optimal hyperband architecture found previously, but once again feel free to play with different combination of text features while training hyperband
tf.keras.utils.set_random_seed(87031800)

inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name='Input_var')
inp_emb_data1 = keras.layers.Input(shape=(len(seq1_train[0]),), name='Input_embedding_et')
inp_emb_data2 = keras.layers.Input(shape=(len(seq2_train[0]),), name='Input_embedding_d')
inp_emb_data3 = keras.layers.Input(shape=(len(seq3_train[0]),), name='Input_embedding_t')
emb_emp_title = keras.layers.Embedding(input_dim=vocab_size1, output_dim=embedding_dim, input_length=max_len1, name='embedding_et')(inp_emb_data1)
emb_desc = keras.layers.Embedding(input_dim=vocab_size2, output_dim=embedding_dim, input_length=max_len2, name='embedding_d')(inp_emb_data2)
emb_title = keras.layers.Embedding(input_dim=vocab_size3, output_dim=embedding_dim, input_length=max_len3, name='embedding_t')(inp_emb_data3)
#flat_emp_title = keras.layers.Flatten()(emb_emp_title)
#flat_desc = keras.layers.Flatten()(emb_desc)
#flat_title = keras.layers.Flatten()(emb_title)
flat_emp_title = keras.layers.GlobalMaxPooling1D()(emb_emp_title)
flat_title = keras.layers.GlobalMaxPooling1D()(emb_title)
flat_desc = keras.layers.GlobalMaxPooling1D()(emb_desc)
concat = keras.layers.Concatenate()([flat_emp_title,flat_desc, flat_title, inp_num])

dense1 = keras.layers.Dense(28, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.1), name = '1')(concat)
dense2 = keras.layers.Dense(23, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.08), name = '2')(dense1)
dense3 = keras.layers.Dense(59, activation=tf.nn.relu,  name = '3')(dense2)
dense4 = keras.layers.Dense(19, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.04), name = '4')(dense3)
dense5 = keras.layers.Dense(39, activation=tf.nn.relu,  name = '5')(dense4)
dense6 = keras.layers.Dense(51, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01), name = '6')(dense5)
Attention = keras.layers.Dense(Xtrain.shape[1], activation="linear", name = 'Attention')(dense6)

Dot = keras.layers.Dot(axes=1)([inp_num, Attention])

Response = keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='Response')(Dot)
model = keras.Model(inputs=[inp_num, inp_emb_data1], outputs=Response)
print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
model.compile(loss=Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.001))

history = model.fit([Xtrain, seq1_train], Ytrain, epochs=1000, batch_size = len(Xtrain),validation_split=0.2, callbacks=[callback], verbose=2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Calibration for the architecture with learned embeddings

to_see = model.predict([Xtest, seq1_test])
pred_calib = pd.DataFrame(to_see, columns=["pred"])
to_see = np.column_stack((to_see, Ytest))
to_mean = to_see[to_see[:, 1] == 1]
mean_saturated = np.mean(to_see[:, 0])

y_pred = to_see[:,0]
y_true = to_see[:,1]
deviance = -2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))
deviance_saturated = deviance.numpy()

calib_test = calib_test.reset_index(drop=True)
pred_calib = pred_calib.reset_index(drop=True)
calib_test = pd.concat([calib_test, pred_calib], axis = 1)


def my_fun(x):
    return np.mean(x)

groups = calib_test.columns[:14].tolist()

to_comp1 = calib_test.groupby(groups)['loan_status'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)
to_comp2 = calib_test.groupby(groups)['pred'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)

O = to_comp1['my_fun']*to_comp1['count']
E = to_comp2['my_fun']*to_comp2['count']
n = to_comp1['count']
HLT = np.sum((O - E)**2/(E*(1-to_comp2['my_fun']) + 1e-10))
pval = 1 - chi2.cdf(HLT, df=(len(to_comp1)-2))

plt.figure(figsize=(7,6), facecolor="white")
plt.scatter(to_comp2['my_fun'], to_comp1['my_fun'], c='blue')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Predicted Frequency')
plt.ylabel('True Frequency')
plt.title('Group Calibration (LocalGLMnet)')
plt.legend([r'$\hat{{C}}$ = {:.3f}'.format(HLT), 'p-value = {:.3f}'.format(pval)], loc='upper left')
plt.grid()


##################################################################################
#   TSNE representation of learned embeddings and cosine similarity computation  #
##################################################################################

# In this section we will assess the quality of our learned embedding as we want to know if similar words are represented close to each other in the embedding space

embedding_layer1 = model.get_layer('embedding_et')
embedding_layer2 = model.get_layer('embedding_d')
embedding_layer3 = model.get_layer('embedding_t')

learned_embeddings1 = embedding_layer1.get_weights()[0]
learned_embeddings2 = embedding_layer2.get_weights()[0]
learned_embeddings3 = embedding_layer3.get_weights()[0]


# set the range of number of clusters to try
cluster_range = range(1, 31)

# define a function to compute the WCSS for each number of clusters
def compute_wcss(embedding):
    wcss_list = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(embedding)
        wcss = kmeans.inertia_
        wcss_list.append(wcss)
    return wcss_list

# compute the WCSS for each embedding space
wcss_list1 = compute_wcss(learned_embeddings1)
wcss_list2 = compute_wcss(learned_embeddings2)
wcss_list3 = compute_wcss(learned_embeddings3)

# plot the WCSS against the number of clusters
plt.plot(cluster_range, wcss_list1, label='Embedding 1')
plt.plot(cluster_range, wcss_list2, label='Embedding 2')
plt.plot(cluster_range, wcss_list3, label='Embedding 3')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.legend()
plt.show()


kmeans1 = KMeans(n_clusters=5, random_state=0).fit(learned_embeddings1)
labels1 = kmeans1.labels_
kmeans2 = KMeans(n_clusters=5, random_state=0).fit(learned_embeddings2)
labels2 = kmeans2.labels_
kmeans3 = KMeans(n_clusters=5, random_state=0).fit(learned_embeddings3)
labels3 = kmeans3.labels_


# create a dictionary that maps index to word for each feature
index_to_word1 = {index: word for word, index in word_index1.items()}
index_to_word2 = {index: word for word, index in word_index2.items()}
index_to_word3 = {index: word for word, index in word_index3.items()}

# create a dictionary that maps index to embedding for each feature
index_to_embedding1 = {index: embedding for index, embedding in enumerate(learned_embeddings1)}
index_to_embedding2 = {index: embedding for index, embedding in enumerate(learned_embeddings2)}
index_to_embedding3 = {index: embedding for index, embedding in enumerate(learned_embeddings3)}

# create a dictionary that maps cluster label to embeddings for each feature
cluster_embedding_dict1 = {}
cluster_embedding_dict2 = {}
cluster_embedding_dict3 = {}

for i, cluster_label in enumerate(labels1):
    # get the word index for the first feature
    word_index1 = index_to_word1.get(i+1)  # add 1 to index to account for reserved index 0
    
    # get the embedding for the first feature
    embedding1 = index_to_embedding1[i]
    
    # add the embedding to the dictionary for the first feature
    if cluster_label not in cluster_embedding_dict1:
        cluster_embedding_dict1[cluster_label] = [embedding1]
    else:
        cluster_embedding_dict1[cluster_label].append(embedding1)

for i, cluster_label in enumerate(labels2):
    # get the word index for the second feature
    word_index2 = index_to_word2.get(i+1)  # add 1 to index to account for reserved index 0
    
    # get the embedding for the second feature
    embedding2 = index_to_embedding2[i]
    
    # add the embedding to the dictionary for the second feature
    if cluster_label not in cluster_embedding_dict2:
        cluster_embedding_dict2[cluster_label] = [embedding2]
    else:
        cluster_embedding_dict2[cluster_label].append(embedding2)
        
for i, cluster_label in enumerate(labels3):
    # get the word index for the third feature
    word_index3 = index_to_word3.get(i+1)  # add 1 to index to account for reserved index 0
    
    # get the embedding for the third feature
    embedding3 = index_to_embedding3[i]
    
    # add the embedding to the dictionary for the third feature
    if cluster_label not in cluster_embedding_dict3:
        cluster_embedding_dict3[cluster_label] = [embedding3]
    else:
        cluster_embedding_dict3[cluster_label].append(embedding3)

from sklearn.metrics.pairwise import cosine_similarity

# compute the average cosine similarity for each cluster and feature
avg_similarity_dict1 = {}
avg_similarity_dict2 = {}
avg_similarity_dict3 = {}

for cluster_label in cluster_embedding_dict1:
    # compute the average embedding for each cluster and feature
    avg_embedding1 = np.mean(cluster_embedding_dict1[cluster_label], axis=0)
    avg_embedding2 = np.mean(cluster_embedding_dict2[cluster_label], axis=0)
    avg_embedding3 = np.mean(cluster_embedding_dict3[cluster_label], axis=0)
    
    # compute the cosine similarity between each word embedding and the average embedding
    similarity1 = cosine_similarity(avg_embedding1.reshape(1,-1), learned_embeddings1)
    similarity2 = cosine_similarity(avg_embedding2.reshape(1,-1), learned_embeddings2)
    similarity3 = cosine_similarity(avg_embedding3.reshape(1,-1), learned_embeddings3)
    
    # get the indices of the words in the cluster
    cluster_indices = [i for i, label in enumerate(labels1) if label == cluster_label]
    
    # compute the average cosine similarity for each feature
    avg_similarity1 = np.mean(similarity1[:, cluster_indices], axis=1)
    avg_similarity2 = np.mean(similarity2[:, cluster_indices], axis=1)
    avg_similarity3 = np.mean(similarity3[:, cluster_indices], axis=1)
    
    # add the average cosine similarity to the dictionary for each feature
    avg_similarity_dict1[cluster_label] = avg_similarity1[0]
    avg_similarity_dict2[cluster_label] = avg_similarity2[0]
    avg_similarity_dict3[cluster_label] = avg_similarity3[0]

# TSNE is a good way to represent embeddings in 2D space but sincen it's less clear than in 3D space, you can go to the next section
tsne_embeddings = TSNE(n_components=2).fit_transform(learned_embeddings1)

### 2D plot ###

plt.figure(figsize=(7,6), facecolor="white")
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels1, cmap='viridis')
plt.colorbar()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('TSNE of clustered high dimensional embedding weights')
plt.show()

tsne_embeddings = TSNE(n_components=3).fit_transform(learned_embeddings1)

fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], tsne_embeddings[:, 2], c=labels1, cmap='viridis')
   
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')

plt.show()




##################################
#   hyperband for embeddings     #
##################################

# Example for 1 feature representation with TSNE and 3D space mapping

import keras_tuner
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def build_model(hp):
    inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name='Input_var')
    inp_seq1 = keras.layers.Input(shape=(seq1_train.shape[1],), name='Input_seq1')
    
    # embedding layers for the sequences
    emb_seq1 = keras.layers.Embedding(input_dim=vocab_size1, output_dim=embedding_dim, input_length=max_len1)(inp_seq1)
    
    # global max pooling of the embedded sequences
    pooled_seq1 =  keras.layers.GlobalMaxPooling1D()(emb_seq1)
    
    # concatenate the pooled sequences and numerical input
    concat = keras.layers.Concatenate()([pooled_seq1, inp_num])
    
    # Tune the number of layers
    x = concat
    for i in range(hp.Int("num_layers", 4, 10)):
        x = Dense(
            units=hp.Int(f"units_{i}", min_value=25, max_value=100, step=2),
            activation=hp.Choice("activation", ["relu"]),
            kernel_regularizer=regularizers.l2(hp.Float(f"l2_reg_{i}", 0.0, 0.1, step=0.01)),
        )(x)
    Attention = layers.Dense(Xtrain.shape[1], activation="linear", name='Attention')(x)
    Dot = layers.Dot(axes=1)([inp_num, Attention])
    response = Dense(1, activation=tf.keras.activations.sigmoid, name='Response')(Dot)
    
    # Tune the learning rate
    learning_rate = hp.Choice("learning_rate", values=[1e-5, 1e-4, 1e-3])
    model = Model(inputs=[inp_num, inp_seq1], outputs=response)

    # Compile the model with the custom loss function
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=Deviance(eps=1e-05, w=1)
    )
    return model

    
# Define the hyperparameter search space
tuner = keras_tuner.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=1000,
    factor=3,
    seed=87031800,
    directory="tuning_dir",
    project_name="model_embeddings_ff",
    hyperband_iterations=3
)

# Start the hyperparameter search
tuner.search([Xtrain, seq1_train], Ytrain, epochs=1000, validation_split=0.2, batch_size=len(Xtrain), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=20)])


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0].values
print(f"Number of Dense Layers: {best_hps.get('num_layers')}")
print(f"Number of Units in Dense Layers: {[best_hps.get(f'units_{i}') for i in range(best_hps.get('num_layers'))]}")
print(f"Activation Function: {best_hps.get('activation')}")
print(f"L2 Regularization: {[best_hps.get(f'l2_reg_{i}') for i in range(best_hps.get('num_layers'))]}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")
print(f"epochs number: {best_hps.get('epochs')}")
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
# Get the best model and evaluate it on the test set
best_model = tuner.get_best_models(num_models=1)[0]

from tabulate import tabulate
table = []
for key, value in best_hps.items():
    table.append([key, value])
    
latex_tabular = tabulate(table, headers=['Hyperparameter', 'Value'], tablefmt='latex')

print(latex_tabular)


##################################
#   hyperband optimal 3 text     #
##################################

tf.keras.utils.set_random_seed(87031800)

inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name='Input_var')
inp_seq1 = keras.layers.Input(shape=(seq1_train.shape[1],), name='Input_seq1')
inp_seq2 = keras.layers.Input(shape=(seq2_train.shape[1],), name='Input_seq2')
inp_seq3 = keras.layers.Input(shape=(seq3_train.shape[1],), name='Input_seq3')

# embedding layers for the sequences
emb_seq1 = keras.layers.Embedding(input_dim=vocab_size1, output_dim=embedding_dim, input_length=max_len1, name='embedding_et')(inp_seq1)
emb_seq2 = keras.layers.Embedding(input_dim=vocab_size2, output_dim=embedding_dim, input_length=max_len2, name='embedding_d')(inp_seq2)
emb_seq3 = keras.layers.Embedding(input_dim=vocab_size3, output_dim=embedding_dim, input_length=max_len3, name='embedding_t')(inp_seq3)

# global max pooling of the embedded sequences
pooled_seq1 =  keras.layers.GlobalMaxPooling1D()(emb_seq1)
pooled_seq2 =  keras.layers.GlobalMaxPooling1D()(emb_seq2)
pooled_seq3 =  keras.layers.GlobalMaxPooling1D()(emb_seq3)

# concatenate the pooled sequences and numerical input
concat = keras.layers.Concatenate()([pooled_seq1, pooled_seq2, pooled_seq3, inp_num])

dense1 = keras.layers.Dense(75, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01), name = '1')(concat)
dense2 = keras.layers.Dense(73, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.06), name = '2')(dense1)
dense3 = keras.layers.Dense(87, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.02), name = '3')(dense2)
dense4 = keras.layers.Dense(53, activation=tf.nn.relu, name = '4')(dense3)
dense5 = keras.layers.Dense(71, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.1), name = '5')(dense4)
Attention = keras.layers.Dense(Xtrain.shape[1], activation="linear", name = 'Attention')(dense5)

Dot = keras.layers.Dot(axes=1)([inp_num, Attention])

Response = keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='Response')(Dot)
model = keras.Model(inputs=[inp_num, inp_seq1, inp_seq2, inp_seq3 ], outputs=Response)
print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.compile(loss=Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.001))

history = model.fit([Xtrain, seq1_train, seq2_train, seq3_train], Ytrain, epochs=44, batch_size = len(Xtrain),validation_split=0.2, callbacks=[callback], verbose=2)


to_see = model.predict([Xtest, seq1_test,seq2_test, seq3_test])
pred_calib = pd.DataFrame(to_see, columns=["pred"])
to_see = np.column_stack((to_see, Ytest))
to_mean = to_see[to_see[:, 1] == 1]
mean_saturated = np.mean(to_see[:, 0])

y_pred = to_see[:,0]
y_true = to_see[:,1]
deviance = -2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))
deviance_saturated = deviance.numpy()

calib_test = calib_test.reset_index(drop=True)
pred_calib = pred_calib.reset_index(drop=True)
calib_test = pd.concat([calib_test, pred_calib], axis = 1)


def my_fun(x):
    return np.mean(x)

groups = calib_test.columns[:14].tolist()

to_comp1 = calib_test.groupby(groups)['loan_status'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)
to_comp2 = calib_test.groupby(groups)['pred'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)

O = to_comp1['my_fun']*to_comp1['count']
E = to_comp2['my_fun']*to_comp2['count']
n = to_comp1['count']
HLT = np.sum((O - E)**2/(E*(1-to_comp2['my_fun']) + 1e-10))
pval = 1 - chi2.cdf(HLT, df=(len(to_comp1)-2))

plt.figure(figsize=(7,6), facecolor="white")
plt.scatter(to_comp2['my_fun'], to_comp1['my_fun'], c='blue')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Predicted Frequency')
plt.ylabel('True Frequency')
plt.title('Group Calibration (LocalGLMnet)')
plt.legend([r'$\hat{{C}}$ = {:.3f}'.format(HLT), 'p-value = {:.3f}'.format(pval)], loc='upper left')
plt.grid()


embedding_layer1 = model.get_layer('embedding_et')
embedding_layer2 = model.get_layer('embedding_d')
embedding_layer3 = model.get_layer('embedding_t')

learned_embeddings1 = embedding_layer1.get_weights()[0]
learned_embeddings2 = embedding_layer2.get_weights()[0]
learned_embeddings3 = embedding_layer3.get_weights()[0]

from sklearn.cluster import KMeans
# set the range of number of clusters to try
cluster_range = range(1, 31)

# define a function to compute the WCSS for each number of clusters
def compute_wcss(embedding):
    wcss_list = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(embedding)
        wcss = kmeans.inertia_
        wcss_list.append(wcss)
    return wcss_list

# compute the WCSS for each embedding space
wcss_list1 = compute_wcss(learned_embeddings1)
wcss_list2 = compute_wcss(learned_embeddings2)
wcss_list3 = compute_wcss(learned_embeddings3)

# plot the WCSS against the number of clusters
plt.plot(cluster_range, wcss_list1, label='Embedding 1')
plt.plot(cluster_range, wcss_list2, label='Embedding 2')
plt.plot(cluster_range, wcss_list3, label='Embedding 3')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.legend()
plt.show()


kmeans1 = KMeans(n_clusters=3, random_state=0).fit(learned_embeddings1)
labels1 = kmeans1.labels_
kmeans2 = KMeans(n_clusters=3, random_state=0).fit(learned_embeddings2)
labels2 = kmeans2.labels_
kmeans3 = KMeans(n_clusters=3, random_state=0).fit(learned_embeddings3)
labels3 = kmeans3.labels_


# create a dictionary that maps index to word for each feature
index_to_word1 = {index: word for word, index in word_index1.items()}
index_to_word2 = {index: word for word, index in word_index2.items()}
index_to_word3 = {index: word for word, index in word_index3.items()}

# create a dictionary that maps index to embedding for each feature
index_to_embedding1 = {index: embedding for index, embedding in enumerate(learned_embeddings1)}
index_to_embedding2 = {index: embedding for index, embedding in enumerate(learned_embeddings2)}
index_to_embedding3 = {index: embedding for index, embedding in enumerate(learned_embeddings3)}

# create a dictionary that maps cluster label to embeddings for each feature
cluster_embedding_dict1 = {}
cluster_embedding_dict2 = {}
cluster_embedding_dict3 = {}

for i, cluster_label in enumerate(labels1):
    # get the word index for the first feature
    word_index1 = index_to_word1.get(i+1)  # add 1 to index to account for reserved index 0
    
    # get the embedding for the first feature
    embedding1 = index_to_embedding1[i]
    
    # add the embedding to the dictionary for the first feature
    if cluster_label not in cluster_embedding_dict1:
        cluster_embedding_dict1[cluster_label] = [embedding1]
    else:
        cluster_embedding_dict1[cluster_label].append(embedding1)

for i, cluster_label in enumerate(labels2):
    # get the word index for the second feature
    word_index2 = index_to_word2.get(i+1)  # add 1 to index to account for reserved index 0
    
    # get the embedding for the second feature
    embedding2 = index_to_embedding2[i]
    
    # add the embedding to the dictionary for the second feature
    if cluster_label not in cluster_embedding_dict2:
        cluster_embedding_dict2[cluster_label] = [embedding2]
    else:
        cluster_embedding_dict2[cluster_label].append(embedding2)
        
for i, cluster_label in enumerate(labels3):
    # get the word index for the third feature
    word_index3 = index_to_word3.get(i+1)  # add 1 to index to account for reserved index 0
    
    # get the embedding for the third feature
    embedding3 = index_to_embedding3[i]
    
    # add the embedding to the dictionary for the third feature
    if cluster_label not in cluster_embedding_dict3:
        cluster_embedding_dict3[cluster_label] = [embedding3]
    else:
        cluster_embedding_dict3[cluster_label].append(embedding3)

from sklearn.metrics.pairwise import cosine_similarity

# compute the average cosine similarity for each cluster and feature
avg_similarity_dict1 = {}
avg_similarity_dict2 = {}
avg_similarity_dict3 = {}

for cluster_label in cluster_embedding_dict1:
    # compute the average embedding for each cluster and feature
    avg_embedding1 = np.mean(cluster_embedding_dict1[cluster_label], axis=0)
    avg_embedding2 = np.mean(cluster_embedding_dict2[cluster_label], axis=0)
    avg_embedding3 = np.mean(cluster_embedding_dict3[cluster_label], axis=0)
    
    # compute the cosine similarity between each word embedding and the average embedding
    similarity1 = cosine_similarity(avg_embedding1.reshape(1,-1), learned_embeddings1)
    similarity2 = cosine_similarity(avg_embedding2.reshape(1,-1), learned_embeddings2)
    similarity3 = cosine_similarity(avg_embedding3.reshape(1,-1), learned_embeddings3)
    
    # get the indices of the words in the cluster
    cluster_indices = [i for i, label in enumerate(labels1) if label == cluster_label]
    
    # compute the average cosine similarity for each feature
    avg_similarity1 = np.mean(similarity1[:, cluster_indices], axis=1)
    avg_similarity2 = np.mean(similarity2[:, cluster_indices], axis=1)
    avg_similarity3 = np.mean(similarity3[:, cluster_indices], axis=1)
    
    # add the average cosine similarity to the dictionary for each feature
    avg_similarity_dict1[cluster_label] = avg_similarity1[0]
    avg_similarity_dict2[cluster_label] = avg_similarity2[0]
    avg_similarity_dict3[cluster_label] = avg_similarity3[0]


tsne_embeddings = TSNE(n_components=3).fit_transform(learned_embeddings3)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Define custom colormap
colors = ['red', 'black', 'yellow']
cmap = mcolors.ListedColormap(colors)

fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], tsne_embeddings[:, 2], c=labels3, cmap=cmap)

# Add average similarity values to corresponding clusters
unique_labels = np.unique(labels3)

avg_sim_labels = []
for label in unique_labels:
    avg_sim = avg_similarity_dict3[label]
    avg_sim_str = f"{avg_sim:.3f}"
    avg_sim_labels.append(avg_sim_str)

# Create custom legend
handles, labels = scatter.legend_elements()
legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
legend_handles = handles
legend = ax.legend(legend_handles, legend_labels, loc='lower left', title='Clusters', frameon=True)
legend.get_title().set_fontsize(16)

# Add second legend for average similarity
ax.add_artist(legend)
avg_sim_legend = ax.legend(legend_handles, avg_sim_labels, loc='lower right', title='avg_sim_clusters', frameon=True)
avg_sim_legend.get_title().set_fontsize(16)

# Set plot title
avg_similarity3f = avg_similarity3[0]
ax.set_title(f'Average Similarity = {avg_similarity3f:.3f}', fontsize=20)

# Set axis labels
ax.set_xlabel('t-SNE Dimension 1', fontsize=16)
ax.set_ylabel('t-SNE Dimension 2', fontsize=16)
ax.set_zlabel('t-SNE Dimension 3', fontsize=16)

plt.show()



##################################
#   hyperband optimal 2 text     #
##################################

tf.keras.utils.set_random_seed(87031800)

inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name='Input_var')
inp_seq1 = keras.layers.Input(shape=(seq1_train.shape[1],), name='Input_seq1')
inp_seq3 = keras.layers.Input(shape=(seq3_train.shape[1],), name='Input_seq3')

# embedding layers for the sequences
emb_seq1 = keras.layers.Embedding(input_dim=vocab_size1, output_dim=embedding_dim, input_length=max_len1, name='embedding_et')(inp_seq1)
emb_seq3 = keras.layers.Embedding(input_dim=vocab_size3, output_dim=embedding_dim, input_length=max_len3, name='embedding_t')(inp_seq3)

# global max pooling of the embedded sequences
pooled_seq1 =  keras.layers.GlobalMaxPooling1D()(emb_seq1)
pooled_seq3 =  keras.layers.GlobalMaxPooling1D()(emb_seq3)

# concatenate the pooled sequences and numerical input
concat = keras.layers.Concatenate()([pooled_seq1, pooled_seq3, inp_num])

dense1 = keras.layers.Dense(73, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.06), name = '1')(concat)
dense2 = keras.layers.Dense(41, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.06), name = '2')(dense1)
dense3 = keras.layers.Dense(51, activation=tf.nn.relu, name = '3')(dense2)
dense4 = keras.layers.Dense(51, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.09), name = '4')(dense3)
dense5 = keras.layers.Dense(81, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.1), name = '5')(dense4)
dense6 = keras.layers.Dense(31, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01), name = '6')(dense5)
dense7 = keras.layers.Dense(29, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.02), name = '7')(dense6)
dense8 = keras.layers.Dense(81, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.07), name = '8')(dense7)
dense9 = keras.layers.Dense(35, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.06), name = '9')(dense8)
dense10 = keras.layers.Dense(75, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.09), name = '10')(dense9)
Attention = keras.layers.Dense(Xtrain.shape[1], activation="linear", name = 'Attention')(dense10)

Dot = keras.layers.Dot(axes=1)([inp_num, Attention])

Response = keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='Response')(Dot)
model = keras.Model(inputs=[inp_num, inp_seq1, inp_seq3], outputs=Response)
print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
model.compile(loss=Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.0001))

history = model.fit([Xtrain, seq1_train, seq3_train], Ytrain, epochs=268, batch_size = len(Xtrain),validation_split=0.2, callbacks=[callback], verbose=2)


to_see = model.predict([Xtest, seq1_test, seq3_test])
pred_calib = pd.DataFrame(to_see, columns=["pred"])
to_see = np.column_stack((to_see, Ytest))
to_mean = to_see[to_see[:, 1] == 1]
mean_saturated = np.mean(to_see[:, 0])

y_pred = to_see[:,0]
y_true = to_see[:,1]
deviance = -2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))
deviance_saturated = deviance.numpy()

calib_test = calib_test.reset_index(drop=True)
pred_calib = pred_calib.reset_index(drop=True)
calib_test = pd.concat([calib_test, pred_calib], axis = 1)


def my_fun(x):
    return np.mean(x)

groups = calib_test.columns[:14].tolist()

to_comp1 = calib_test.groupby(groups)['loan_status'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)
to_comp2 = calib_test.groupby(groups)['pred'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)

O = to_comp1['my_fun']*to_comp1['count']
E = to_comp2['my_fun']*to_comp2['count']
n = to_comp1['count']
HLT = np.sum((O - E)**2/(E*(1-to_comp2['my_fun']) + 1e-10))
pval = 1 - chi2.cdf(HLT, df=(len(to_comp1)-2))

plt.figure(figsize=(7,6), facecolor="white")
plt.scatter(to_comp2['my_fun'], to_comp1['my_fun'], c='blue')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Predicted Frequency')
plt.ylabel('True Frequency')
plt.title('Group Calibration (LocalGLMnet)')
plt.legend([r'$\hat{{C}}$ = {:.3f}'.format(HLT), 'p-value = {:.3f}'.format(pval)], loc='upper left')
plt.grid()


embedding_layer1 = model.get_layer('embedding_et')
embedding_layer3 = model.get_layer('embedding_t')

learned_embeddings1 = embedding_layer1.get_weights()[0]
learned_embeddings3 = embedding_layer3.get_weights()[0]

from sklearn.cluster import KMeans
# set the range of number of clusters to try
cluster_range = range(1, 31)

# define a function to compute the WCSS for each number of clusters
def compute_wcss(embedding):
    wcss_list = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(embedding)
        wcss = kmeans.inertia_
        wcss_list.append(wcss)
    return wcss_list

# compute the WCSS for each embedding space
wcss_list1 = compute_wcss(learned_embeddings1)
wcss_list3 = compute_wcss(learned_embeddings3)

# plot the WCSS against the number of clusters
plt.plot(cluster_range, wcss_list1, label='Embedding 1')
plt.plot(cluster_range, wcss_list3, label='Embedding 3')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.legend()
plt.show()


kmeans1 = KMeans(n_clusters=3, random_state=0).fit(learned_embeddings1)
labels1 = kmeans1.labels_
kmeans3 = KMeans(n_clusters=3, random_state=0).fit(learned_embeddings3)
labels3 = kmeans3.labels_


# create a dictionary that maps index to word for each feature
index_to_word1 = {index: word for word, index in word_index1.items()}
index_to_word3 = {index: word for word, index in word_index3.items()}

# create a dictionary that maps index to embedding for each feature
index_to_embedding1 = {index: embedding for index, embedding in enumerate(learned_embeddings1)}
index_to_embedding3 = {index: embedding for index, embedding in enumerate(learned_embeddings3)}

# create a dictionary that maps cluster label to embeddings for each feature
cluster_embedding_dict1 = {}
cluster_embedding_dict3 = {}

for i, cluster_label in enumerate(labels1):
    # get the word index for the first feature
    word_index1 = index_to_word1.get(i+1)  # add 1 to index to account for reserved index 0
    
    # get the embedding for the first feature
    embedding1 = index_to_embedding1[i]
    
    # add the embedding to the dictionary for the first feature
    if cluster_label not in cluster_embedding_dict1:
        cluster_embedding_dict1[cluster_label] = [embedding1]
    else:
        cluster_embedding_dict1[cluster_label].append(embedding1)
        
for i, cluster_label in enumerate(labels3):
    # get the word index for the third feature
    word_index3 = index_to_word3.get(i+1)  # add 1 to index to account for reserved index 0
    
    # get the embedding for the third feature
    embedding3 = index_to_embedding3[i]
    
    # add the embedding to the dictionary for the third feature
    if cluster_label not in cluster_embedding_dict3:
        cluster_embedding_dict3[cluster_label] = [embedding3]
    else:
        cluster_embedding_dict3[cluster_label].append(embedding3)

from sklearn.metrics.pairwise import cosine_similarity

# compute the average cosine similarity for each cluster and feature
avg_similarity_dict1 = {}
avg_similarity_dict3 = {}

for cluster_label in cluster_embedding_dict1:
    # compute the average embedding for each cluster and feature
    avg_embedding1 = np.mean(cluster_embedding_dict1[cluster_label], axis=0)
    avg_embedding3 = np.mean(cluster_embedding_dict3[cluster_label], axis=0)
    
    # compute the cosine similarity between each word embedding and the average embedding
    similarity1 = cosine_similarity(avg_embedding1.reshape(1,-1), learned_embeddings1)
    similarity3 = cosine_similarity(avg_embedding3.reshape(1,-1), learned_embeddings3)
    
    # get the indices of the words in the cluster
    cluster_indices = [i for i, label in enumerate(labels1) if label == cluster_label]
    
    # compute the average cosine similarity for each feature
    avg_similarity1 = np.mean(similarity1[:, cluster_indices], axis=1)
    avg_similarity3 = np.mean(similarity3[:, cluster_indices], axis=1)
    
    # add the average cosine similarity to the dictionary for each feature
    avg_similarity_dict1[cluster_label] = avg_similarity1[0]
    avg_similarity_dict3[cluster_label] = avg_similarity3[0]


tsne_embeddings = TSNE(n_components=3).fit_transform(learned_embeddings1)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Define custom colormap
colors = ['red', 'black', 'yellow']
cmap = mcolors.ListedColormap(colors)

fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], tsne_embeddings[:, 2], c=labels1, cmap=cmap)

# Add average similarity values to corresponding clusters
unique_labels = np.unique(labels3)

avg_sim_labels = []
for label in unique_labels:
    avg_sim = avg_similarity_dict1[label]
    avg_sim_str = f"{avg_sim:.3f}"
    avg_sim_labels.append(avg_sim_str)

# Create custom legend
handles, labels = scatter.legend_elements()
legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
legend_handles = handles
legend = ax.legend(legend_handles, legend_labels, loc='lower left', title='Clusters', frameon=True)
legend.get_title().set_fontsize(16)

# Add second legend for average similarity
ax.add_artist(legend)
avg_sim_legend = ax.legend(legend_handles, avg_sim_labels, loc='lower right', title='avg_sim_clusters', frameon=True)
avg_sim_legend.get_title().set_fontsize(16)

# Set plot title
avg_similarity1f = avg_similarity1[0]
ax.set_title(f'Average Similarity = {avg_similarity1f:.3f}', fontsize=20)

# Set axis labels
ax.set_xlabel('t-SNE Dimension 1', fontsize=16)
ax.set_ylabel('t-SNE Dimension 2', fontsize=16)
ax.set_zlabel('t-SNE Dimension 3', fontsize=16)

plt.show()




##################################
#   hyperband optimal 1 text     #
##################################

tf.keras.utils.set_random_seed(87031800)

inp_num = keras.layers.Input(shape=(Xtrain.shape[1],), name='Input_var')
inp_seq1 = keras.layers.Input(shape=(seq1_train.shape[1],), name='Input_seq1')

# embedding layers for the sequences
emb_seq1 = keras.layers.Embedding(input_dim=vocab_size1, output_dim=embedding_dim, input_length=max_len1, name='embedding_et')(inp_seq1)
# global max pooling of the embedded sequences
pooled_seq1 =  keras.layers.GlobalMaxPooling1D()(emb_seq1)

# concatenate the pooled sequences and numerical input
concat = keras.layers.Concatenate()([pooled_seq1, inp_num])

dense1 = keras.layers.Dense(93, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.07), name = '1')(concat)
dense2 = keras.layers.Dense(97, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.05), name = '2')(dense1)
dense3 = keras.layers.Dense(69, activation=tf.nn.relu, name = '3')(dense2)
dense4 = keras.layers.Dense(93, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.05), name = '4')(dense3)
Attention = keras.layers.Dense(Xtrain.shape[1], activation="linear", name = 'Attention')(dense4)

Dot = keras.layers.Dot(axes=1)([inp_num, Attention])

Response = keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='Response')(Dot)
model = keras.Model(inputs=[inp_num, inp_seq1], outputs=Response)
print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.compile(loss=Deviance(eps=1e-5, w=1), optimizer=tf.optimizers.Adam(0.001))

history = model.fit([Xtrain, seq1_train], Ytrain, epochs=51, batch_size = len(Xtrain),validation_split=0.2, callbacks=[callback], verbose=2)


to_see = model.predict([Xtest, seq1_test])
pred_calib = pd.DataFrame(to_see, columns=["pred"])
to_see = np.column_stack((to_see, Ytest))
to_mean = to_see[to_see[:, 1] == 1]
mean_saturated = np.mean(to_see[:, 0])

y_pred = to_see[:,0]
y_true = to_see[:,1]
deviance = -2*K.sum(y_true*K.log((y_pred+K.epsilon())/(1-y_pred+K.epsilon())) + K.log(1-y_pred+K.epsilon()))
deviance_saturated = deviance.numpy()

calib_test = calib_test.reset_index(drop=True)
pred_calib = pred_calib.reset_index(drop=True)
calib_test = pd.concat([calib_test, pred_calib], axis = 1)


def my_fun(x):
    return np.mean(x)

groups = calib_test.columns[:14].tolist()

to_comp1 = calib_test.groupby(groups)['loan_status'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)
to_comp2 = calib_test.groupby(groups)['pred'].agg([('my_fun', my_fun), ('count', 'count')]).reset_index(drop=True)

O = to_comp1['my_fun']*to_comp1['count']
E = to_comp2['my_fun']*to_comp2['count']
n = to_comp1['count']
HLT = np.sum((O - E)**2/(E*(1-to_comp2['my_fun']) + 1e-10))
pval = 1 - chi2.cdf(HLT, df=(len(to_comp1)-2))

plt.figure(figsize=(7,6), facecolor="white")
plt.scatter(to_comp2['my_fun'], to_comp1['my_fun'], c='blue')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Predicted Frequency')
plt.ylabel('True Frequency')
plt.title('Group Calibration (LocalGLMnet)')
plt.legend([r'$\hat{{C}}$ = {:.3f}'.format(HLT), 'p-value = {:.3f}'.format(pval)], loc='upper left')
plt.grid()


embedding_layer1 = model.get_layer('embedding_et')

learned_embeddings1 = embedding_layer1.get_weights()[0]

from sklearn.cluster import KMeans
# set the range of number of clusters to try
cluster_range = range(1, 31)

# define a function to compute the WCSS for each number of clusters
def compute_wcss(embedding):
    wcss_list = []
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(embedding)
        wcss = kmeans.inertia_
        wcss_list.append(wcss)
    return wcss_list

# compute the WCSS for each embedding space
wcss_list1 = compute_wcss(learned_embeddings1)

# plot the WCSS against the number of clusters
plt.plot(cluster_range, wcss_list1, label='Embedding 1')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.legend()
plt.show()


kmeans1 = KMeans(n_clusters=3, random_state=0).fit(learned_embeddings1)
labels1 = kmeans1.labels_


# create a dictionary that maps index to word for each feature
index_to_word1 = {index: word for word, index in word_index1.items()}

# create a dictionary that maps index to embedding for each feature
index_to_embedding1 = {index: embedding for index, embedding in enumerate(learned_embeddings1)}

# create a dictionary that maps cluster label to embeddings for each feature
cluster_embedding_dict1 = {}

for i, cluster_label in enumerate(labels1):
    # get the word index for the first feature
    word_index1 = index_to_word1.get(i+1)  # add 1 to index to account for reserved index 0
    
    # get the embedding for the first feature
    embedding1 = index_to_embedding1[i]
    
    # add the embedding to the dictionary for the first feature
    if cluster_label not in cluster_embedding_dict1:
        cluster_embedding_dict1[cluster_label] = [embedding1]
    else:
        cluster_embedding_dict1[cluster_label].append(embedding1)

from sklearn.metrics.pairwise import cosine_similarity

# compute the average cosine similarity for each cluster and feature
avg_similarity_dict1 = {}
for cluster_label in cluster_embedding_dict1:
    # compute the average embedding for each cluster and feature
    avg_embedding1 = np.mean(cluster_embedding_dict1[cluster_label], axis=0)    
    # compute the cosine similarity between each word embedding and the average embedding
    similarity1 = cosine_similarity(avg_embedding1.reshape(1,-1), learned_embeddings1)
    
    # get the indices of the words in the cluster
    cluster_indices = [i for i, label in enumerate(labels1) if label == cluster_label]
    
    # compute the average cosine similarity for each feature
    avg_similarity1 = np.mean(similarity1[:, cluster_indices], axis=1)
    
    # add the average cosine similarity to the dictionary for each feature
    avg_similarity_dict1[cluster_label] = avg_similarity1[0]

from sklearn.manifold import TSNE


tsne_embeddings = TSNE(n_components=3).fit_transform(learned_embeddings1)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Define custom colormap
colors = ['red', 'black', 'yellow']
cmap = mcolors.ListedColormap(colors)

fig = plt.figure(figsize=(11, 11))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], tsne_embeddings[:, 2], c=labels1, cmap=cmap)

# Add average similarity values to corresponding clusters
unique_labels = np.unique(labels1)

avg_sim_labels = []
for label in unique_labels:
    avg_sim = avg_similarity_dict1[label]
    avg_sim_str = f"{avg_sim:.3f}"
    avg_sim_labels.append(avg_sim_str)

# Create custom legend
handles, labels = scatter.legend_elements()
legend_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
legend_handles = handles
legend = ax.legend(legend_handles, legend_labels, loc='lower left', title='Clusters', frameon=True)
legend.get_title().set_fontsize(16)

# Add second legend for average similarity
ax.add_artist(legend)
avg_sim_legend = ax.legend(legend_handles, avg_sim_labels, loc='lower right', title='avg_sim_clusters', frameon=True)
avg_sim_legend.get_title().set_fontsize(16)

# Set plot title
avg_similarity1f = avg_similarity1[0]
ax.set_title(f'Average Similarity = {avg_similarity1f:.3f}', fontsize=20)

# Set axis labels
ax.set_xlabel('t-SNE Dimension 1', fontsize=16)
ax.set_ylabel('t-SNE Dimension 2', fontsize=16)
ax.set_zlabel('t-SNE Dimension 3', fontsize=16)

plt.show()

##################################
#   SHAPLEY VALUES EXTRACTION    #
##################################

# This is the way we create our surrogate Shapley values model for the localGLMnet, it's comparable to the way we build SV for any ML implementation
# If you run other architectures, run the optimal Min max scaled architecture for the LocalGLMnet
import shap

X = data_without_my.drop("loan_status", axis= 1)
X = X.drop('Scaled_recoveries', axis=1)
Y = data_without_my["loan_status"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=87031800)
calib_test = pd.concat([Xtest, Ytest], axis = 1)
calib_train = pd.concat([Xtrain, Ytrain], axis = 1)

X_1000 = shap.utils.sample(Xtrain, 1000)
explainer = shap.Explainer(model.predict, X_1000)
shap_values = explainer(Xtest)  # Pay attention this may take a long time to run 
shapley_values = shap_values.values
base_val = np.mean(shap_values.base_values)
# set the background color to white

# Summary of shapley values 
fig = plt.figure(figsize = (20,10), facecolor='white')
shap.summary_plot(shap_values, Xtest, plot_type='dot', max_display=Xtest.shape[1])
plt.show()

# Dependence plot, allows to understand dependences between categorical binary encoded features and continuous ones
for i in range(Xtest.shape[1]):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    shap.dependence_plot(i, shapley_values, Xtest, ax=ax)
    ax.set_title(Xtest.columns[i])
    plt.show()
    
fig = plt.figure(figsize = (20,10), facecolor='white')
shap.plots.bar(shap_values, max_display=Xtest.shape[1], show=False)
plt.show()

# Force plot to understand the impact of features on predictions accross your portfolio 
from IPython.display import display, HTML

shap.initjs() 
force_plot  = shap.force_plot(base_val, shapley_values, features = Xtest, feature_names = Xtest.columns)
force_plot_html = force_plot.html()
display(force_plot)

shap.save_html('force_plot.html', force_plot)



