#!/usr/bin/env python
# coding: utf-8

# <a id='Q0'></a>
# <center><a target="_blank" href="http://www.propulsion.academy"><img src="https://drive.google.com/uc?id=1McNxpNrSwfqu1w-QtlOmPSmfULvkkMQV" width="200" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <center> <h4 style="color:#303030"> Python for Data Science, Homework, template: </h4> </center>
# <center> <h1 style="color:#303030">Predict the quality of white wine from its physico-chemical properties</h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center style="color:#303030"><h4>Propulsion Academy, 2021</h4></center>
# <p style="margin-bottom:1cm;"></p>
# 
# <div style="background:#EEEDF5;border-top:0.1cm solid #EF475B;border-bottom:0.1cm solid #EF475B;">
#     <div style="margin-left: 0.5cm;margin-top: 0.5cm;margin-bottom: 0.5cm">
#         <p><strong>Goal:</strong> Practice Linear Regression on wine data</p>
#         <strong> Sections:</strong>
#         <a id="P0" name="P0"></a>
#         <ol>
#             <li> <a style="color:#303030" href="#SU">Set Up </a> </li>
#             <li> <a style="color:#303030" href="#P1">Exploratory Data Analysis</a></li>
#             <li> <a style="color:#303030" href="#P2">Modeling</a></li>
#         </ol>
#         <strong>Topics Trained:</strong> Clustering.
#     </div>
# </div>
# 
# <nav style="text-align:right"><strong>
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/" title="momentum"> SIT Introduction to Data Science</a>|
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/weeks/week2/day1/index.html" title="momentum">Week 2 Day 1, Applied Machine Learning</a>|
#         <a style="color:#00BAE5" href="https://colab.research.google.com/drive/1DK68oHRR2-5IiZ2SG7OTS2cCFSe-RpeE?usp=sharing" title="momentum"> Assignment, Wine Quality Prediction</a>
# </strong></nav>

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# ### Installing

# In[1]:


get_ipython().system(u'pip3 install -U scikit-learn')
get_ipython().system(u'curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
get_ipython().system(u'pip install auto-sklearn')
get_ipython().system(u'sudo apt install cookiecutter')
get_ipython().system(u'pip install gdown')
get_ipython().system(u'pip install dvc')
get_ipython().system(u"pip install 'dvc[gdrive]'")


# ### Importing

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import set_config
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import logging
from joblib import dump
import datetime


# ### Google Drive connection

# In[2]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# ### Options and settings

# In[3]:


data_path = "/content/drive/MyDrive/Introduction2DataScience/data/"
model_path = "/content/drive/MyDrive/Introduction2DataScience/w2d2/models/"


# In[4]:


timesstr = str(datetime.datetime.now()).replace(' ', '_')
logging.basicConfig(filename=f"{model_path}explog_{timesstr}.log", level=logging.INFO)


# Please Download the data from [this source](https://drive.google.com/file/d/1gncbcW3ow8vDz_eyrvgDYwiMgNrsgwzz/view?usp=sharing), and upload it on your Introduction2DS/data google drive folder.

# In[5]:


set_config(display='diagram')


# In[6]:


wine = pd.read_csv(f'{data_path}winequality-red.csv', sep=';')


# In[7]:


test_size = 0.2
random_state = 0
train, test = train_test_split(wine, test_size=test_size, random_state=random_state)
train.to_csv(f'{data_path}winequality-red-train.csv', index=False, sep=';')
train = train.copy()
test.to_csv(f'{data_path}winequality-red-test.csv', index=False, sep=';')
test = test.copy()


# In[8]:


logging.info(f'train test split with test_size={test_size} and random state={random_state}')


# <a id='P2' name="P2"></a>
# ## [Modelling](#P0)

# In[9]:


X_train, y_train = train.iloc[:,:-1], train.iloc[:,-1] 


# ### Pipeline Definition

# In[10]:


regr = LinearRegression()
model = Pipeline(steps=[('regressor', regr)])


# ### Model Training

# In[11]:


import autosklearn.regression


# In[12]:


automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=600,
    per_run_time_limit=30,
)
automl.fit(X_train, y_train)


# In[13]:


total_time=600
per_run_time=30


# In[14]:


logging.info(f'Ran autosklearn regressor for a total time of {total_time} seconds, with a maximum of {per_run_time} seconds per model run')


# In[15]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[16]:


logging.info(f'Saved regressor model at {model_path}model{timesstr}.pkl ')


# In[17]:


logging.info(f'autosklearn model statistics:')
logging.info(automl.sprint_statistics())


# ### Model Evaluation

# In[18]:


X_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]


# In[19]:


y_pred = automl.predict(X_test)


# In[20]:


from sklearn.metrics import mean_squared_error


# In[21]:


mean_squared_error(y_test, y_pred)
automl.score(X_test, y_test)


# In[22]:


df = pd.DataFrame(np.concatenate((X_test, y_test.to_numpy().reshape(-1,1), y_pred.reshape(-1,1)),  axis=1))


# In[23]:


df


# In[24]:


df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality True', 'quality Predicted']


# In[25]:


fig = px.scatter(df, x='quality True', y='quality Predicted')
fig.show()


# In[26]:


get_ipython().system(u'pip install shap')


# In[27]:


import shap


# In[28]:


explainer = shap.KernelExplainer(model = automl.predict, data = X_test.iloc[:50, :], link = "identity")


# In[29]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X = X_test.iloc[X_idx:X_idx+1,:], nsamples = 100)
X_test.iloc[X_idx:X_idx+1,:]
# print the JS visualization code to the notebook
shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = X_test.iloc[X_idx:X_idx+1,:]
                )


# In[30]:


shap_values = explainer.shap_values(X = X_test.iloc[0:50,:], nsamples = 100)


# In[31]:


# print the JS visualization code to the notebook
shap.initjs()
shap.summary_plot(shap_values = shap_values,
                  features = X_test.iloc[0:50,:]
                  )


# In[32]:


mse = mean_squared_error(y_test, y_pred)
mse


# In[33]:


R_squared = automl.score(X_test, y_test)
R_squared


# In[34]:


logging.info(f"Mean Squared Error is {mse}, \n R2 score is {R_squared}")

