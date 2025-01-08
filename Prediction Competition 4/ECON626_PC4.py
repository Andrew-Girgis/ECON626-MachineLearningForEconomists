# %% [markdown]
# # ECON 626 Prediction Competition 4 Code
# 

# %% [markdown]
# Objective: Utilize regression algorithms (linear regression, LASSO, Ridge, Subset Selection) to train a model that predicts the natural logarithm of car price.

# %% [markdown]
# ## Importing Librarys

# %%
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model # Linear regression
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder


# %% [markdown]
# ## Importing data

# %%
small_data_path = "/Users/andrew/Downloads/UW courses/ECON 626/Prediction Competition 4/pc3_and_pc4_training_data_small_v1.csv"
small_df= pd.read_csv(small_data_path)
#create a dataframe for our smaller dataset

large_data_path = "/Users/andrew/Downloads/UW courses/ECON 626/Prediction Competition 4/pc3_and_pc4_training_data_large_v1.csv"
large_df = pd.read_csv(large_data_path)
#create a dataframe for our larger dataset

test_data_path = "/Users/andrew/Downloads/UW courses/ECON 626/Prediction Competition 4/pc4_test_data_without_response_v1.csv"
test_df = pd.read_csv(test_data_path)
#create a dataframe for our larger dataset

total_df  = pd.concat([small_df, large_df], axis = 0)
#create a dataframe containing both small and large df

# %% [markdown]
# ## Inspect data

# %%
#function:
def inspect_dataset(dataset):
    # Print the head of the dataset
    print("Head of the dataset:")
    print(dataset.head())
    print("\n")

    # Print the info of the dataset
    print("Info of the dataset:")
    print(dataset.info())
    print("\n")

    # Print the shape of the dataset
    print("Shape of the dataset:")
    print(dataset.shape)
    print("\n")

    # Print value counts for columns of type object
    object_columns = dataset.select_dtypes(include=['object']).columns
    for column in object_columns:
        print(f"Value counts for column '{column}':")
        print(dataset[column].value_counts())
        print("\n")

# %%
small_df.head()

# %%
small_df.info()

# %%
small_df.isna().sum()

# %%
small_df.shape

# %% [markdown]
# ## Data Visualizations

# %%
small_df['body_type'].value_counts().plot(kind='bar')

# %%
small_df['fuel_type'].value_counts().plot(kind='bar')

# %%
small_df['wheel_system'].value_counts().plot(kind='bar')


# %%
numerical_columns = small_df.select_dtypes(include=np.number).columns.tolist()

num_drop_list = 'latitude', 'price', 'longitude'

numerical_columns = list(set(numerical_columns) - set(num_drop_list))

# %%
categorical_columns = small_df.select_dtypes(exclude=np.number).columns.tolist()
categorical_columns

cat_drop_list = 'back_legroom', 'height', 'length', 'listed_date', 'wheelbase', 'width', 'exterior_color'

categorical_columns = list(set(categorical_columns) - set(cat_drop_list))

# %%
plt.figure(figsize = (50, 25))
for i, value in enumerate(numerical_columns[:-1]):
    plt.subplot(2,2,i+1)
    plt.title( value + ' vs price')
    sns.scatterplot(data = large_df, x = value, y = 'price', hue = 'wheel_system')

# %%
plt.figure(figsize = (20, 15))
for i, value in enumerate(categorical_columns):
    plt.subplot(2,2,i+1)
    plt.title( value + ' vs price')
    sns.boxplot(data = large_df, x = value, y = 'price')

# %%
numeric_df = small_df.select_dtypes(include='number')

# Create pairplot
sns.pairplot(numeric_df)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

og_cols = ['price', 'engine_displacement', 'highway_fuel_economy', 'horsepower', 'mileage', 'year']
num_cols = len(og_cols)

# Calculate the number of rows and columns needed for subplots
num_rows = (num_cols + 2) // 3  # Ceiling division to ensure we have enough rows
num_cols = min(num_cols, 3)  # Limit the number of columns to 3

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))  # Adjust figsize as needed

for i, col in enumerate(og_cols):
    row_idx = i // num_cols
    col_idx = i % num_cols

    if num_rows == 1:
        ax = axes[col_idx]
    else:
        ax = axes[row_idx, col_idx]

    sns.histplot(small_df[col], ax=ax, kde=True, line_kws={'lw': 5, 'ls': ':'})  # Add KDE plot
    sns.histplot(test_df[col], ax=ax, kde=True, alpha=0.25)
    
    ax.set_title('Distribution of ' + col)

# Adjust layout to prevent overlap of subplots
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns

og_cat_cols = ['body_type', 'fuel_type', 'wheel_system']
num_cols = len(og_cat_cols)

# Calculate the number of rows and columns needed for subplots
num_rows = (num_cols + 2) // 3  # Ceiling division to ensure we have enough rows
num_cols = min(num_cols, 3)  # Limit the number of columns to 3

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))  # Adjust figsize as needed

for i, col in enumerate(og_cat_cols):
    row_idx = i // num_cols
    col_idx = i % num_cols
    
    if num_rows == 1:
        ax = axes[col_idx]
    else:
        ax = axes[row_idx, col_idx]

    sns.histplot(small_df[col], ax=ax, kde=True, line_kws={'lw': 5, 'ls': ':'}) 
    sns.histplot(test_df[col], ax=ax, kde=True, alpha=0.25)
    
    ax.set_title('Distribution of ' + col)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# Adjust layout to prevent overlap of subplots
plt.tight_layout()
plt.show()


# %%
    small_df['listed_date'] = pd.to_datetime(small_df['listed_date'])
    listed_year = small_df['listed_date'].dt.year
    small_df['age_at_listing'] = listed_year - small_df['year']

# %%
small_df

# %% [markdown]
# ## Data Preprocessing

# %%
def extract_numeric_value(text):
    # Use regular expression to extract numeric value
    match = re.match(r'(\d+)', str(text))
    if match:
        return int(match.group(1))
    else:
        return None

# %%
#Testing
# small_df['length'] = pd.to_numeric(small_df['length'].str.replace(' in', ''), errors='coerce')
# small_df['length']

small_df['length'] = pd.to_numeric(small_df['length'].astype(str).str.replace(' in', ''), errors='coerce')
small_df['width'] = pd.to_numeric(small_df['width'].astype(str).str.replace(' in', ''), errors='coerce')
small_df['height'] = pd.to_numeric(small_df['height'].astype(str).str.replace(' in', ''), errors='coerce')

small_df.isna().sum()

#small_df["car_vol"] = small_df["length"] * small_df["width"]* small_df["height"]

# %%


# %%
small_df[small_df['length'].isnull()].head()

# %%
#Groups the lengths by catagorys then fills NA with mean from that catagory
small_df['length'] = small_df['length'].fillna(small_df.groupby('body_type')['length'].transform('mean'))
small_df['width'] = small_df['width'].fillna(small_df.groupby('body_type')['width'].transform('mean'))
small_df['height'] = small_df['height'].fillna(small_df.groupby('body_type')['height'].transform('mean'))

# %%
small_df.isna().sum()


# %%
small_df['fuel_type'].value_counts()

# %%
#Creating a function to preprocess the data
def prep_data(dataset):
    
    dataset_cols = dataset.columns
    if 'price' in dataset_cols:
        dataset['log_price'] = np.log(dataset['price'])
    else:
        pass

    #Feature Engineering
    dataset['length'] = pd.to_numeric(dataset['length'].astype(str).str.replace(' in', ''), errors='coerce')
    dataset['length'] = dataset['length'].fillna(dataset.groupby('body_type')['length'].transform('mean'))

    dataset['width'] = pd.to_numeric(dataset['width'].astype(str).str.replace(' in', ''), errors='coerce')
    dataset['width'] = dataset['width'].fillna(dataset.groupby('body_type')['width'].transform('mean'))

    dataset['height'] = pd.to_numeric(dataset['height'].astype(str).str.replace(' in', ''), errors='coerce')
    dataset['height'] = dataset['height'].fillna(dataset.groupby('body_type')['height'].transform('mean'))

    # dataset['wheelbase'] = pd.to_numeric(dataset['wheelbase'].astype(str).str.replace(' in', ''), errors='coerce')
    # dataset['wheelbase'] = dataset['wheelbase'].fillna(dataset.groupby('body_type')['wheelbase'].transform('mean'))

    dataset["car_vol"] = dataset["length"] * dataset["width"]* dataset["height"]

    dataset['listed_date'] = pd.to_datetime(dataset['listed_date'])
    listed_year = dataset['listed_date'].dt.year
    dataset['age_at_listing'] = listed_year - dataset['year']

    # fuel_type_map = {'Gasoline': 1, 'Diesel': 0, 'Flex Fuel Vehicle' : 0, 'Hybrid': 0, 'Biodiesel' : 0, 'Compressed Natural Gas' : 0  }

    # # Apply the mapping to the 'fuel_type' column
    # dataset['fuel_type_binary'] = dataset['fuel_type'].map(fuel_type_map)

    drop = ['price', 'back_legroom', 'wheelbase', 'latitude', 'longitude', 'listed_date', 'exterior_color']
    for col in drop:
        dataset = dataset.drop([col], axis=1)
    
    col_encode = [ 'body_type', 'fuel_type', 'wheel_system']
    le = LabelEncoder()
    for col in col_encode:
        new_col = col+'_enc'
        dataset[new_col] = le.fit_transform(dataset[col])
    dataset['litres'] = (dataset['engine_displacement']/1000).astype(float)
    dataset = dataset.drop(['engine_displacement'], axis=1)
    drop_enc= ['body_type', 'fuel_type', 'wheel_system']
    for col in drop_enc:
        dataset = dataset.drop([col], axis=1)
    return dataset

# %%
train_df = prep_data(total_df)

# %%
test_df = prep_data(test_df)

# %%
train_df

# %%
test_df

# %%

#small_df['fuel_type_binary'].value_counts()

# %%
drop_low_importance = ['age_at_listing', 'car_vol', 'width']

for col in drop_low_importance:
    train_df = train_df.drop([col], axis=1)
    test_df = test_df.drop([col], axis=1)

#After running through the predictions previously I have found that these two columns are not important in predicting the price.

# %% [markdown]
# ## Linear Regression

# %%
def split_data(dataset):
    global x_train, x_val_test, y_train, y_val_test, x_val, x_test, y_val, y_test
    #Must make the variables global to access the variables outside of the function

    columns_x = list(dataset.columns)
    if 'log_price' in columns_x:
        columns_x.remove('log_price')
    else:
        pass
    x_train, x_val_test, y_train, y_val_test = train_test_split(dataset[columns_x], dataset['log_price'], test_size=0.2, random_state=123)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=123)
    
    #return x_train, x_val, x_test, y_train, y_val, y_test
    

# %%
split_data(train_df)

# %%
print(x_train.shape, x_val.shape, x_test.shape)

# %%
x_train.head()

# %%
x_train.isna().sum()

# %%
min_max_scaler = preprocessing.MinMaxScaler()

min_max_scaler.fit(x_train)
# transform
x_train_scaled = min_max_scaler.transform(x_train)
x_val_scaled = min_max_scaler.transform(x_val)
x_test_scaled = min_max_scaler.transform(x_test)

# %%

#  Create linear regression object
lr = linear_model.LinearRegression()

# Train the model using the training set
lr.fit(x_train_scaled, y_train)

# Make predictions on the training and validation sets
y_train_pred_lr = lr.predict(x_train_scaled) 
y_val_pred_lr = lr.predict(x_val_scaled)
y_test_pred_lr = lr.predict(x_test_scaled)

# You can use either x_train or x_train_scaled with regression models. 
# To easily interpret the coefficients, unscaled variables are preferred.

# %%
 # Print sq root of MSE on both sets
print('MSE root and mean on training set:', mean_squared_error(y_train, y_train_pred_lr)**0.5,  y_test.mean())
print('MSE root and mean on validation set:', mean_squared_error(y_val, y_val_pred_lr)**0.5, y_test.mean())
print('MSE root and mean on test set:', mean_squared_error(y_test, y_test_pred_lr)**0.5, y_test.mean())
# Print R squared on both sets
print('R squared on training set:', round(r2_score(y_train, y_train_pred_lr),3))
print('R squared on validation set:', round(r2_score(y_val, y_val_pred_lr), 3))
print('R squared on test set:', round(r2_score(y_test, y_test_pred_lr), 3))

# %% [markdown]
# ## LASSO

# %%
lr_lasso = linear_model.Lasso(alpha=0.0005) #alpha is the lambda in the regularization formula
lr_lasso.fit(x_train_scaled, y_train)

# Make predictions on the training and validation sets
y_train_pred = lr_lasso.predict(x_train_scaled) 
y_val_pred = lr_lasso.predict(x_val_scaled)
y_test_pred = lr_lasso.predict(x_test_scaled)

# %%
# Print sq root of MSE on both sets
print('MSE and mean on training set:', mean_squared_error(y_train, y_train_pred)**0.5,  y_test.mean())
print('MSE and mean on validation set:', mean_squared_error(y_val, y_val_pred)**0.5, y_test.mean())
print('MSE and mean on test set:', mean_squared_error(y_test, y_test_pred)**0.5, y_test.mean())
# Print R squared on both sets
print('R squared on training set:', r2_score(y_train, y_train_pred))
print('R squared on validation set:', r2_score(y_val, y_val_pred))
print('R squared on test set:', r2_score(y_test, y_test_pred))

# %%
coefficients = pd.DataFrame()
coefficients['feature_name'] = x_train.columns
coefficients['coefficients'] = pd.Series(lr_lasso.coef_)
coefficients

# %% [markdown]
# ### LASSO Hyperparameter tuning

# %%
lambdas = 1 * 0.90 ** np.arange(1,100)

# %%
best_lambda = None
r2 = 0
# Step 2
# Estimate Lasso regression for each regularization parameter in grid
# Save if performance on validation is better than that of previous regressions
for lambda_j in lambdas:
    linear_reg_j = linear_model.Lasso(alpha = lambda_j)
    linear_reg_j.fit(x_train_scaled, y_train)
    # evaluate on validation set
    y_val_pred_j = linear_reg_j.predict(x_val_scaled)
    r2_j = r2_score(y_val, y_val_pred_j)
    if r2_j > r2:
        best_lambda = lambda_j
        r2 = r2_j
print(best_lambda, r2)

# %%
x_train_scaled_final = np.concatenate((x_train_scaled, x_val_scaled))
y_train_final = pd.concat([y_train,y_val], axis = 0)
lr_lasso_best = linear_model.Lasso(alpha = best_lambda)
lr_lasso_best.fit(x_train_scaled_final, y_train_final)

# %%
y_test_pred = lr_lasso_best.predict(x_test_scaled)
# Print MAPE 
print('MSE and mean on test set:', mean_squared_error(y_test, y_test_pred), y_test.mean())
# Print R squared 
print('R squared on test set:', r2_score(y_test, y_test_pred))

# %%
from sklearn.linear_model import LassoCV
lr_lasso_cv = LassoCV(cv=10, alphas= lambdas)
lr_lasso_cv.fit(x_train_scaled_final, y_train_final)

# %%
lr_lasso_cv.alpha_

# %%
y_test_pred = lr_lasso_cv.predict(x_test_scaled)
# Print MAPE 
print('MSE and mean on test set:', mean_squared_error(y_test, y_test_pred)**0.5, y_test.mean())

r2_cv = r2_score(y_test, y_test_pred)

# Print R squared 
print('R squared on test set:', r2_cv)

# %%
coefficients = pd.DataFrame()
coefficients['feature_name'] = x_train.columns
coefficients['coefficients_val_best'] = pd.Series(lr_lasso_best.coef_)
coefficients['coefficients_cv'] = pd.Series(lr_lasso_cv.coef_)
coefficients

# %%

coefficients = pd.DataFrame()
coefficients['feature_name'] = x_train.columns
coefficients['coefficients'] = pd.Series(lr_lasso.coef_)


# Sort the coefficients by absolute value
coefficients = coefficients.reindex(coefficients['coefficients'].abs().sort_values(ascending=False).index)

# Plot the variable importance
plt.figure(figsize=(6, 4))
sns.barplot(data=coefficients, x='coefficients', y='feature_name', palette='magma')
plt.xlabel('Coefficient')
plt.ylabel('Feature Name')
plt.title('Variable Importance (LASSO)')

# Add a dotted vertical line at x = 0
plt.axvline(x=0, color='black', linestyle='--')


plt.show()

# %%
small_df.head()

# %%
test_df = test_df.drop('log_price', axis=1)
test_df.head()

# %%
min_max_scaler.fit(test_df)

# transform
X_final_scaled = min_max_scaler.transform(test_df)

# Make predictions on the test set using the lr_lasso_cv model
final_pred_test = lr_lasso_cv.predict(X_final_scaled)

# Print the predictions
print(final_pred_test)



# %%
predictions_df = pd.DataFrame({'predictions': final_pred_test})

header = pd.DataFrame({
    'predictions': [21108082, 'GojoSatoru', round(r2_cv,3)]
})

header

output_df = pd.concat([header, predictions_df], axis=0)

output_df.to_csv('predictions_output.csv', index=False, header=False)


