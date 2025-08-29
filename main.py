"""
https://www.kaggle.com/competitions/ds-20-basic-ml-regression-predict-machine-price

Goal: Develop a model to predict the auction sale prices of heavy machinery based on past auction data.
      Ensure your model minimizes the RMSE to achieve a better score in the competition.

For each ID in the validation set, you must predict the SalePrice for the machine. The file should contain a header and be formatted as follows:

SalesID,SalePrice
1,10000
2,15000
...
Make sure your submission file includes all IDs from the validation set and follows the specified format to be successfully scored.


Dataset Description:

The dataset contains detailed information about historical sales of heavy machinery. Below is a description of the variables included:

    SalesID: Unique identifier for a particular sale.
    MachineID: Identifier for a particular machine; machines may have multiple sales.
    ModelID: Identifier for a unique machine model.
    Datasource: Source of the sale record; different sources may vary in reporting detail.
    AuctioneerID: Identifier for the auctioneer company that sold the machine.
    YearMade: Year the machine was manufactured.
    MachineHoursCurrentMeter: Current usage of the machine in hours at the time of sale; null or 0 indicates no hours reported.
    UsageBand: Usage level (low, medium, high) compared to the average usage for the model.
    SaleDate: Date of sale.
    SalePrice: Sale price in USD.
    fiModelDesc: Description of a unique machine model.
    fiBaseModel, fiSecondaryDesc, fiModelSeries, fiModelDescriptor: Disaggregated parts of the model description.
    ProductSize: Size class grouping for a product group.
    ProductClassDesc: Description of the second level hierarchical grouping of the model.
    State: US state where the sale occurred.
    ProductGroup: Identifier for the top-level hierarchical grouping of the model.
    ProductGroupDesc: Description of the top-level hierarchical grouping of the model.
    Drive_System: Machine configuration, typically describing wheel drive.
    Enclosure: Indicates if the machine has an enclosed cab.
    Forks: Attachment used for lifting.
    Pad_Type: Type of treads a crawler machine uses.
    Ride_Control: Optional feature on loaders for smoother rides.
    Stick: Type of control.
    Transmission: Describes type of transmission (automatic or manual).
    Turbocharged: Engine type (naturally aspirated or turbocharged).
    Blade_Extension: Extension of the standard blade.
    Blade_Width: Width of the blade.
    Enclosure_Type: Describes if the machine has an enclosed cab.
    Engine_Horsepower: Engine horsepower rating.
    Hydraulics: Type of hydraulics system.
    Pushblock: Optional feature.
    Ripper: Implement attached to machine for tilling soil.
    Scarifier: Implement attached to machine for soil conditioning.
    Tip_control: Type of blade control.
    Tire_Size: Size of primary tires.
    Coupler: Type of implement interface.
    Coupler_System: Type of implement interface.
    Grouser_Tracks: Describes ground contact interface.
    Hydraulics_Flow: Normal or high flow hydraulic system.
    Track_Type: Type of treads a crawler machine uses.
    Undercarriage_Pad_Width: Width of crawler treads.
    Stick_Length: Length of machine digging implement.
    Thumb: Attachment used for grabbing.
    Pattern_Changer: Can adjust the operator control configuration.
    Grouser_Type: Type of treads a crawler machine uses.
    Backhoe_Mounting: Optional interface for adding a backhoe attachment.
    Blade_Type: Describes type of blade.
    Travel_Controls: Describes operator control configuration.
    Differential_Type: Differential type, typically locking or standard.
    Steering_Controls: Describes operator control configuration.
"""

"""
Lessons
    13.08:
        Feature Engineering Random Forest Regression
        Data cleansing
             - mode is good for categorial values fill of missing values   
             - preventing data leakage only fill of missing values based on training dataset and not on all the data
               and impute or fill them on training and validation data.
             - from sklearn.impute import SimpleImputer lesson 17/08 time: 2:00:03                
             - TargetEncode: 
                te=TargetEncode()
                X_train[cat_cols]=te.fit_transform(X_train[cat_cols], y_train)) -- smoothing helps with rare categories
             - cut data in bins of equal size and use this as a new feature with pd.qcut(df.carat, q=5, labels=['xs','s','m','l','xl'])  for example
               qcut() = quartile cuts and cut() bin sizes lesson 17/8 time: 2:26:00  
               X_train['price_per_cut_carat'] = X_train.assign(price=y_train).groupby(['cut', 'carat_cat']).price.transform('mean')
X_train.head()
        Feature importance
        train test split
            Data leakage - lookout for time dependent data
                           far historical data or delete or transform to be relevant today 
        Hyper parameters (last phase)
            max_features: does take a maximum number of random selection of features from the available features
                            minimizes the train and test results
            max_samples: each tree is given a specified fraction/percentage of the records to learn from
                            minimizes the train and test results
            tuning of hyper parameters: first build a model, validate and only then start playing with parameters
        Results: most situations train should be better than test, if not then a bug in program


        permutation performance: importance of features on the training

from sklearn.inspection import permutation_performance

when random forest and permutation performance disagree the random forest regressor is wrong.

what can we do? We can change the cardinality of the specific table



Tractor dataset

- Sales Date very important for model

- tip1: check validation data vs test data too to get hints of which data you can expect

- tip2: find the model description column that gives description of horsepower etc. - FirstProductClassDesc

        make of this feature new features

- remove irrelevant features

- imputing nans

- label encoding of categorial columns + another encoding style

- dummies

- ordering of categorial values - target encoding if no ordering exists and sorting when ordering exist but not in a natural way

- feature importance testing

- correct of splitting train and test data - train the past and test the future -> date?

- random forest hyper parameters tuning - only at the end of the process

--> validation set - what is similar or different with the test data

--> split the data so to make the test data similar to the validation data

--> check if there are vehicles which are not tractors, if exist then delete

Isolation Forest:
when non-valid data exists (correlation plot outliers), the algorithm deletes a percentage of them
unsupervised learning (we have no target, search for patterns) which data is 'normal' and which data abnormal
when a low number of checks in the tree to get to the value then abnormal else normal


Aviad: Train vs Validation data
Here are some techniques, some simple and some more advanced to compare the validation dataset with the training dataset. Are they dissimilar in some ways? 
---

EDA checks (fast + effective)

Compare summary stats
Look for: big gaps in mean/median/std between train vs valid.
Helps: catches obvious distribution shift that can bias your model.

Two-sample tests
Look for: small p-values from KS (numeric) or Chi-square (categorical).
Helps: tells you if the difference is statistically significant, not just visual.

Distribution plots
Look for: misaligned histograms/KDEs, outliers only in one split.
Helps: visual confirmation of drift, easier to explain to others.

Missingness and cardinality drift
Look for: higher null rate in one set, new categories appearing only in valid.
Helps: null handling and unseen categories are classic sources of test failure.

Correlation / covariance drift
Look for: feature relationships changing (e.g. X and Y correlated in train, uncorrelated in valid).
Helps: signals that your model may rely on unstable patterns.

---
Adversarial validation

Train a classifier to predict “is_train” vs “is_valid”.
Look for: high ROC-AUC (>0.7 means the sets are distinguishable).
Helps: if a model can separate them, your real model will also exploit those differences and fail to generalize.
Bonus: check feature importances/SHAP — they tell you which features drift most.

---
Drift indices & divergences (advanced)

Population Stability Index (PSI)
Look for: PSI > 0.2 on a feature = meaningful drift.
Helps: used in production (especially finance) to monitor input stability.

Jensen–Shannon / KL / Wasserstein distances
Look for: large divergence values.
Helps: quantifies how far distributions have moved, even if visually subtle.

Maximum Mean Discrepancy (MMD)
Look for: large test statistic rejecting null hypothesis of same distribution.
Helps: sensitive to high-dimensional drift beyond single features.

---
Structure / geometry diagnostics (advanced)

Dimensionality reduction (PCA/UMAP/t-SNE)
Look for: train and valid points clustering apart when colored by split.
Helps: shows multivariate drift visually, not just per-feature.

Density ratio estimation (logistic or k-NN)
Look for: sample weights far from 1.
Helps: tells you which regions of feature space are under- or over-represented in validation.
---
Target & leakage checks
Target distribution
Look for: different label balance or mean/variance of target.
Helps: target drift is expected in temporal splits, but you need to know about it.

Leakage probes
Look for: features highly predictive of split or target only in one set.
Helps: prevents training on information that won’t exist in real deployment.

Pipelines helps preventing data leakage: do this for train, test and validation datasets
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import IsolationForest
import shap
from IPython.display import display
from typing import Tuple  # for python Tuple type annotations
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.target_encoder import TargetEncoder

pio.renderers.default = "browser"



# RMSE function
def rmse(y: pd.DataFrame, y_pred: pd.DataFrame) -> float:  # check if it should be a dataframe or float
    return mean_squared_error(y, y_pred) ** 0.5


# return numeric columns
def keep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=['number'])


# drop list of column names from df
def drop_columns(df: pd.DataFrame, lst: list[str]) -> pd.DataFrame:
    return df.drop(columns=lst)


def category_cat_code(X: pd.DataFrame) -> pd.DataFrame:
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')

        if X[col].dtype == 'category':
            X[col] = X[col].cat.codes
    return X

def dummies(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df)


def prepare_data(df_train_data: pd.DataFrame) -> pd.DataFrame:
    #Data Cleaning / Handling Missing Values
    # Fix invalid values in 'YearMade':
    # - Replace 1000 with NaN
    # - Fill missing values with median per ModelID - because machines of the same model usually share a similar production year.
    # - Fill any remaining missing values with overall median
    df_train_data["YearMade"] = df_train_data["YearMade"].replace(1000, np.nan)
    df_train_data["YearMade"] = df_train_data.groupby("ModelID")["YearMade"].transform(lambda x: x.fillna(x.median()))
    df_train_data["YearMade"] = df_train_data["YearMade"].fillna(df_train_data["YearMade"].median())

    #Enclosure
    df_train_data["Enclosure"].isna().sum() # 0.08% missing values
    df_train_data["Enclosure"] = df_train_data["Enclosure"].fillna(df_train_data["Enclosure"].mode()[0])

    #ProductSize
    #df_train_data["ProductSize"].unique()
    #df_train_data["fiProductClassDesc"].unique()
    #df_train_data["fiProductClassDesc"].nunique()
    #print(df_train_data.groupby("fiProductClassDesc")["ProductSize"].count())
    #fiProductClassDesc has NO missing values and it's strongly related to ProductSize.
    # Note: solution drafted with ChatGPT

    productsize_mapping = df_train_data.groupby("fiProductClassDesc")["ProductSize"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )
    df_train_data["ProductSize"] = df_train_data["ProductSize"].fillna(df_train_data["fiProductClassDesc"].map(productsize_mapping))
    df_train_data["ProductSize"] = df_train_data["ProductSize"].fillna("Missing")

    #Hydraulics
    #Option 1: fill missing with a new category "Missing"
    df_train_data["Hydraulics"] = df_train_data["Hydraulics"].fillna("Missing")
    #Option 2: fill missing with the most frequent value (mode)
    df_train_data["Hydraulics"] = df_train_data["Hydraulics"].fillna(df_train_data["Hydraulics"].mode()[0])

    # Drop ID columns and features with >80% missing (maybe???)
    #ID columns
    #df_train_data = df_train_data.drop(columns=["SalesID", "MachineID", "ModelID", "datasource"])

    #80% missing
    df_train_data = df_train_data.drop(columns=[
        "fiModelSeries", "fiModelDescriptor",
        "Blade_Extension", "Blade_Width", "Enclosure_Type", "Engine_Horsepower",
        "Pushblock", "Scarifier", "Tip_Control",
        "Coupler_System", "Grouser_Tracks", "Hydraulics_Flow",
        "Track_Type", "Undercarriage_Pad_Width", "Stick_Length", "Thumb",
        "Pattern_Changer", "Grouser_Type",
        "Differential_Type", "Steering_Controls"
    ])

    #Feature Engineering
    #saledate -features from saledate (year, month, day, weekday, day of year)
    df_train_data["saledate"] = pd.to_datetime(df_train_data["saledate"])
    df_train_data["SaleYear"] = df_train_data["saledate"].dt.year
    df_train_data["SaleMonth"] = df_train_data["saledate"].dt.month
    df_train_data["SaleDay"] = df_train_data["saledate"].dt.day
    df_train_data["SaleDayOfWeek"] = df_train_data["saledate"].dt.dayofweek
    df_train_data["SaleDayOfYear"] = df_train_data["saledate"].dt.dayofyear
    df_train_data = df_train_data.drop("saledate", axis=1) #Remove original saledate column

    #machine_age at sale (how old the machine is when sold)
    df_train_data["MachineAge"] = df_train_data["SaleYear"] - df_train_data["YearMade"]

    return df_train_data


# split the data into train and test
def split_data(df: pd.DataFrame, y_name: str, cat_ind: int, t_size: float, rand_state: int) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=[y_name])
    y = df[y_name]

    # Feature Engineering: call category_cat_code(X) function to take care of object and category variables
    if cat_ind > 0:
        X = category_cat_code(X)
        print('print X.head() after object-category cat code')
        print(X.head())
    # split to train and test data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=rand_state)
    return X_train, X_test, y_train, y_test


# permutation importance function
def perm_importance_df(rf, xs, perm_n_repeat: int) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = xs
    # Compute Permutation Importance
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=perm_n_repeat, random_state=42, n_jobs=-1)
    # Display the results
    print('Feature Importance Results on Test Model Data:')
    importance_df = pd.DataFrame({
        "Feature": rf.feature_names_in_,
        "Permutation Importance": perm_importance.importances_mean,
        "Permutation Std Deviation": perm_importance.importances_std,
        'model importance': rf.feature_importances_
    }).sort_values(by="Permutation Importance", ascending=False)
    display(importance_df)
    return importance_df


def iso_forest(df_: pd.DataFrame,X_train_: pd.DataFrame, X_test_: pd.DataFrame, contam: float) -> pd.DataFrame:
    iso_forest_ = IsolationForest(n_estimators=100, contamination=contam ,random_state=42)
    iso_forest_.fit(X_train_)
    X_test_copy = X_test_.copy()

    df_['anomaly_score'] = iso_forest.decision_function(X_test_copy)  # Quantitative weirdness
    df_['anomaly'] = iso_forest.predict(X_test_copy)  # Binary anomaly label
    df_['anomaly_label'] = df_['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

    return df_

def pipe_lines():
    # ----------------------------
    # 1. Load dataset
    # ----------------------------
    diamonds = sns.load_dataset("diamonds")

    # Features & Target
    X = diamonds.drop(columns=["price"])
    y = diamonds["price"]

    # Train/Test split (no leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------------
    # 2. Define categorical & numeric features
    # ----------------------------
    cat_features = X.select_dtypes(include=["object","category"]).columns.tolist()
    num_features = X.select_dtypes(include=["int4","float64"]).columns.tolist()

    # ----------------------------
    # 3. Preprocessor (Target Encoding for cats, passthrough for nums)
    # ----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat",
             TargetEncoder(
                 # cols=cat_features
                 smoothing=20,
                 handle_missing="value",
                 handle_unknown="value"
             ),
            cat_features),
            ("num", "passthrough", num_features)
        ]
    )

    # ----------------------------
    # 4. Pipeline with RandomForest
    # ----------------------------
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # Fit
    pipe.fit(X_train, y_train)

    # ----------------------------
    # 5. Predictions & RMSE
    # ----------------------------
    y_pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse:,.2f}")

    #predictions vs actual: Checks bias & heteroscedasticity (if errors increase with price).
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual Diamond Prices")
    plt.show()

    #residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals vs Predicted")
    plt.show()

    #feature importance
    # Extract feature names after preprocessing
    feature_names = (
            pipe.named_steps["preprocess"]
            .transformers_[0][1].get_feature_names_out(cat_features).tolist()
            + num_features
    )

    importances = pipe.named_steps["model"].feature_importances_

    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_imp = feat_imp.sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(feat_imp["Feature"], feat_imp["Importance"])
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()

    #distribution of errors
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=50, color="purple", alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.show()


# def run_rf(df: pd.DataFrame, y_name: str, rfr_m_samp: float, rfr_msl:int ,t_size: float, rand_state: int)-> None:
#    model = RandomForestRegressor(max_samples=rfr_m_samp, min_samples_leaf=rfr_msl)
def run_rf(df: pd.DataFrame, spl_y_name: str, cat_ind: int, spl_t_size: float,
           spl_rand_state: int) -> None:  # Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # cat codes and object to category
    # split data
    X_train, X_test, y_train, y_test = split_data(df, spl_y_name, cat_ind, spl_t_size, spl_rand_state)
    # train model
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    # test model
    y_test_pred = rf.predict(X_test)
    # print rmse results
    print(f" RMSE Train: {rmse(y_train, y_train_pred)}")
    print(f" RMSE Test: {rmse(y_test, y_test_pred)}")
    # Feature Importance
    # perm_importance_df(rf, (X_train, X_test, y_train, y_test))
    perm_importance_df(rf, (X_train, X_test, y_train, y_test), 5)
    # return X_train, X_test, y_train, y_test


# set category columns as codes
def set_cat_as_code(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes('category').columns:
        df[col] = df[col].cat.codes
    return df


def get_df_properties(df: pd.DataFrame) -> None:
    print('---------- head(5) ----------')
    print(df.head(5))
    print('---------- info() ----------')
    print(df.info())
    print('---------- describe() ----------')
    print(df.describe())
    print('---------- shape ----------')
    print(df.shape)
    print('---------- nunique() ----------')
    print(df.nunique())
    print('---------- df.isna().sum() ----------')
    print(df.isna().sum())


# load data:
df_train_data = pd.read_csv('Train.csv',
                            low_memory=False)  # low_memory=false because got error because some columns have more than one type (str, num) as values
# get data properties:
get_df_properties(df_train_data)

# df_valid_data=pd.read_csv('Valid.csv', low_memory=False)
# get_df_properties(df_valid_data)

'''Feature engineering - Prepare Data'''
# df = set_cat_as_code(df)
# df=keep_numeric(df)
# df = drop_columns(df, ['depth', 'table'])

# Ideas for engineering
# look how many unique values in each column to know whch is more granular, trial and error...
#  להקטין גרנולריות של שדה שמשמש את העץ יותר מידי בהחחלטות
# df.table.nunique(), df.depth.nunique(),
# pd.qcut(df.table, q=100, duplicates='drop')

# pd_train_data=category_cat_code(pd_train_csv) function
# pd_valid_data=category_cat_code(pd_valid_csv) function

# rf = RandomForestRegressor(n_jobs=-1, max_samples=0.5)
# run_rf(df_train_data, 'price', 1 ,0.3, 42) #cat_ind =1 then call function to handle objects ad categories and codes

# Natalie's Prepare Function
df_train_data=prepare_data(df_train_data)

# Iso Forest - prediction of anomaly score (outliers i.e.)
# Plot carat vs. price with hue as anomaly score (quantitative weirdness)
X_train, X_test, y_train, y_test = split_data(df_train_data, 'SalePrice', 1, 0.3,42)
df = iso_forest(df_train_data, X_train , X_test, 0.01)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df['carat'], df['price'], c=df['anomaly_score'], cmap='coolwarm',
    # edgecolor='k',
    alpha=0.8
)

plt.colorbar(scatter, label='Anomaly Score (Weirdness)')
plt.title('Carat vs. Price with Anomaly Score as Hue')
plt.suptitle('Anomaly Score: lower is more anomalous')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.grid(True)
plt.show()


sns.displot(data=df, x='anomaly_score', kde=True)
df.sort_values("anomaly_score").sample(20)

df.sort_values("anomaly_score").head()
import seaborn as sns

sns.displot(data=df[df.anomaly == -1].carat)

print((df.carat>3).mean()) # percentage of diamonds with carat > 3

"""run random forest regression"""
rf = RandomForestRegressor(n_jobs=-1, n_estimators=100, max_samples=0.5, min_samples_leaf=10)
run_rf(df_train_data, 'SalePrice', 1, 0.3,
       42)  # cat_ind =1 then call function to handle objects ad categories and codes

"""run random forest regression for different number of trees"""
# for n in [1,5,10, 100, 200, 500]:
#     print(f"n= {n} :")
#     rf = RandomForestRegressor(n_jobs=-1, n_estimators=n,max_samples=0.5, min_samples_leaf=10)
#     run_rf(df, 'price', 0 ,0.3, 42) #cat_ind =1 then call function to handle objects ad categories and codes

'''validate model'''

#NOTE: Data Cleaning & Feature Engineering – Natalie’s part

