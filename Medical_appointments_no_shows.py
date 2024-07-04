import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
from sklearn.model_selection import KFold, cross_val_predict, RandomizedSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline



# Importing the csv file and creating a dataframe using pandas
data = pd.read_csv("KaggleV2-May-2016.csv")

# Printing the head of the dataframe 
display(data.head())

# Printing the column names of the dataframe
display(data.columns)

# Renaming the names of the columns 
data = data.rename(columns={"Handcap":"Handicap","Hipertension":"Hypertension", "No-show":"No_show", "SMS_received":"SMS"})

# Printing the head of the dataframe 
print("\nThe new dataframe with the renamed columns is: ")
display(data.head())

# General statistics
print(f"The dimensions of the dataframe are:\n {data.shape}")
print(f"\nThe data types of the dataframe are:\n {data.dtypes}")
display(data.describe())
display(data.info())

# Checking for missing values
display(data.isnull().sum()) # No missing values detected

# Checking for duplicate values if any according to unique valued columns PatientId and AppointmentID  
print(f"\nThe number of duplicated values is:\n{data.duplicated(subset=['PatientId', 'AppointmentID']).sum()}")  # No duplicated values detected

# Converting columns ScheduledDay and AppointmentDay in datetime format
data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'])
data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])
display(data.head())


# Dropping columns PatientId and AppointmentID which are not any more useful for our analysis
data.drop(columns=["PatientId", "AppointmentID"], inplace=True)
display(data.columns)

# Checking the unique values of each column
for column in data.columns:
    print(f"\nThe unique values of column {column} are:\n\n {data[column].unique()}")

# As we can see there are Age values = -1 and Age values = 0
print(f"The number of rows with Age = -1 is: {len((data[data['Age'] == -1]))}") 
print(f"The number of rows with Age different from  -1 is: {len((data[data['Age'] != -1]))}")

data.drop(data[data["Age"] == -1].index, inplace=True)

print(len((data[data["Age"] == 0])))  # 3539 rows with Age = 0, we assume they are babies

# We are going to create a new column named "Show" containing the reverse values of the excisting "No_show" column for redability reasons.
data["Show"] = data["No_show"].apply(lambda x: "Yes" if x=="No" else "No")
display(data.head())

# Dropping the "No_show", "Neighbourhood" columns
data.drop(columns=['No_show', 'Neighbourhood'], inplace=True)
display(data.head())


# Calculating the waiting time in days
data['WaitingDays'] = (data['AppointmentDay'].dt.normalize() - data['ScheduledDay'].dt.normalize()).dt.days

# Setting any negative waiting days to zero
data['WaitingDays'] = data['WaitingDays'].apply(lambda x: max(x, 0))

# Creating a plot comparing the waiting days with the Show status
fig = px.histogram(
    data,
    x='WaitingDays',
    color='Show',
    barmode='group',
    title='Comparison of Waiting Days with Show Status',
    labels={'WaitingDays': 'Waiting Days', 'Show': 'Show Status'},
    text_auto=True,
    nbins=10,
    log_y=True
)

# Updating layout for better visualization
fig.update_layout(
    xaxis_title='Waiting Days',
    yaxis_title='Count',
    bargap=0.2,
)


fig.show()


# Defining the list of columns of interest (excluding 'Handicap')
columns_of_interest = ['Gender', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'SMS']

# Creating subplots: 2 rows, 3 columns per row
fig = make_subplots(rows=2, cols=3, subplot_titles=columns_of_interest)

# Iterating through the specified columns to create histograms and add them to the subplots
for i, column in enumerate(columns_of_interest):
    if column in data.columns:  # Check if the column exists in the DataFrame
        hist = px.histogram(
            data,
            x=column,
            color="Show",
            text_auto=True,
            category_orders={column: sorted(data[column].unique())},
            labels={"Show": "Show Status"}
        )
        hist.update_traces(marker_line_width=1.5, marker_line_color='rgb(0, 0, 0)')
        row = (i // 3) + 1
        col = (i % 3) + 1
        for trace in hist['data']:
            fig.add_trace(trace, row=row, col=col)

# Updating layout for the entire figure
fig.update_layout(
    height=800,
    width=1000,
    title_text="Distributions of Columns of Interest",
    showlegend=False,  
    barmode='group',  
    bargap=0.2,  
)

# Displaying the figure
fig.show()


# Defining the column of interest
column_of_interest = 'Handicap'


# Creating the histogram for the 'Handicap' column
fig = px.histogram(
    data,
    x=column_of_interest,
    text_auto=True,
    color="Show",
    category_orders={column_of_interest: sorted(data[column_of_interest].unique())},
    log_y=True
)

# Adding a gap between bars
fig.update_traces(marker_line_width=1.5, marker_line_color='rgb(0, 0, 0)', bingroup=1)

# Updating layout
fig.update_layout(
    title=f'Distribution of {column_of_interest}',
    xaxis_title=column_of_interest,
    yaxis_title='Count',
    bargap=0.3,
    barmode='group',  
    showlegend=True,
)

# Displaying the histogram
fig.show()


# Using map for encoding Gender and Show columns
data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})
data['Show'] = data['Show'].map({'No': 0, 'Yes': 1})
display(data.head())


# One-hot encoding for 'Handicap' column
handicap_encoded = pd.get_dummies(data['Handicap'], prefix='Handicap')

# Concatenating the one-hot encoded columns with the original data
data = pd.concat([data, handicap_encoded], axis=1)

# Dropping the original 'Handicap' column
data.drop('Handicap', axis=1, inplace=True)

# Checking the updated DataFrame
display(data.head())


# Calculating correlation matrix
corr_matrix = data.corr()
# Creating the heatmap using Plotly Express
fig = px.imshow(
    corr_matrix, 
    text_auto=True,  
    color_continuous_scale='RdYlBu_r',
    title='Correlation Matrix Heatmap',
    width=1000,  # Adjust the width of the plot if needed
    height=1000
)

fig.show()




# Extracting useful features from datetime columns
data['ScheduledDay_year'] = data['ScheduledDay'].dt.year
data['ScheduledDay_month'] = data['ScheduledDay'].dt.month
data['ScheduledDay_day'] = data['ScheduledDay'].dt.day
data['ScheduledDay_dayofweek'] = data['ScheduledDay'].dt.dayofweek
data['AppointmentDay_year'] = data['AppointmentDay'].dt.year
data['AppointmentDay_month'] = data['AppointmentDay'].dt.month
data['AppointmentDay_day'] = data['AppointmentDay'].dt.day
data['AppointmentDay_dayofweek'] = data['AppointmentDay'].dt.dayofweek
display(data.head())

# Dropping the original datetime columns
data.drop(['ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)


""" Regression Model """
data_without_Show = data.drop("Show", axis=1)# Excluding the target variable
data_Show = data['Show'] #Defining the target variable

X, y = data_without_Show, data_Show

# Initialize Logistic Regression Model
lreg = LogisticRegression(random_state=42, max_iter=1000)

# Using K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation predictions
y_pred = cross_val_predict(lreg, X, y, cv=kf)



print(f"Confusion Matrix:\n {confusion_matrix(y, y_pred)}")
print(f"\nClassification Report:\n {classification_report(y, y_pred)}")
print(f"\nAccuracy Score: {accuracy_score(y, y_pred)}")



"""In order to handle theese problems we are going to implement Random Forest with SMOTE using K-Fold Cross Validation."""


X = data.drop('Show', axis=1)
y = data['Show']

# Initializing SMOTE and Random Forest
smote = SMOTE(random_state=42)
rf = RandomForestClassifier(random_state=42, n_estimators=100)

# Creating a pipeline with SMOTE and Random Forest
pipeline = Pipeline([('smote', smote), ('rf', rf)])

# Initialize K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation predictions
y_pred = cross_val_predict(pipeline, X, y, cv=kf)

# Evaluation Metrics
conf_matrix = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred)
acc_score = accuracy_score(y, y_pred)

print(f"Confusion Matrix:\n {conf_matrix}")
print(f"\nClassification Report:\n {class_report}")
print(f"\nAccuracy Score: {acc_score}")


"""We are going to set up the environment for performing hyperparameter tuning using RandomizedSearchCV with a RandomForestClassifier within an imblearn.pipeline.Pipeline that includes SMOTE for handling class imbalance. It initializes the necessary imports, defines the parameter distribution for the random search, sets up the pipeline, and fits the RandomizedSearchCV to find the best model. Finally, it prints out the best parameters found, fits the best model on the entire dataset, and evaluates its performance using confusion matrix, classification report, and accuracy score metrics. Adjust as needed based on your specific requirements and further analysis."""


X, y = data_without_Show, data_Show

# Define the parameter distribution for hyperparameter tuning
param_dist = {
    'randomforestclassifier__n_estimators': randint(100, 300),
    'randomforestclassifier__max_depth': [None, 10, 20, 30],
    'randomforestclassifier__min_samples_split': randint(2, 10),
    'randomforestclassifier__min_samples_leaf': randint(1, 4),
    'randomforestclassifier__max_features': ['sqrt', 'log2', None]
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Set up the pipeline with SMOTE and Random Forest
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('randomforestclassifier', rf)
])

# Set up the RandomizedSearchCV with reduced iterations and 3-fold CV
random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist, n_iter=20, cv=3, 
    scoring='accuracy', n_jobs=-1, random_state=42, verbose=2
)

# Fit the RandomizedSearchCV to the data
random_search.fit(X, y)

# Get the best parameters and the best model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

print(f"Best Parameters: {best_params}")

# Predict using the best model
y_pred = best_model.predict(X)

# Evaluate the best model
print(f"Confusion Matrix:\n {confusion_matrix(y, y_pred)}")
print(f"\nClassification Report:\n {classification_report(y, y_pred)}")
print(f"\nAccuracy Score: {accuracy_score(y, y_pred)}")
