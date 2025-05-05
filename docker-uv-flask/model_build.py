'''                     Import libraries.                       '''
import polars as pl
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay#, accuracy_score, precision_score, recall_score, f1_score


'''                     User defined variables.                       '''
# Target field.
col_target = 'target'
dict_target = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

# Random seed.
random_seed = 42

# File name.
filename_model = 'model_rfc.pkl'


'''                     Load and analyse the data.                       '''
# Load the dataset.
# https://en.wikipedia.org/wiki/Iris_flower_data_set
data_iris = load_iris()
print(data_iris['DESCR'])

# Create a Polars DataFrame from the iris dataset.
'''
X, y = load_iris(return_X_y=True, as_frame=True)

Note: In Polars, we use the `DataFrame` constructor to create a new dataframe.
The `schema` parameter is used to specify the column names.
The data is in the `data` attribute and the feature names are in `feature_names`.
The `target` attribute contains the target variable.

Note: In Polars, we use the `with_columns` method to add a new column.
'''
df_iris = pl.DataFrame(data_iris.data, schema=data_iris.feature_names)
df_iris = df_iris.with_columns(pl.Series('target', data_iris.target))
print(df_iris.head())

'''
df_iris.describe()

df_iris['target'].value_counts()
df_iris['target'].value_counts(normalize=True)

df_iris.corr()
'''

### Scatter plot the data.
'''
https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
https://matplotlib.org/stable/api/markers_api.html
https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html
'''
## Sepal length vs Sepal width
# Plot each class separately
for target_class in df_iris['target'].unique():
    plt.scatter(
        x=df_iris.filter(pl.col('target') == target_class)['sepal length (cm)'],
        y=df_iris.filter(pl.col('target') == target_class)['sepal width (cm)'],
        s=100,
        # c=df_iris.filter(pl.col('target') == target_class)['target'],  # Provide a sequence for colormapping
        # cmap='viridis',
        label=dict_target[target_class],  # Add a label for each class
        edgecolor='k',
        alpha=0.7,
        marker='X'
    )

# Add labels and title
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset Scatter Plot')
plt.legend()  # Display the legend
plt.show()

## Petal length vs Petal width
# Plot each class separately
for target_class in df_iris['target'].unique():
    plt.scatter(
        x=df_iris.filter(pl.col('target') == target_class)['petal length (cm)'],
        y=df_iris.filter(pl.col('target') == target_class)['petal width (cm)'],
        s=100,
        # c=df_iris.filter(pl.col('target') == target_class)['target'],  # Provide a sequence for colormapping
        # cmap='viridis',
        label=dict_target[target_class],  # Add a label for each class
        edgecolor='k',
        alpha=0.7,
        marker='X'
    )

# Add labels and title
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Dataset Scatter Plot')
plt.legend()  # Display the legend
plt.show()

# Check for missing values.
print(df_iris.select(pl.all().is_null().sum()))


'''                     Model build.                       '''
# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    df_iris.drop(col_target),
    df_iris[col_target],
    test_size=0.2,
    random_state=random_seed,
)

'''
type(y_test)
y_test.n_unique()
y_test.unique()
y_test.unique_counts()
'''

# Random forest classifier.
model_rfc = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=random_seed,
    n_jobs=-1,)

# Fit the model.
model_rfc.fit(X=X_train, y=y_train)

# Predict the test set results.
y_pred = model_rfc.predict(X=X_test)

# Print the classification report.
print(classification_report(
    y_true=y_test,
    y_pred=y_pred,
    target_names=list(dict_target.values()),
    # output_dict=True
))

# Plot the confusion matrix.
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(dict_target.values()))
cm_disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

'''
print(f"Accuracy score : {accuracy_score(y_true=y_test, y_pred=y_pred)}")
print(f"Precision score : {precision_score(y_true=y_test, y_pred=y_pred, average='weighted')}")
print(f"Recall score : {recall_score(y_true=y_test, y_pred=y_pred, average='weighted')}")
print(f"F1 score : {f1_score(y_true=y_test, y_pred=y_pred, average='weighted')}")
'''

# Plot the feature importances.
df_fea_imp = pl.DataFrame(
    data={
        'Feature': model_rfc.feature_names_in_,
        'Importance': model_rfc.feature_importances_,
        }
    )
df_fea_imp = df_fea_imp.sort(by='Importance', descending=True)
print(df_fea_imp)

plt.barh(
    y=df_fea_imp['Feature'],
    width=df_fea_imp['Importance'],
    color='blue',
    alpha=0.7,
)
plt.xlabel('Importance')
plt.title('Random Forest Classifier Feature Importances - Iris Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='x')
plt.show()

# Save the model object.
joblib.dump(value=model_rfc, filename=filename_model)

# Test the model.
list_input = [4, 3, 2, 1]
df_input = pl.DataFrame(
    data=[list_input], 
    schema=list(model_rfc.feature_names_in_),
    orient='row'
    )
print(df_input)
model_rfc.predict(df_input)
model_rfc.predict_proba(df_input)

# model_rfc.predict_proba(df_input).max(axis=1)
# model_rfc.predict_proba(df_input).argmax(axis=1)