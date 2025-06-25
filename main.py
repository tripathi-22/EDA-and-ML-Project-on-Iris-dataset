import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df=pd.read_csv("Iris.csv")

df=df.drop(columns=["Id"])

df.rename(columns={
    'SepalLengthCm': 'Sepal_Length',
    'SepalWidthCm': 'Sepal_Width',
    'PetalLengthCm': 'Petal_Length',
    'PetalWidthCm': 'Petal_Width'
}, inplace=True)

# Display the first few rows of the dataset
df.head()

# Step 1: Basic EDA

# Shape and data types
shape=df.shape

summary=df.describe()

print(df.info())
print("\nMissing values:\n", df.isnull().sum())

le=LabelEncoder()
df["SpeciesEncoded"]=le.fit_transform(df["Species"])

#Remove outliers using IQR
def remove_outliers(data,column):
    Q1=data[column].quantile(0.25)
    Q3=data[column].quantile(0.75)
    IQR=Q3-Q1
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

for col in ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']:
    df=remove_outliers(df,col)

sns.set(style="whitegrid")
features=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

#Boxplot
plt.figure(figsize=(12,8))
for i,col in enumerate(features):
    plt.subplot(2,2,i+1)
    sns.boxplot(x=df[col], color="steelblue")
    plt.title(f'Boxplot - {col}')
plt.tight_layout()
plt.show()

#Voilin plot
plt.figure(figsize=(10,6))
sns.violinplot(data=df[features])
plt.title("Voilin Plot of features")
plt.show()

#Displot
for col in features:
    sns.displot(df[col],kde=True)
    plt.title(f'Distribution - {col}')
    plt.show()

#Scatter plot
sns.scatterplot(data=df, x="Sepal_Length", y="Sepal_Width", hue="Species")
plt.title("Sepal Dimensions Scatter")
plt.show()

x=df[features]
y=df['SpeciesEncoded']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(x)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=42)

#KNN Model
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print("\nAccuracy:",accuracy_score(y_test,y_pred))
print("\nClassifications Report:\n", classification_report(y_test,y_pred, target_names=le.classes_))

cm=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.title("KNN Confusion Matrix")
plt.show()