# SpaceX Landing Prediction Capstone Project

# --- 1. Import Required Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 2. Load Dataset ---
df = pd.read_csv("dataset_part_1.csv")

# --- 3. Data Wrangling ---
df.dropna(subset=['PayloadMass', 'LandingPad'], inplace=True)
df['Class'] = df['Outcome'].apply(lambda x: 1 if str(x).startswith('True') else 0)

# --- 4. EDA: Success by Launch Site ---
site_success = df.groupby('LaunchSite')['Class'].mean().sort_values(ascending=False)
site_success.plot(kind='bar', title='Success Rate by Launch Site')
plt.ylabel('Success Rate')
plt.tight_layout()
plt.show()

# --- 5. Folium Map ---
site_map = folium.Map(location=[28.5, -80.5], zoom_start=4)
mc = MarkerCluster()
for idx, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['LaunchSite'],
        icon=folium.Icon(color='green' if row['Class'] else 'red')
    ).add_to(mc)
mc.add_to(site_map)
site_map.save("spacex_launch_map.html")

# --- 6. Feature Engineering ---
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
y = df['Class']
X = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])

# --- 7. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 8. Model Training & Evaluation ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(confusion_matrix(y_test, y_pred))

# --- 9. Hyperparameter Tuning (Decision Tree) ---
dt = DecisionTreeClassifier()
params = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
gscv = GridSearchCV(dt, params, cv=5)
gscv.fit(X_train_scaled, y_train)
print("\nBest Decision Tree Parameters:", gscv.best_params_)
y_pred = gscv.predict(X_test_scaled)
print("\nTuned Decision Tree Accuracy:", accuracy_score(y_test, y_pred) * 100)

# --- 10. Dashboard Placeholder (Dash code to be added in separate file) ---
print("\nNOTE: Please run the Dash dashboard using app.py separately.")
