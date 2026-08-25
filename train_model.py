import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1. Load dataset
data = pd.read_csv("Student_data.csv")

print("\nDataset loaded successfully!")
print("Total Students:", len(data))


# 2. Select input features
X = data[
    [
        "Attendance",
        "StudyHours",
        "PreviousMarks",
        "AssignmentScore",
        "InternalMarks"
    ]
]


# 3. Select target
y = data["Performance"]


# 4. Convert text labels into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# 6. Create different ML models
models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# 7. Train and compare models
results = {}

print("\nModel Performance:")
print("----------------------------")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print(
        name,
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )


# 8. Select best model
best_model_name = max(
    results,
    key=results.get
)

best_model = models[best_model_name]

print("\n----------------------------")
print("Best Model:", best_model_name)
print(
    "Best Accuracy:",
    round(results[best_model_name] * 100, 2),
    "%"
)


# 9. Save best model
with open(
    "student_performance_model.pkl",
    "wb"
) as file:

    pickle.dump(
        (best_model, encoder),
        file
    )


print("\nBest model saved successfully!")