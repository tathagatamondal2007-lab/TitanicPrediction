import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_and_predict(train_df, test_df):
    # 1. Separate features (X) and target (y) for training
    X = train_df.drop(columns=['Survived'])
    y = train_df['Survived']

    # 2. Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Initialize and train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate on validation set
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    print(f"Random Forest Validation Accuracy: {accuracy * 100:.2f}%")

    # 5. Generate Predictions for Kaggle Test Set
    # Note: test.csv doesn't have PassengerId in cleaned df, but we need it for Kaggle submission!
    # Let's reload raw test_df just to grab the PassengerId column.
    raw_test_df = pd.read_csv('data/test.csv')
    test_passenger_ids = raw_test_df['PassengerId']

    # Predict on the fully cleaned test dataset
    test_predictions = model.predict(test_df)

    # 6. Create the submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': test_predictions
    })

    # Save to a CSV file
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file 'submission.csv' generated successfully!")

    return model