import pandas as pd


def load_data():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def clean_data(df):
    # 1. Feature Engineering: Create FamilySize before dropping SibSp/Parch
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 2. Feature Engineering: Extract Titles from Name before dropping it
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Group rare titles into a single "Rare" category or map them cleanly
    title_mapping = {
        "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,
        "Dr": 5, "Rev": 5, "Col": 5, "Major": 5, "Mlle": 2, "Countess": 3,
        "Ms": 2, "Lady": 3, "Jonkheer": 5, "Don": 5, "Dona": 3, "Mme": 3,
        "Capt": 5, "Sir": 5
    }
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)  # Fill any unexpected titles with 0

    # 3. Fill missing Age with the median age
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)

    # 4. Fill missing Fare (happens in test.csv sometimes) with median fare
    if 'Fare' in df.columns:
        median_fare = df['Fare'].median()
        df['Fare'] = df['Fare'].fillna(median_fare)

    # 5. Fill missing Embarked with the most common value (mode)
    common_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(common_embarked)

    # 6. Drop columns we no longer need raw forms of
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # 7. Encode Categorical Features
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    return df