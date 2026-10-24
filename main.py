from preprocessor import load_data, clean_data
from model import train_and_predict

if __name__ == '__main__':
    print("Starting Titanic Prediction Pipeline...\n")

    # 1. Load raw data
    train_data, test_data = load_data()

    # 2. Clean both datasets using the exact same preprocessor rules
    print("\n--- Cleaning Data ---")
    train_cleaned = clean_data(train_data)
    test_cleaned = clean_data(test_data)

    # 3. Train Model & Generate Kaggle Submission File
    print("\n--- Training Model & Generating Submission ---")
    model = train_and_predict(train_cleaned, test_cleaned)

    print("\nPipeline execution completed successfully!")