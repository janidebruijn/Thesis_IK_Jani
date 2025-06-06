# Import required packages
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


def main():
    # Load both CSV files
    df1 = pd.read_csv("binary_gold_standard_test.csv")
    df2 = pd.read_csv("qwen_bi_zeroshot_output_uhhh.csv")

    # Align both dataframes by 'name'
    df1.set_index('name', inplace=True)
    df2.set_index('name', inplace=True)

    # Only take the rows of names that are in both dataframes
    common_names = df1.index.intersection(df2.index)
    df1 = df1.loc[common_names]
    df2 = df2.loc[common_names]

    # Define the columns to compare
    columns_to_compare = [
        'agency', 'event_sequencing', 'world_making', 'story'
    ]

    # For each column, compare predictions and print metrics
    for col in columns_to_compare:
        print(f"\n=== Metrics for {col} ===")
        y_true = df1[col]
        y_pred = df2[col]
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=3))


if __name__ == '__main__':
    main()
