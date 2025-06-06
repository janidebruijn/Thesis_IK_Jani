# Import required packages
import pandas as pd


def main():
    # Define input and output file(s)
    file_1 = 'threads1000_format_preprocessed.csv'
    file_2 = 'qwen_bi_zeroshot_output_lg.csv'
    output_file = 'merged_output.csv'

    # Load the files into dataframes
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)

    # Only merge the matches on 'name'
    merged_df = pd.merge(df1[['name', 'deltas']], df2, on='name', how='inner')

    # Save to the output file
    merged_df.to_csv(output_file, index=False)

    print(f'Merged {len(merged_df)} rows and saved to {output_file}.')


if __name__ == '__main__':
    main()
