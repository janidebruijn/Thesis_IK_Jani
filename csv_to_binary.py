# Import required package
import pandas as pd


def main():
    df = pd.read_csv('gold_standard_test.csv')
    columns = ['agency', 'event_sequencing', 'world_making']

    # Binary transformation
    df[columns] = df[columns].applymap(lambda x: 1 if x > 2.5 else 0)

    # Save the result to a file
    outfile = 'binary_gold_standard_test.csv'
    df.to_csv(outfile, index=False)

    print(f'Binary transformation complete. Saved as {outfile}')


if __name__ == '__main__':
    main()
