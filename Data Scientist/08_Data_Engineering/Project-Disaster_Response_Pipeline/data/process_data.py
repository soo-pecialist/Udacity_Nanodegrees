import sys
import pandas as pd
from sqlalchemy import create_engine
import re

def load_data(messages_filepath='data/disaster_messages.csv', 
              categories_filepath='data/disaster_categories.csv'):
    """
    This function loads raw messages and categories data

    > Parameters:
    messages_filepath: directory of message file
    categories_filepath:; directory of category file

    > Returns:
    df: merged dataframe
    """
    
    ### 1. load messages & categories dataset
    messages = pd.read_csv(messages_filepath, low_memory=False) 
    categories = pd.read_csv(categories_filepath, low_memory=False)

    ### 2. Merge datasets.
    # - Merge the messages and categories datasets using the common id
    # - Assign this combined dataset to `df`, which will be cleaned in the following steps

    # merge datasets
    df = pd.merge(messages, categories, how='inner')

    return df

def clean_data(df):
    """
    This function preprocess the data so it can be ready for machine learning algorithms

    > Parameters: 
    df: loaded dataframe

    >Returns: preprocessed dataframe
    """

    ### 1. Split `categories` into separate category columns.
    # - Split the values in the `categories` column on the `;` character so that each value becomes a separate column.
    # - Use the first row of categories dataframe to create column names for the categories data.
    # - Rename columns of `categories` with new column names.

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0].str.extract('([a-zA-z]+)(?=-)', expand=False)
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.values
    # rename the columns of `categories`
    categories.columns = category_colnames

    ### 2. Convert category values to just numbers 0 or 1.
    for column in categories.columns:
        # set each value to be the last character of the string & convert column from string to numeric
        categories[column] = categories[column].str.extract('(?<=-)([0-9])', expand=False).astype(int)

    ### 3. Replace `categories` column in `df` with new category columns.
    # - Drop the categories column from the df dataframe since it is no longer needed.
    # - Concatenate df and categories data frames.

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    ### 6. Remove duplicates.
    # - Check how many duplicates are in this dataset.
    # - Drop the duplicates.
    # - Confirm duplicates were removed.

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename='data/DisasterResponse.db', 
                  tablename='EightFigureTable'):
    """
    Save the clean dataset into an sqlite database.
    
    > Parameters:
    df: dataframe
    database_filename: database filename

    > Returns: None
    """

    # database_filename = re.search('(?<=/)([a-zA-Z0-9_-]+.db)', database_filename).group(0)
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(tablename, engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()