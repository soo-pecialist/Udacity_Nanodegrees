import sys
import re
import numpy as np
import pandas as pd
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

def load_data(database_filepath='data/DisasterResponse.db', tablename='EightFigureTable'):
    """
    This function loads data from database

    > Parameters:
    database_filepath: relative or absolute directory of file

    > Returns:
    X: messages
    Y: multiouput labels
    categories: names of categories
    """

    # load data from database
    # database_filepath = re.search('(?<=/)([a-zA-Z0-9_-]+.db)', database_filepath).group(0)
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql(tablename, con=engine)

    ## Define feature and target variables X and Y
    X = df['message'].values
    ## define classes names
    categories = df.columns[4:].values
    Y = df[categories].values

    return X, Y, categories

def tokenize(text):
    """
    Normalize, tokenize, lemmatize, clean texts
    
    > Parameters:
    text: raw text
    
    > Returns:
    clean_tokens: tokens that went through aformentioned procedures
    """

    ## normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    ## tokenize texts
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        ## lemmatize, lowercase, strip spaces
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def important_word_count(X, stopwords=stopwords):
    """
    This function takes message feature and return important vocabularies 
    with importance rank and count

    > Parameters:
    X: message array

    > Returns:
    vocab_df: dataframe of words, rank, and count
    """
    ## tfidf matrix to achieve important words
    tfidf = TfidfVectorizer(min_df=0.01, max_df=0.90, tokenizer=tokenize, stop_words=stopwords)
    vect = CountVectorizer(tokenizer=tokenize, stop_words=stopwords)
    tfidf.fit(X)
    vect.fit(X)

    tfidf_vocab = tfidf.vocabulary_
    vect_vocab = vect.vocabulary_

    ## make tfidf dataframe - word & rank
    tfidf_df = pd.DataFrame.from_dict(tfidf_vocab, orient='index').reset_index()
    tfidf_df.columns = ['word', 'rank']

    ## make count vectorizer dataframe - word & count
    vect_df = pd.DataFrame.from_dict(vect_vocab, orient='index').reset_index()
    vect_df.columns = ['word', 'count']

    ## merge two dataframe in the order of count
    vocab_df = pd.merge(tfidf_df, vect_df, how='left', on=['word']).sort_values('count', ascending=False).reset_index(drop=True)
    vocab_df = vocab_df[['word', 'count']]

    ## randomly choose 30 important words
    vocab_df = vocab_df.sample(30, random_state=76321).sort_values('count', ascending=False).reset_index(drop=True)

    return vocab_df


def build_model(stopwords=stopwords):
    ## adaboost pipeline
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, stop_words=stopwords)),
            ('tfidf', TfidfTransformer()),
            ('rfc', MultiOutputClassifier(RandomForestClassifier(n_estimators=50, 
                                                                 n_jobs=25, 
                                                                 random_state=4999)))
        ])
    
    return pipeline


def multioutput_classification_report(y_true, y_pred, category_names):
    """
    This is classification report for multioutput classifiers
    
    > Parameters:
    y_true: true labels; numpy.ndarray
    y_pred: predicted labels; numpy.ndarray
    
    > Returns: None
    """
    
    supports = y_true.sum(axis=0)
    print("{:>24s}{:>12s}{:>12s}{:>12s}{:>12s}".format('', 'Precision', 'Recall', 'F1_score', 'support'))
    for i in range(0, y_true.shape[1]):
        _ = precision_recall_fscore_support(y_true[:, i], y_pred[:, i], average='micro')
        print("{:>24s}{:>12.2f}{:>12.2f}{:>12.2f}{:>12.2f}".format(category_names[i], _[0], _[1], _[2], supports[i]))


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model with multioutput classification report

    > Paramters:
    model: classifier
    X_test: Features ; numpy.ndarray
    Y_test: true labels; numpy.ndarray
    category_names: names of categories

    > Returns: None
    """
    Y_pred = model.predict(X_test)
    multioutput_classification_report(Y_test, Y_pred, category_names)

def save_model(model, model_filepath='models/classifier.pkl'):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

## get stopwords ready
stopwords = tokenize(" ".join(stopwords.words('english')))
stopwords.extend([str(i) for i in range(0, 1000)])
stopwords.extend(['000'])

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1004)
        
        vocab_df = important_word_count(X, stopwords)

        print('Building model...')
        model = build_model(stopwords)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        save_model(vocab_df, 'data/vocab_df.pkl')

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()