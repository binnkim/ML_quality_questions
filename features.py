import pandas as pd
from scipy.stats import multinomial
from data_sources import *

def simple_mean_confidence_model():
    train_data = get_train_data()
    answer_data = get_answer_data()

    # Get mean confidence for each questions
    merged_data = answer_data.merge(train_data, on='AnswerId')
    questions_with_mean = merged_data.groupby('QuestionId')[['Confidence']].mean()

    # Get mean confidence for all questions
    mean_all_questions = questions_with_mean['Confidence'].mean()

    # Fill in na with mean confidence for all questions
    # There are 948 questions (unique)
    df = pd.DataFrame({'QuestionId':range(948), 'Confidence':mean_all_questions})
    df['Confidence'] = questions_with_mean['Confidence']
    df.fillna(mean_all_questions, inplace=True)

    # Rank
    # Assume the higher confidence is the better
    # Ranking sorted should be ascending
    df['ranking'] = df['Confidence'].rank(method='min', ascending=False).astype('int16') 
    df.sort_values(by='ranking', ascending=True, inplace=True) 
    df.drop(columns=['Confidence'], inplace=True)
    
    return df


def simple_correct_rate_model():
    train_data = get_train_data()
    answer_data = get_answer_data()

    # Get mean correct rate for each questions
    merged_data = answer_data.merge(train_data, on='AnswerId')
    questions_with_mean = merged_data.groupby('QuestionId')[['IsCorrect']].mean().rename(columns={'IsCorrect': 'CorrectRate'})

    df = questions_with_mean
    df['ranking'] = df['CorrectRate'].rank(method='min', ascending=False).astype('int16')
    df.sort_values(by='ranking', ascending=True, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'QuestionId'}, inplace=True)
    df.drop(columns=['CorrectRate'])
    
    return df


