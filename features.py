import pandas as pd
from scipy.stats import multinomial
from data_sources import *


def __get_rank(df, df_key, rank_ascending=False):
    df['ranking'] = df[df_key].rank(method='min', ascending=rank_ascending) \
                              .astype('int16') 
    df.sort_values(by='ranking', ascending=True, inplace=True) 
    return df


def simple_mean_confidence():
    train_data = get_train_data()
    answer_data = get_answer_data()

    # Get mean confidence for each questions
    merged_data = answer_data.merge(train_data, on='AnswerId')
    questions_with_mean = merged_data.groupby('QuestionId')[['Confidence']].mean()

    # Get mean confidence for all questions
    mean_all_questions = questions_with_mean['Confidence'].mean()

    # There are 948 questions (unique)
    df = pd.DataFrame({'QuestionId':range(948), 'Confidence':mean_all_questions})
    df['Confidence'] = questions_with_mean['Confidence']
    df.fillna(mean_all_questions, inplace=True)

    # Rank
    # Assume the higher confidence is the better
    # Ranking sorted should be ascending
    return __get_rank(df, 'Confidence')


def simple_mean_confidence_reversed():
    df = simple_mean_confidence().rename(columns={'ranking': 'PrevRank'})
    return __get_rank(df, 'PrevRank')


def simple_correct_rate():
    train_data = get_train_data()
    answer_data = get_answer_data()

    # Get mean correct rate for each questions
    merged_data = answer_data.merge(train_data, on='AnswerId')
    questions_with_mean = merged_data.groupby('QuestionId')[['IsCorrect']].mean().rename(columns={'IsCorrect': 'CorrectRate'})

    df = questions_with_mean
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'QuestionId'}, inplace=True)
    return __get_rank(df, 'CorrectRate')


def simple_correct_rate_reversed():
    df = simple_correct_rate().rename(columns={'ranking': 'PrevRank'})
    return __get_rank(df, 'PrevRank')


def half_correct_rate():
    train_data = get_train_data()
    answer_data = get_answer_data()

    # Get mean correct rate for each questions
    merged_data = answer_data.merge(train_data, on='AnswerId')
    questions_with_mean = merged_data.groupby('QuestionId')[['IsCorrect']].mean().rename(columns={'IsCorrect': 'CorrectRate'})
    
    # Closer to 0.5 is the better (the lower value is the better)
    df = questions_with_mean
    df['CloserValue'] = (0.5 - df['CorrectRate']).abs() * 2
    
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'QuestionId'}, inplace=True)    
    return __get_rank(df, 'CloserValue', rank_ascending=True)
    

def appropriateness():
    train_data = get_train_data()
    
    user_correct_rate_map = {}
    for user_id, correct_rate in train_data.groupby('UserId')['IsCorrect'].mean().reset_index().values:
        user_correct_rate_map[user_id] = correct_rate

    train_data['UserCorrectRate'] = \
        train_data['UserId'].map(user_correct_rate_map)

    groupby_data = train_data.groupby('QuestionId')
    question_ids = []
    apprs = []
    for _, (question_id, df) in enumerate(groupby_data):
        question_ids.append(question_id)
        apprs.append((df['IsCorrect'] - df['UserCorrectRate']).abs().mean())

    df = pd.DataFrame()
    df['QuestionId'] = question_ids
    df['Appr'] = apprs
    
    return __get_rank(df, 'Appr')
