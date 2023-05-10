import pandas as pd
from data_sources import *

def simple_confidence_average_model():
    train_data = get_train_data()
    answer_data = get_answer_data()

    # merge the datasets
    merged_data = answer_data.merge(train_data, on='AnswerId')

    # get the average confidence for each question
    average_conf_df = merged_data.groupby('QuestionId')[['Confidence']].mean()
    average_confidence = average_conf_df['Confidence'].mean()

    # fill in any missing questions with the average confidence
    results_df = pd.DataFrame({'QuestionId':range(0,1000), 'Confidence':average_confidence})
    results_df['Confidence'] = average_conf_df['Confidence']
    results_df.fillna(average_conf_df['Confidence'].mean(), inplace=True)

    # rank the questions based on Confidence
    results_df['ranking'] = results_df['Confidence'].rank(method='first', ascending=False).astype('int16')
    results_df.sort_values(by='ranking', ascending=True, inplace=True)
    results_df.drop(columns = ['Confidence'], inplace=True)  
    return results_df


print(simple_confidence_average_model())
