import pandas as pd

DATA_DIR = './data'

def get_train_data():
    return pd.read_csv(f'{DATA_DIR}/train_data/train_task_3_4.csv')

def get_answer_data():
    return pd.read_csv(f'{DATA_DIR}/metadata/answer_metadata_task_3_4.csv')

def get_question_data():
    return pd.read_csv(f'{DATA_DIR}/metadata/question_metadata_task_3_4.csv')

def get_student_data():
    return pd.read_csv(f'{DATA_DIR}/metadata/student_metadata_task_3_4.csv')

def get_subject_data():
    return pd.read_csv(f'{DATA_DIR}/metadata/subject_metadata.csv')


