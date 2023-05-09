import pandas as pd

DATA_DIR = '../data'

train_data = pd.read_csv(f'{DATA_DIR}/train_data/train_task_3_4.csv')
answer_data = pd.read_csv(f'{DATA_DIR}/metadata/answer_metadata_task_3_4.csv')
question_data = pd.read_csv(f'{DATA_DIR}/metadata/question_metadata_task_3_4.csv')
student_data = pd.read_csv(f'{DATA_DIR}/metadata/student_metadata_task_3_4.csv')
subject_data = pd.read_csv(f'{DATA_DIR}/metadata/subject_metadata.csv')


print('train_data\n', train_data.head())
print('answer_data\n', answer_data.head())
print('question_data\n', question_data.head())
print('student_data\n', student_data.head())
print('subject_data\n', subject_data.head())
