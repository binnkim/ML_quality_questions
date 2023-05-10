import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_preference(scores):
    preference = np.ones(len(scores), dtype=int)

    idx_two = scores > 1.5
    preference[idx_two] = 2
    
    return list(preference)

def evaluate(truth, submission):
    # extract ranking
    left = list(truth.left)
    right = list(truth.right)
#     if len(left) != len(right):
#     message = 'left and right lengths are not the same'
#     sys.exit(message)

    submission_left = []
    submission_right = []
    submission_preference = []
    for idx in range(len(left)):
        submission_left.append(left[idx])
        submission_right.append(right[idx])
        ranking_left = submission[submission.QuestionId==left[idx]].ranking.values[0]
        ranking_right = submission[submission.QuestionId==right[idx]].ranking.values[0]
        preference = 1 if ranking_left < ranking_right else 2
        submission_preference.append(preference)
    print(submission_preference)



data = pd.read_csv('../data/train_data/train_task_3_4.csv')

eval_validation = pd.read_csv('../data/test_data/quality_response_remapped_public.csv')

eval_validation['score'] = eval_validation.filter(regex='^T', axis = 1).mean(axis=1)
eval_validation['score'].hist()
plt.show()

print('unique\n', pd.unique(eval_validation['T1_ALR']))
print(pd.unique(eval_validation['T2_CL']))

eval_validation['preference'] = calc_preference(eval_validation['score'])
eval_validation['preference'].hist()
print('eval_validation\n', eval_validation.head())
plt.show()


eval_test = pd.read_csv('../data/test_data/quality_response_remapped_private.csv')
eval_test.head()

# evaluate

template = pd.read_csv('submission/template.csv')
template['ranking'] = 1


evaluate(eval_validation, template)
