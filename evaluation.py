import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import features

def calc_preference(scores):
    preference = np.ones(len(scores), dtype=int)

    idx_two = scores > 1.5
    preference[idx_two] = 2

    return list(preference)

def evaluate(truth, submission):
    left = list(truth['left'])
    right = list(truth['right'])

    submission_left = []
    submission_right = []
    submission_preference = []
    evaluation = []
    truth_preference = list(truth['preference'])

    length = len(left)
    for idx in range(length):
        submission_left.append(left[idx])
        submission_right.append(right[idx])
        ranking_left = submission[submission.QuestionId==left[idx]].ranking.values[0]
        ranking_right = submission[submission.QuestionId==right[idx]].ranking.values[0]
        preference = 1 if ranking_left < ranking_right else 2
        submission_preference.append(preference)
        evaluation.append(1 if preference == truth_preference[idx] else 0)
        
    return sum(evaluation) / length


eval_validation = pd.read_csv('../data/test_data/quality_response_remapped_public.csv')
eval_validation['score'] = eval_validation.filter(regex='^T', axis=1).mean(axis=1)
eval_validation['preference'] = calc_preference(eval_validation['score'])

# TODO
eval_test = pd.read_csv('../data/test_data/quality_response_remapped_private.csv')
eval_test['score'] = eval_test.filter(regex='^T', axis=1).mean(axis=1)
eval_test['preference'] = calc_preference(eval_test['score'])
eval_test.head()

# evaluate

template = features.simple_confidence_average_model()
print('evaluation', evaluate(eval_validation, template))
print('evaluation', evaluate(eval_test, template))

