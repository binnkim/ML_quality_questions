
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import features


def calc_preference(scores):
    preference = np.ones(len(scores), dtype=int)

    idx_two = scores > 1.5
    preference[idx_two] = 2

    return list(preference)


def __evaluate(truth, truth_preference, submission):
    left = list(truth['left'])
    right = list(truth['right'])

    submission_preference = []
    evaluation = []

    length = len(left)
    for idx in range(length):
        ranking_left = submission[submission.QuestionId==left[idx]].ranking.values[0]
        ranking_right = submission[submission.QuestionId==right[idx]].ranking.values[0]
        preference = 1 if ranking_left < ranking_right else 2 # 1 is left, 2 is right : preference means the higher quality of a question
        submission_preference.append(preference)
        evaluation.append(1 if preference == truth_preference[idx] else 0)
    return sum(evaluation) / len(evaluation)


def evaluate(truth, submission):
    return __evaluate(truth, calc_preference(truth['score']), submission)


def evaluate2(truth, submission):
    evaluation_with_panels = [
        __evaluate(truth, truth['T1_ALR'], submission),
        __evaluate(truth, truth['T2_CL'], submission),
        __evaluate(truth, truth['T3_GF'], submission),
        __evaluate(truth, truth['T4_MQ'], submission),
        __evaluate(truth, truth['T5_NS'], submission)
    ]
    return max(evaluation_with_panels)


eval_validation = pd.read_csv('../data/test_data/quality_response_remapped_public.csv')
eval_validation['score'] = eval_validation.filter(regex='^T', axis=1).mean(axis=1)
eval_validation['preference'] = calc_preference(eval_validation['score'])

eval_test = pd.read_csv('../data/test_data/quality_response_remapped_private.csv')
eval_test['score'] = eval_test.filter(regex='^T', axis=1).mean(axis=1)
eval_test['preference'] = calc_preference(eval_test['score'])


def run(template):
    print('evaluation (rounding) public', evaluate(eval_validation, template))
    print('evaluation (rounding) private', evaluate(eval_test, template))
    print('evaluation2 (max val) pulic', evaluate2(eval_validation, template))
    print('evaluation2 (max val) private', evaluate2(eval_test, template))


FEATURE_LIST = [
    'simple_mean_confidence',
    'simple_mean_confidence_reversed',
    'simple_correct_rate',
    'simple_correct_rate_reversed',
    'half_correct_rate',
    'appropriateness',
]
# evaluate
for feature_name in FEATURE_LIST:
    print('==================================')
    print(feature_name)
    run(getattr(features, feature_name)())


