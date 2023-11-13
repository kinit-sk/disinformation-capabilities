from scipy.stats import norm
from scipy import stats
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import nltk
nltk.download('punkt')


def bootstrap_ci(scores, alpha=0.95):
    """
    Bootstrapping based estimate.

    Return mean and confidence interval (lower and upper bound)
    """

    loc, scale = norm.fit(scores)
    bootstrap = [sum(random.choices(scores, k=len(scores))) /
                 len(scores) for _ in range(1000)]
    lower, upper = norm.interval(alpha, *norm.fit(bootstrap))

    return loc, lower, upper


def mann_whitney_u_test(scores1, scores2):
    """
    Mann-Whitney U test.

    Return p-value
    """

    return stats.mannwhitneyu(scores1, scores2, alternative='two-sided')[1]


def visualize_bar_plots(data, xlabels, ylabels, titles, xticklabels, yticklabels=None, ymax=None, name='graph'):
    num_plots = len(data)  # Number of bar plots
    print(f'Number of plots: {num_plots}')
    x = list(range(len(data[0])))

    if num_plots == 1:
        fig, axs = plt.subplots(1, num_plots, figsize=(6, 5))
    elif num_plots == 2:
        fig, axs = plt.subplots(1, num_plots, figsize=(12, 5))
    else:
        fig, axs = plt.subplots(1, num_plots, figsize=(18, 5))

    if num_plots == 1:
        axs.bar(x, data[0])
        axs.set_xlabel(xlabels[0])
        axs.set_ylabel(ylabels[0])
        # axs.set_title(titles[0])

        axs.set_xticks(x)
        axs.set_xticklabels(xticklabels, rotation=45)
        if ymax is not None:
            axs.set_yticks([y_i + 1 for y_i in range(ymax)])
            axs.set_yticklabels(yticklabels)

        for j, value in enumerate(data[0]):
            axs.text(j, value, str(round(value, 2)), ha='center', va='bottom')
    else:

        for i in range(num_plots):
            axs[i].bar(x, data[i])
            axs[i].set_xlabel(xlabels[i])
            axs[i].set_ylabel(ylabels[i])
            # axs[i].set_title(titles[i])

            axs[i].set_xticks(x)
            axs[i].set_xticklabels(xticklabels, rotation=45)
            if ymax is not None:
                axs[i].set_yticks([y_i + 1 for y_i in range(ymax)])
                axs[i].set_yticklabels(yticklabels)

            for j, value in enumerate(data[i]):
                axs[i].text(j, value, str(round(value, 2)),
                            ha='center', va='bottom')

    plt.subplots_adjust(bottom=0.75)
    plt.tight_layout()

    # Show the plot
    plt.savefig(f'./images/{name}.svg')
    plt.show()


def get_scores(df, questions):
    mean_scores = [
        {
            key: 0 for key in range(len(questions))
        }
        for _ in range(len(df))
    ]
    min_scores = [
        {
            key: 6 for key in range(len(questions))
        }
        for _ in range(len(df))
    ]
    max_scores = [
        {
            key: 0 for key in range(len(questions))
        }
        for _ in range(len(df))
    ]
    number_of_occurences = [
        {
            key: {
                idx: 0 for idx in range(1, 6)
            } for key in range(len(questions))
        }
        for _ in range(len(df))
    ]

    for index, (_, row) in enumerate(df.iterrows()):
        answers = row['answers']
        for answer in answers:
            for i, idx_num in enumerate(questions):
                question_answer = answer[f'{idx_num}']
                try:
                    mean_scores[index][i] += question_answer
                except:
                    print(f'index: {index}, i: {i}, idx_num: {idx_num}')
                    raise Exception
                if question_answer > max_scores[index][i]:
                    max_scores[index][i] = question_answer
                if question_answer != 0 and question_answer < min_scores[index][i]:
                    min_scores[index][i] = question_answer

                number_of_occurences[index][i][question_answer] += 1

        for idx_num in range(len(questions)):
            mean_scores[index][idx_num] /= len(answers)

    return mean_scores, min_scores, max_scores, number_of_occurences


def get_mean_max_min(mean_scores, max_scores, min_scores, questions):

    if mean_scores is not None:
        means = []
        for i in range(len(questions)):
            mean = 0
            for _, row in enumerate(mean_scores):
                mean += row[i]
            if len(mean_scores) == 0:
                means.append(0)
            else:
                means.append(mean / len(mean_scores))
    else:
        means = None

    if max_scores is not None:
        maxs = []
        for i in range(len(questions)):
            max = 0
            for _, row in enumerate(max_scores):
                if row[i] > max:
                    max = row[i]
            maxs.append(max)
    else:
        maxs = None

    if min_scores is not None:
        mins = []
        for i in range(len(questions)):
            min = 6
            for _, row in enumerate(min_scores):
                if row[i] < min and row[i] != 0:
                    min = row[i]
            mins.append(min)
    else:
        mins = None

    return means, maxs, mins


def get_statistics(df, questions, model_name='all', narrative='all'):
    questions = [i for i in range(len(questions))]  # +  ['filters']

    if model_name != 'all':
        df = df[df['model'] == model_name]
    else:
        df = df.copy()

    if narrative != 'all':
        df = df[df['narrative_idx'] == narrative]

    print(
        f'------------------------------------------------\nModel: {model_name}')
    print(f'Number of rows: {len(df)}')

    mean_scores, min_scores, max_scores, number_of_occurences = get_scores(
        df, questions)
    means, maxs, mins = get_mean_max_min(
        mean_scores, min_scores, max_scores, questions)

    visualize_bar_plots(
        data=[means],  # , maxs, mins],
        # , f'Max for each question ({model_name})', f'Min for each question ({model_name})'],
        titles=[f'Mean for each question ({model_name})'],
        xlabels=['Question number'],  # * 3,
        ylabels=['Mean score'],  # , 'Max score', 'Min score'],
        xticklabels=['Q1\n(Meaningfulness)', 'Q2\n(Article)', 'Q3\n(Agreement)',
                     'Q4\n(Disagreement)', 'Q5\n(In favor)', 'Q6\n(Against)'],
        yticklabels=['Does not apply', 'Few parts',
                     'Some parts', 'Most parts', 'Completly applies'],
        ymax=5,
        name='graph1'
    )

    data_questions = []

    for question_idx in range(len(questions)):
        data = [0 for i in range(5)]
        for occurences in number_of_occurences:
            answer = occurences[question_idx]
            for idx in range(1, 6):
                data[idx - 1] += answer[idx]
        data_questions.append(data)

    visualize_bar_plots(
        data_questions,
        titles=[f'Question {i}' for i in range(1, 7)],  # + ['Safety filters'],
        xlabels=['Answers'] * len(data_questions),
        ylabels=['Count'] * len(data_questions),
        xticklabels=[i for i in range(1, 6)],
        name='graph2'
    )

    differences_2 = {key: {'count': 0, 'indexes': []}
                     for key in range(6)}  # 2 and more
    differences_1 = {key: {'count': 0, 'indexes': []}
                     for key in range(6)}  # 1 and more
    for text_idx in range(len(df)):
        for i in range(6):
            values = list(number_of_occurences[text_idx][i].values())
            values_indexes_more_than_zero = [
                idx for idx, val in enumerate(values) if val > 0]
            if len(values_indexes_more_than_zero) > 1 and abs(values_indexes_more_than_zero[0] - values_indexes_more_than_zero[1]) >= 2:
                differences_2[i]['count'] += 1
                differences_2[i]['indexes'].append(text_idx + 1)
            if len(values_indexes_more_than_zero) > 1 and abs(values_indexes_more_than_zero[0] - values_indexes_more_than_zero[1]) == 1:
                differences_1[i]['count'] += 1
                differences_1[i]['indexes'].append(text_idx + 1)

    print(f'Differences (more than 2 levels)')
    print_differences(differences_2)
    print(f'\nDifferences (1 level)')
    print_differences(differences_1)

    return means


def get_narrative_statistics(df, models, narrative, questions):

    mean_scores = []

    if narrative != 'all':
        df = df[df['narrative_idx'] == narrative]

    for model_idx, model_name in enumerate(models):
        current_df = df[df['model'] == model_name]

        mean_score, _, _, _ = get_scores(current_df, questions)
        mean_scores.append(mean_score)

    means = []
    for model_idx, _ in enumerate(models):
        model_means, _, _ = get_mean_max_min(
            mean_scores[model_idx], None, None, questions)
        means.append(model_means)

    print(len(means))

    visualize_bar_plots(
        data=means,
        titles=[f'{model_name}' for model_name in models],
        xlabels=['Question number'] * len(models),
        ylabels=['Mean score'] * len(models),
        xticklabels=['Q1\n(Meaningfulness)', 'Q2\n(Article)', 'Q3\n(Agreement)',
                     'Q4\n(Disagreement)', 'Q5\n(In favor)', 'Q6\n(Against)'],
        yticklabels=['Does not apply', 'Few parts',
                     'Some parts', 'Most parts', 'Completly applies'],
        ymax=5
    )


def stats_per_text(df, num_questions, indexes=None):
    for row_index, (_, row) in enumerate(df.iterrows()):
        if indexes is not None and row_index + 1 not in indexes:
            continue
        answers = row['answers']
        text = answers[0]['text']
        model_name = answers[0]['model']

        print(f'\n---------------------------------------------------------------------------\n')
        print(f'Model: {model_name}\n')
        print(text)
        print(f'\nText {row_index + 1} statistics:\n')

        for q_idx in range(num_questions):
            print(
                f"Q{q_idx + 1}: {' '.join(map(str, [answer[f'{q_idx}'] for answer in answers]))}")

        print(
            f"Safety filters: {' '.join(map(str, [answer['filters'] for answer in answers]))}")
        for idx, answer in enumerate(answers):
            print(f"A{idx+1} - Note: {answer['evaluation']}")


def print_questions(questions):
    for idx, question in enumerate(questions):
        if idx == 0 or question['type'] != questions[idx - 1]['type']:
            if idx != 0:
                print('\n')
            print(question['type'])
        print(f'Q{idx + 1}: {question["question"]}')


def remove_narrative(lst, narrative):
    for i in range(len(lst)):
        if lst[i]["narrative"] == narrative:
            lst.pop(i)
            break

    return lst


def print_narratives(narratives):
    for idx, narrative in enumerate(narratives):
        print(f'{idx + 1}. {narrative["narrative"]}')


def brief_comparison(df, questions, model_name='all', narrative='all'):
    questions = [i for i in range(len(questions))]

    if model_name != 'all':
        df = df[df['model'] == model_name]
    else:
        df = df.copy()

    if narrative != 'all':
        df = df[df['narrative_idx'] == narrative]

    mean_scores, _, _, _ = get_scores(df[df['brief'] == False], questions)
    brief_mean_scores, _, _, _ = get_scores(df[df['brief'] == True], questions)

    means, _, _ = get_mean_max_min(mean_scores, None, None, questions)
    brief_means, _, _ = get_mean_max_min(
        brief_mean_scores, None, None, questions)

    visualize_bar_plots(
        data=[means, brief_means],
        titles=['Without brief', 'With brief'],
        xlabels=['Question number'] * len(questions),
        ylabels=['Mean score'] * 2,
        xticklabels=['Q1\n(Meaningfulness)', 'Q2\n(Article)', 'Q3\n(Agreement)',
                     'Q4\n(Disagreement)', 'Q5\n(In favor)', 'Q6\n(Against)'],
        yticklabels=['Does not apply', 'Few parts',
                     'Some parts', 'Most parts', 'Completly applies'],
        ymax=5
    )


def check_answer_points(entries):
    for entry in entries:
        answers = entry
        if answers['0'] < 3 or answers['1'] < 3:
            return False
    return True


def filter_bad_texts(df):
    return df[df['answers'].apply(check_answer_points)]
