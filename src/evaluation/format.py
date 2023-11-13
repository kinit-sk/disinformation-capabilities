import pandas as pd


def change_narrative_index(df):
    new_index = [f'\\textbf{{N{i + 1}}}' for i in range(len(df))]
    df.index = new_index

    df.rename(columns={'narrative': '\\text{Narrative}',
              'category': '\\textbf{Category}'}, inplace=True)

    return df


def change_model_names(df, set_index=True):
    new_index = []
    names = df.index if set_index else df.columns

    for index in names:
        if index == 'gpt-3.5-turbo':
            new_index.append('ChatGPT')
        elif index == 'text-davinci-003':
            new_index.append('GPT-3 Davinci')
        elif index == 'text-curie-001':
            new_index.append('GPT-3 Curie')
        elif index == 'text-babbage-001':
            new_index.append('GPT-3 Babbage')
        elif index == 'falcon-40b-instruct':
            new_index.append('Falcon')
        elif index == 'opt-iml-max-30b':
            new_index.append('OPT-IML-Max')
        elif index == 'vicuna-33b-v1.3':
            new_index.append('Vicuna')
        elif index == 'Llama-2-70b-chat-hf':
            new_index.append('Llama')
        elif index == 'gpt-4':
            new_index.append('GPT-4')
        elif index == 'Mistral-7B-Instruct-v0.1':
            new_index.append('Mistral')
        else:
            new_index.append(f'{index}')

    if set_index:
        df.index = new_index
    else:
        df.columns = new_index

    return df


def rename_detectors(df):
    new_index = []
    for index in df.index:
        if index == 'gpt2-finetuned-en3-all':
            new_index.append('GPT-2*')
        elif index == 'electra-small-discriminator-finetuned-en3-all':
            new_index.append('ELECTRA\\textsubscript{SMALL}*')
        elif index == 'electra-large-discriminator-finetuned-en3-all':
            new_index.append('ELECTRA\\textsubscript{LARGE}*')
        elif index == 'bert-base-multilingual-cased-finetuned-en3-all':
            new_index.append('BERT\\textsubscript{BASE}*')
        elif index == 'roberta-large-openai-detector-finetuned-en3-all':
            new_index.append('RoBERTa\\textsubscript{LARGE} OpenAI Detector*')
        elif index == 'xlm-roberta-large-finetuned-en3-all':
            new_index.append('XLM-RoBERTa\\textsubscript{LARGE}*')
        elif index == 'mdeberta-v3-base-finetuned-en3-all':
            new_index.append('MDeBERTa\\textsubscript{BASE}*')
        elif index == 'gpt2-medium-finetuned-en3-all':
            new_index.append('GPT-2\\textsubscript{MEDIUMl}*')
        elif index == 'mGPT-finetuned-en3-all':
            new_index.append('mGPT*')
        elif index == 'opt-iml-max-1.3b-finetuned-en3-all':
            new_index.append('OPT-IML-Max-1.3B*')
        elif index == 'electra-large-discriminator-finetuned-en3-gpt-3.5-turbo':
            new_index.append('ELECTRA\\textsubscript{LARGE} (ChatGPT)*')
        elif index == 'electra-large-discriminator-finetuned-en3-opt-iml-max-1.3b':
            new_index.append(
                'ELECTRA\\textsubscript{LARGE} (OPT-IMAL-Max-1.3B)*')
        elif index == 'electra-large-discriminator-finetuned-en3-text-davinci-003':
            new_index.append('ELECTRA\\textsubscript{LARGE} (GPT-3 Davinci)*')
        elif index == 'electra-large-discriminator-finetuned-en3-vicuna-13b':
            new_index.append('ELECTRA\\textsubscript{LARGE} (Vicuna 13B)*')
        elif index == 'electra-large-discriminator-finetuned-en3-gpt-4':
            new_index.append('ELECTRA\\textsubscript{LARGE} (GPT-4)*')
        elif index == 'roberta-large-openai-detector':
            new_index.append('RoBERTa\\textsubscript{LARGE} OpenAI Detector')
        elif index == 'grover':
            new_index.append('Grover')
        elif index == 'llmdet':
            new_index.append('LLMDet')
        elif index == 'zerogpt':
            new_index.append('ZeroGPT')
        elif index == 'gptzero':
            new_index.append('GPTZero')
        elif index == 'gltr':
            new_index.append('GLTR')
        elif index == 'longformer':
            new_index.append('Longformer')
        elif index == '':
            new_index.append('ChatGPT Detector RoBERTa')
        else:
            new_index.append(index)
    df.index = new_index

    return df


def format_table(df, axis=1, rounding=5):
    def nth_greatest(row, n):
        unique_values = row.unique()
        unique_values.sort()

        if len(unique_values) < n:
            return None

        return unique_values[-n]

    max_vals = df.max(axis=axis)
    second_greatest_vals = df.apply(
        lambda row: nth_greatest(row, 2), axis=axis)

    def custom_format(val, row_idx, col_idx):
        # use custom rounding
        if axis == 1 and val == max_vals[row_idx]:
            return f"\\textbf{{{val:.{rounding}f}}}"
        if val == max_vals[row_idx]:
            return f"\\textbf{{{val:.{rounding}f}}}"
        elif val == second_greatest_vals[row_idx]:
            return f"\\underline{{{val:.{rounding}f}}}"
        else:
            return f"{val:.{rounding}f}"

    formatted_df = pd.DataFrame(index=df.index, columns=df.columns)

    for row_idx, row in enumerate(df.values):
        formatted_row = [custom_format(val, row_idx, col_idx)
                         for col_idx, val in enumerate(row)]
        formatted_df.iloc[row_idx] = formatted_row

    return formatted_df
