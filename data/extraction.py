"""
This file contain extractor utils used to extract the training data for different setup mentioned in the PORT paper
"""

import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field


@dataclass
class ExtractorScriptArguments:
    scheme: str = field(default="sft", metadata={"help": "the construction scheme"})
    llm: str = field(default=None, metadata={"help": "The llm model used for data corruption"})
    dataset: str = field(default="gsm8k", metadata={"help": "The dataset to use"})


def remove_tail_nulls(input_list):
    for i in range(len(input_list) - 1, -1, -1):
        if input_list[i] != 'null':
            return input_list[:i + 1]

    return input_list


def extract(tokenizer, dataframe, scheme, llm=None):
    if scheme == 'sft':
        out = pd.DataFrame(columns=['prompt', 'completion'])
        for row in tqdm(dataframe.itertuples()):
            steps = remove_tail_nulls(row[2:dataframe.columns.get_loc('z1_c1') + 1])
            steps = [s[:-1] if s.endswith('.') else s for s in steps]
            for i, step in enumerate(steps):
                if i == 0:
                    out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]), step + "."]
                else:
                    out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]) + ".", step + "."]
        return out
    
    if 'kto' in scheme:
        if scheme == 'kto_corr_only_1':
            out = pd.DataFrame(columns=['prompt', 'completion', 'label'])
            for row in tqdm(dataframe.itertuples()):
                steps = remove_tail_nulls(row[2:dataframe.columns.get_loc('z1_c1') + 1])
                steps = [s[:-1] if s.endswith('.') else s for s in steps]
                start = dataframe.columns.get_loc('z1_c1') + 1
                corrs = row[start: start + (3 * len(steps))]
                for i, step in enumerate(steps):
                    if i == 0:
                        out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]), step + ".", True]
                    else:
                        out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]) + ".", step + ".", True]
                    out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]), corrs[i * 3], False]
            return out
        else:
            raise NotImplementedError(f"Scheme {scheme} not implemented yet")

    if 'corr_only' in scheme:
        out = pd.DataFrame(columns=['prompt', 'chosen', 'rejected'])
        for row in tqdm(dataframe.itertuples()):
            steps = remove_tail_nulls(row[2:dataframe.columns.get_loc('z1_c1') + 1])
            steps = [s[:-1] if s.endswith('.') else s for s in steps]
            start = dataframe.columns.get_loc('z1_c1') + 1
            corrs = row[start: start + (3 * len(steps))]
            for i, step in enumerate(steps):
                if scheme == 'corr_only_3':
                    for j, corr in enumerate(corrs[3 * i: 3 * i + 3]):
                        if i == 0:
                            out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]), step + ".", corr]
                        else:
                            out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]) + ".", step + ".", corr]
                elif scheme == 'corr_only_1':
                    if i == 0:
                        out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]), step + ".", corrs[i * 3]]
                    else:
                        out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]) + ".", step + ".", corrs[i * 3]]
        return out

    if 'llm_only' in scheme:
        out = pd.DataFrame(columns=['prompt', 'chosen', 'rejected'])
        for row in tqdm(dataframe.itertuples()):
            steps = remove_tail_nulls(row[2:dataframe.columns.get_loc('z1_c1') + 1])
            steps = [s[:-1] if s.endswith('.') else s for s in steps]
            if 'unc' in scheme:
                start = dataframe.columns.get_loc(f"{llm}_z1_g1_unc") + 1
            else:
                start = dataframe.columns.get_loc(f"{llm}_z1_g1") + 1
            corrs = row[start: start + (3 * len(steps))]
            for i, step in enumerate(steps):
                if 'llm_only_3' in scheme:
                    for j, corr in enumerate(corrs[3 * i: 3 * i + 3]):
                        # skip long llm generation to avoid OOM
                        a = len(tokenizer(row.question + " " + ". ".join(steps[:i]) + "." + step + "." + corr)[
                                    'input_ids'])
                        if a > 400:
                            print('found!!!')
                            continue
                        if i == 0:
                            out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]), step + ".", corr]
                        else:
                            out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]) + ".", step + ".", corr]
                elif 'llm_only_1' in scheme:
                    # skip long llm generation to avoid OOM
                    a = len(tokenizer(row.question + " " + ". ".join(steps[:i])+ "." + step + "." + corrs[i * 3])['input_ids'])
                    if a > 400:
                        print('found!!!')
                        continue
                    if i == 0:
                        out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]), step + ".", corrs[i * 3]]
                    else:
                        out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]) + ".", step + ".", corrs[i * 3]]

        return out

    if "llms_mix" in scheme:
        out = pd.DataFrame(columns=['prompt', 'chosen', 'rejected'])
        for row in tqdm(dataframe.itertuples()):
            steps = remove_tail_nulls(row[2:dataframe.columns.get_loc('z1_c1') + 1])
            steps = [s[:-1] if s.endswith('.') else s for s in steps]
            g_start = dataframe.columns.get_loc(f"gemma2b-it_z1_g1") + 1
            l_start = dataframe.columns.get_loc(f"llama7b-it_z1_g1") + 1
            corrs = {"gemma": row[g_start: g_start + (3 * len(steps))],
                     "llama": row[l_start: l_start + (3 * len(steps))]}
            for i, step in enumerate(steps):
                if scheme == 'llms_mix_3':
                    for model in corrs.keys():
                        for j, corr in enumerate(corrs[model][3 * i: 3 * i + 3]):
                            # skip long llm generation to avoid OOM
                            a = len(tokenizer(
                                row.question + " " + ". ".join(steps[:i]) + "." + step + "." + corr)[
                                        'input_ids'])
                            if a > 400:
                                print('found!!!')
                                continue
                            if i == 0:
                                out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]), step + ".", corr]
                            else:
                                out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]) + ".", step + ".", corr]
                elif scheme == 'llms_mix_1':
                    for model in corrs.keys():
                        # skip long llm generation to avoid OOM
                        a = len(tokenizer(
                            row.question + " " + ". ".join(steps[:i]) + "." + step + "." + corrs[model][i * 3])[
                                    'input_ids'])
                        if a > 400:
                            print('found!!!')
                            continue
                        if i == 0:
                            out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]), step + ".",
                                                 corrs[model][i * 3]]
                        else:
                            out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]) + ".", step + ".",
                                                 corrs[model][i * 3]]
        return out

    if "corr_llm" in scheme:
        out = pd.DataFrame(columns=['prompt', 'chosen', 'rejected'])
        for row in tqdm(dataframe.itertuples()):
            steps = remove_tail_nulls(row[2:dataframe.columns.get_loc('z1_c1') + 1])
            steps = [s[:-1] if s.endswith('.') else s for s in steps]
            corr_start = dataframe.columns.get_loc('z1_c1') + 1
            l_start = dataframe.columns.get_loc(f"{llm}_z1_g1") + 1
            corrs = {"corr": row[corr_start: corr_start + (3 * len(steps))],
                     "llm": row[l_start: l_start + (3 * len(steps))]}
            for i, step in enumerate(steps):
                if scheme == 'corr_llm_3':
                    for type_ in corrs.keys():
                        for j, corr in enumerate(corrs[type_][3 * i: 3 * i + 3]):
                            # skip long llm generation to avoid OOM
                            a = len(tokenizer(row.question + " " + ". ".join(steps[:i]) + "." + step + "." + corr)['input_ids'])
                            if a > 400:
                                print('found!!!')
                                continue
                            if i == 0:
                                out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]), step + ".", corr]
                            else:
                                out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]) + ".", step + ".", corr]
                elif scheme == 'corr_llm_1':
                    for type_ in corrs.keys():
                        # skip long llm generation to avoid OOM
                        a = len(tokenizer(row.question + " " + ". ".join(steps[:i]) + "." + step + "." + corrs[type_][i * 3])[
                                    'input_ids'])
                        if a > 400:
                            print('found!!!')
                            continue
                        if i == 0:
                            out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]), step + ".",
                                                 corrs[type_][i * 3]]
                        else:
                            out.loc[len(out)] = [row.question + " " + ". ".join(steps[:i]) + ".", step + ".",
                                                 corrs[type_][i * 3]]
        return out


def data_extractor(tokenizer, scheme, model=None, split=None):
    valid_schemes = ["sft", "corr_only_3", "corr_only_1", "kto_corr_only_1", 'llm_only_1', 'llm_only_3', "llms_mix_1",
                     "llms_mix_3", "corr_llm_1", "corr_llm_3", 'unc_llm_only_1', 'unc_llm_only_3']
    valid_models = ['gemma2b-it', 'llama7b-it', 'llama7b', 'iterative', None]
    if scheme not in valid_schemes:
        raise ValueError("results: scheme must be one of %r." % valid_schemes)
    if model not in valid_models:
        raise ValueError("results: scheme must be one of %r." % valid_models)
    if 'llm_' in scheme and model is None:
        raise ValueError('You must specify the value of --model from [\'gemma2b-it\', \'llama7b-it\', \'llama7b\', \'iterative\']')

    file_name = f"../data/all_{split}.csv"
    all_data = pd.read_csv(file_name, keep_default_na=False)
    print(f"Successfully loaded {file_name} data")

    return extract(tokenizer, all_data, scheme, model)
