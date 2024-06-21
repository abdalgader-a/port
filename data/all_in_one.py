"""
This script generate (num_devices) data splits multiple GPUs and one split in case of a CPU/1GPU.
We ran this script on 8 different GPU devices.
"""
import os
import re
import csv
import copy
import math
import glob
import torch
import random
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

introducers_of_final_answer = ["Therefore, the answer is ", "The answer is then ",
                               "The solution is ", "The solution to the problem is ",
                               "This means that the answer is ", "Finally, the solution is "]
fixed_introducer_of_final_answer = "The answer is "


def template(question, steps):
    questions_cot = [
        """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
        A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.""",
        """Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
        A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.""",
        """Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
        A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.""",
        """Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
        A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.""",
        """Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
        A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.""",
        """Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
        A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.""",
        """Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
        A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.""",
        """Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
        A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8."""
    ]

    random.shuffle(questions_cot)
    n_cot = random.choice([2, 3, 4, 5, 6])
    selected_questions = questions_cot[:n_cot]

    content2 = selected_questions[0]
    for j in range(n_cot - 1):
        content2 += f"\n\n{selected_questions[j + 1]}"
    content2 += f"\n\nQ: {question}\nA: {' '.join(steps)}"

    return content2


def get_actual_answer(datum):
    # for AQuA dataset only - copies the answer from the options field
    if datum['correct'] not in 'ABCDE':
        return None
    return datum['options'][ord(datum['correct']) - ord('A')][2:]


def parse_answer_lines(item: dict, keep_brackets=False, fixed_introducer=False):
    aqua = False
    if 'actual_answer' in item.keys():  # for AQuA dataset
        aqua = True
    answers = item['rationale' if aqua else 'answer'].split('\n')
    answers = [answer.rstrip('.') + '.' for answer in answers]
    for i in range(len(answers)):
        if not keep_brackets:
            answers[i] = re.sub(r'<<.*?>>', '', answers[i])
        if i == len(answers) - 1:
            if fixed_introducer:
                introducer = fixed_introducer_of_final_answer
            else:
                introducer = random.choice(introducers_of_final_answer)
            if not aqua:
                assert answers[i].startswith("#### ")  # for GSM8k dataset
                # replace that with one of the introducers of the final answer (randomly chosen)
                answers[i] = answers[i].replace("#### ", introducer)
            else:
                answers[i] = introducer + actual_answer
    return answers


def sanity_check_answers(answers: list):
    for i, answer in enumerate(answers):
        if i == len(answers) - 1:
            return answer.startswith("#### ")


def sanity_check_datasets(dataset):
    for item in dataset:
        answers = item['answer'].split('\n')
        if not sanity_check_answers(answers):
            return False
    return True


def corrupt_part_of_answer(answer: str, digit_corruption_probability=1):
    # this function corrupts a part of the answer
    # by replacing each digit with a random digit
    # first, it checks if the part of the answer contains a digit
    # returns the new answer and a boolean that is True if the answer was not corrupted
    if not any(char.isdigit() for char in answer):
        return answer, True
    new_answer = ""
    for char in answer:
        if char.isdigit():
            # change it with probability digit_corruption_probability
            if random.random() < digit_corruption_probability:
                new_answer += str(random.randint(0, 9))
            else:
                new_answer += char
        else:
            new_answer += char
    return new_answer, new_answer == answer


def generate(sample, models, max_steps, num_iter, keep_brackets=False, add_uncorrupted_llama=False,
             fixed_introducer=False):
    row = []
    question = sample["question"].strip()
    steps = parse_answer_lines(sample, keep_brackets, fixed_introducer)
    steps = [step.strip() for step in steps]

    # append query and answer
    if "options" in sample.keys():  # AQuA dataset
        row.append(question + '\n' + ' '.join(sample['options']))
    else:  # GSM8k dataset
        row.append(question)
    row += steps
    row += ["null"] * (max_steps - len(steps))  # fill the steps difference with 'null'

    # append corruptions
    for answer in steps:
        for j in range(num_iter):
            new_answer, equal = corrupt_part_of_answer(answer)
            if not equal:
                if keep_brackets:
                    row.append(new_answer)
                else:
                    row.append(re.sub(r'<<.*?>>', '', new_answer))
            else:
                row.append("null")
    row += ["null"] * ((max_steps - len(steps)) * num_iter)  # fill the steps difference with 'null'

    # append generations
    for name, pipe in models.items():
        all_llm_answers = []
        for j, step in enumerate(steps):
            if keep_brackets:
                query = question + " " + " ".join(steps[:j])
            else:
                query = re.sub(r'<<.*?>>', '', question + " " + " ".join(steps[:j]))
            query = template(query, steps[:j])
            for _ in range(num_iter):
                llm_answer = pipe(query, eos_token_id=pipe.tokenizer.eos_token_id, temperature=1)[0]
                # Setting the character limit to 1000 because why not?
                llm_answer = llm_answer['generated_text'].split('\n')[0].split('. ')[0].strip()[:1000]
                if len(llm_answer) == 0:
                    llm_answer = 'null'
                    continue
                if keep_brackets:
                    all_llm_answers.append(llm_answer)
                else:
                    all_llm_answers.append(re.sub(r'<<.*?>>', '', llm_answer))
        corrupted_all_llm_answers = [corrupt_part_of_answer(llm_answer)[0] for llm_answer in all_llm_answers]
        row.extend(corrupted_all_llm_answers)
        row += ["null"] * ((max_steps - len(steps)) * num_iter)  # fill the steps difference with 'null'
        if 'llama' in name and add_uncorrupted_llama:
            row.extend(all_llm_answers)
            row += ["null"] * ((max_steps - len(steps)) * num_iter)  # fill the steps difference with 'null'
    return row


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gsm8k", help="dataset name")
    parser.add_argument('--batch_id', type=int, default=0, help="number of batches per GPU/CPU")
    parser.add_argument('--device', type=int, default=0, help="GPU index")
    parser.add_argument('--repeat', type=int, default=3, help='how many times we query llm and digit corrupt')
    parser.add_argument('--keep_brackets', action='store_true', help='keep calculator annotation <<>> on gsm8k dataset')
    parser.add_argument('--add_uncorrupted_llama', action='store_true', help='add uncorrupted llama-7b',)
    parser.add_argument('--fixed_introducer', action='store_true')
    args = parser.parse_args()

    repeat = args.repeat
    keep_brackets = args.keep_brackets
    add_uncorrupted_llama = args.add_uncorrupted_llama
    fixed_introducer = args.fixed_introducer

    # load row dataset
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", name="main")
        assert sanity_check_datasets(dataset['train'])
        assert sanity_check_datasets(dataset['test'])
    elif args.dataset == "aqua":
        dataset = load_dataset("aqua_rat", name="raw")
    else:
        raise ValueError("Unknown dataset")
    split = ["train", "test"]

    # weak llm
    # add model path and uncomment 'iterative' in case iterative model exist
    models_dict = {
        "gemma2b": "google/gemma-2b",
        "llama7b": "meta-llama/Llama-2-7b-hf",
        "llama7b-it": "meta-llama/Llama-2-7b-chat-hf",
        # "iterative": "model_path"
    }

    # get max number of reasoning steps & construct the dataframe header accordingly.
    max_steps_dict = {"train": 0, "test": 0}
    headers = {"train": ["question"], "test": ["question"]}
    for split in max_steps_dict.keys():
        for datum in tqdm(dataset[split]):
            if args.dataset == "gsm8k":
                reasoning_trace = datum['answer']
            elif args.dataset == "aqua":
                reasoning_trace = datum['rationale']
            else:
                raise ValueError("Unknown dataset")
            max_steps_dict[split] = min(max(len(reasoning_trace.split('\n')), max_steps_dict[split]),
                                        12)  # Setting 12 as the largest number of steps
        # header of ground-truth reasoning steps
        for i in range(max_steps_dict[split]):
            headers[split].append(f"z{i + 1}")
        # corruption header
        zs = copy.deepcopy(headers[split][1:])
        for z in zs:
            for i in range(repeat):
                headers[split].append(f"{z}_c{i + 1}")

        for model_name in models_dict.keys():
            for z in zs:
                for i in range(repeat):
                    headers[split].append(f"{model_name}_{z}_g{i + 1}")
            if 'llama' in model_name and add_uncorrupted_llama:
                for z in zs:
                    for i in range(repeat):
                        headers[split].append(f"{model_name}_{z}_g{i + 1}_unc")

    # initiate text-generation model
    pipelines = {}
    print("Load model from HF ...")
    for model_name, model_path in models_dict.items():
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = pipeline(task='text-generation', model=model, tokenizer=tokenizer, framework='pt',
                       temperature=1,
                       do_sample=True,
                       max_new_tokens=128,
                       return_full_text=False,
                       device=f"cuda:{args.device}" if torch.cuda.is_available() else 'cpu'
                       )
        pipelines[model_name] = llm

    # generate full rows
    for split in ['train', 'test']:
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        batch_size = math.floor(len(dataset[split]) / num_devices)
        start = batch_size * args.batch_id
        end = (start + batch_size) if args.batch_id < (num_devices-1) else (start + batch_size + 1)
        data = Dataset.from_dict(dataset[split][start: end])

        print(f"Generating {split} data....")
        if not os.path.exists(f"{split}_splits"):
            os.makedirs(f"{split}_splits")
        with open(f"{split}_splits/{split}_{args.batch_id}.csv", 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(headers[split])
            for datum in tqdm(data):
                if args.dataset == "aqua":
                    reasoning_trace = datum['rationale']
                    if len(reasoning_trace.split('\n')) > 12:
                        continue
                    actual_answer = get_actual_answer(datum)
                    if actual_answer is None:
                        continue
                    datum['actual_answer'] = actual_answer
                row = generate(datum, pipelines, max_steps_dict[split], repeat, keep_brackets, add_uncorrupted_llama,
                               fixed_introducer)
                csv_writer.writerow(row)

        # concatenate csv splits into one csv file...
        split_dirs = glob.glob(f"{split}_splits/*.csv")
        all_data = pd.concat([pd.read_csv(path) for path in split_dirs], ignore_index=True).fillna('null')
        all_data.to_csv(f'all_{split}.csv', index=False)

