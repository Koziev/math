"""
Тестирование точности решения арифметических задач по датасету https://huggingface.co/datasets/inkoziev/arithmetic
"""

import io
import json
import os
import re

from sklearn.model_selection import train_test_split
import torch
import numpy as np
import scipy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
import transformers
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, TrainerCallback
import tqdm
import datasets


def count_numbers(text):
    return len(re.findall(r'\d+', text))


def extract_number(text):
    m = re.search(r'\d+', text)
    if m is not None:
        return int(m.group(0))
    else:
        return None



def merge_dialog(messages):
    return '\n'.join(('- ' + msg) for msg in messages)


class FinetuneDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = 0
        self.samples = []

        self.bos_token_id = tokenizer.encode('<s>')[0]
        self.eos_token_id = tokenizer.encode('</s>')[0]
        self.pad_token_id = tokenizer.encode('<pad>')[0]

        for sample in samples:
            text = merge_dialog(sample)
            input_ids = [self.bos_token_id] + tokenizer.encode(text) + [self.eos_token_id]
            self.samples.append((text, input_ids))
            self.max_len = max(self.max_len, len(input_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        text, input_ids0 = self.samples[index]
        input_len = len(input_ids0)
        npad = self.max_len - input_len
        input_ids = input_ids0 + npad * [self.pad_token_id]
        labels = input_ids0 + npad * [-100]
        attention_mask = [1] * input_len + [0] * npad
        return {'input_ids': torch.LongTensor(input_ids), 'labels': torch.LongTensor(labels), 'attention_mask': torch.LongTensor(attention_mask)}


if __name__ == '__main__':
    tmp_dir = os.path.expanduser('~/polygon/chatbot/tmp')
    output_dir = os.path.expanduser('~/polygon/chatbot/tmp')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device={}'.format(device))

    # Датасет с сэмплами доступен тут https://huggingface.co/datasets/inkoziev/arithmetic
    ds = datasets.load_dataset('inkoziev/arithmetic')
    data = [sample['conversation'] for sample in ds['train']]

    # Вариант загрузки из локального файла
    #with open(os.path.expanduser('~/polygon/chatbot/tmp/qa_arith.json'), 'r') as f:
    #    data = json.load(f)

    train_data, test_data = train_test_split(data, test_size=0.1, random_state=123456789)

    # Из диалогов извлечем сэмплы контекст+ответ с единственным числом в ответе.
    test_samples = []
    for item in test_data:
        for istep in range(1, len(item), 2):
            if count_numbers(item[istep]) == 1:
                history = merge_dialog(item[:istep])
                reply = item[istep]
                test_samples.append((history, reply))

    # Будем использовать эту модель для файнтюна и определения качества инференса
    pretrained_model_name = 'sberbank-ai/rugpt3small_based_on_gpt2'
    learning_rate = 1e-5
    batch_size = 32

    #pretrained_model_name = 'sberbank-ai/rugpt3large_based_on_gpt2'
    #learning_rate = 1e-5
    #batch_size = 6

    #pretrained_model_name = 'sberbank-ai/ruT5-base'
    #learning_rate = 1e-6
    #batch_size = 32

    print('Loading pretrained model "{}"...'.format(pretrained_model_name))
    if 't5' in pretrained_model_name.lower():
        model = transformers.T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})

    # НАЧАЛО ОТЛАДКИ
    #train_data = train_data[:10000]
    # КОНЕЦ ОТЛАДКИ
    train_dataset = FinetuneDataset(train_data, tokenizer)

    training_args = TrainingArguments(
        report_to="none",
        evaluation_strategy='no',
        #eval_steps=5000,
        disable_tqdm=False,
        save_total_limit=0,  # Only last 1 model is saved. Older ones are deleted.
        output_dir=output_dir,
        save_strategy='no',
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        #per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.001,
        fp16=True,
        push_to_hub=False,
        logging_strategy='no',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=None,
        #compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print('Start training on...')
    trainer.train()

    print('Testing on {} samples...'.format(len(test_samples)))
    model.eval()
    sample_errs = []
    sample_hits = []
    with open(os.path.join(tmp_dir, 'test_arith_accuracy.log'), 'w') as wrt_log:
        try:
            for history, reply in test_samples:
                encoded_prompt = tokenizer.encode('<s>' + history + '\n', add_special_tokens=False, return_tensors="pt").to(device)
                output = model.generate(input_ids=encoded_prompt,
                                        max_length=train_dataset.max_len,
                                        top_k=30,
                                        top_p=0.90,
                                        do_sample=True,
                                        num_return_sequences=1,
                                        pad_token_id=tokenizer.pad_token_id)
                output_tokens = output[0].tolist()
                output_tokens = output_tokens[encoded_prompt.shape[1] + 1:]
                output_text = tokenizer.decode(output_tokens)
                if '</s>' in output_text:
                    output_text = output_text[:output_text.find('</s>')]
                if '\n' in output_text:
                    output_text = output_text[:output_text.find('\n')]

                num_num = count_numbers(output_text)
                label = 'SKIP'
                if num_num == 1:
                    true_num = extract_number(reply)
                    pred_num = extract_number(output_text)
                    sample_hits.append(true_num == pred_num)
                    if true_num == pred_num:
                        label = 'HIT'
                    else:
                        label = 'ERROR'

                    if true_num != 0:
                        err = abs(pred_num-true_num) / float(true_num)
                        sample_errs.append(err)
                        if 0 == (len(sample_errs) % 10):
                            print('support={}  mean err={:5.3}%  mean hits={:5.3f}'.format(len(sample_errs), np.mean(sample_errs)*100, np.mean(sample_hits)))
                            wrt_log.flush()
                elif num_num == 0:
                    sample_hits.append(False)

                wrt_log.write('{:5s} true_reply={} output_text={}\n\n'.format(label, reply, output_text))

                if len(sample_errs) >= 1000:
                    break
        except KeyboardInterrupt:
            print('Keyboard interrupt.')

    print('-'*80)
    print('support={}  mean err={:5.3}%  mean hits={:5.3f}'.format(len(sample_errs), np.mean(sample_errs)*100, np.mean(sample_hits)))
