"""
This code is a modified version of the official documentation for the
transformer library by Hugging Face which can be found below.

We prioritized following the official documentation as close as possible to ensure we're using
robust methods. And also, to improve maintainability as they update the documentation.

https://huggingface.co/transformers/custom_datasets.html#sequence-classification-with-imdb-reviews"""

import csv
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import AdamW
from happytransformer.trainer import Trainer


class TCTrainer(Trainer):

    def __init__(self, model, model_type, tokenizer, device, logger):
        super().__init__(model, model_type, tokenizer, device, logger)

    def train(self, input_filepath, args):
        contexts, labels = self.__get_data(input_filepath)
        train_encodings = self.tokenizer(contexts, truncation=True, padding=True)
        train_dataset = TextClassificationDataset(train_encodings, labels)

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)

        optim = AdamW(self.model.parameters(), lr=args['learning_rate'])
        self.model.train()
        pbar = tqdm(total=args['epochs']*len(contexts))


        for epoch in range(args['epochs']):
            epoch_output = "Epoch: " + str(epoch) + "\n\n"
            self.logger.info(epoch_output)
            batch_num = 1
            for batch in train_loader:

                batch_output = "Batch: " + str(batch_num)
                self.logger.info(batch_output)
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optim.step()
                batch_num += 1
                pbar.update(args['batch_size'])
        pbar.close()
        self.model.eval()




    def eval(self, input_filepath, solve, output_filepath, args):
        contexts, labels = self.__get_data(input_filepath)

        correct = 0
        count = 0
        total = len(contexts)
        results = list()

        for case in zip(contexts, labels):
            context = case[0]
            label = case[1]

            output = solve(context)
            result = output["answer"]
            softmax = output["softmax"]

            # todo modify the qa functionality to output with correct capitalization

            if result == label:
                correct += 1

            results.append(
                {
                    "text": context,
                    "answer": label,
                    "result": result,
                    "correct": result == label,
                    "softmax": softmax

                }
                )
            count += 1


        score = correct/total
        ending = str(round(score, 2) * 100) + "%"

        result_output = "Evaluating Result: " + str(correct) + "/" + str(total) + " -- " + ending
        self.logger.info(result_output)

        fieldnames = ["text", "answer", "result", "correct", "softmax"]
        self._output_result_to_csv(output_filepath, fieldnames, results)

        return score



    def test(self, input_filepath, solve, output_filepath, args):
        contexts = self.__get_data(input_filepath, True)


    @staticmethod
    def __get_data(filepath, test_data=False):
        """
        Used for parsing data for training and evaluating (both contain labels)
        :param filepath: a string that contains the location of the data
        :return:
        """
        contexts = []
        labels = []
        with open(filepath, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                contexts.append(row['text'])
                if not test_data:
                    labels.append(int(row['label']))
        csv_file.close()

        if not test_data:
            return contexts, labels
        return contexts


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)