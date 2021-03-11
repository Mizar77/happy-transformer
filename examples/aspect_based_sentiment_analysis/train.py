from dataset import load_data
import csv
from happytransformer.happy_aspect_based_sentiment_analysis import HappyAspectBasedSentimentAnalysis
import os


def main():
    dataset_dir = '..\\..\\..\\..\\dataset\\semeval16_task5_subtask1\\semeval16'
    train_path = os.path.join(dataset_dir, 'en_train.csv')
    test_path = os.path.join(dataset_dir, 'en_test.csv')

    happy_tc = HappyAspectBasedSentimentAnalysis(model_type="BERT",
                                                 model_name="bert-base-multilingual-cased", num_labels=3)

    before_loss = happy_tc.test(test_path)
    # happy_tc.train(train_path)
    # after_loss = happy_tc.eval(test_path)

    print("Before loss: ", before_loss.loss)
    # print("After loss: ", after_loss.loss)


def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["text", "label"])
        for case in dataset:
            # some cases have multiple labels,
            # so each one becomes its own training case
            for label in case["labels"]:
                text = case["text"]
                writter.writerow([text, label])


if __name__ == "__main__":
    main()
