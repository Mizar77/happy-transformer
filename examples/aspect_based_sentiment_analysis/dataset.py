import json
import os
import xml.dom.minidom
import pandas


def load_data(path):
    dataset_json_dir = '..\\..\\..\\..\\dataset\\semeval16_task5_subtask1\\semeval16'
    train_json_path = os.path.join(dataset_json_dir, 'en_train.xml')
    test_json_path = os.path.join(dataset_json_dir, 'en_test.xml')
    df = xml2df(path)
    return df


def xml2df(path):
    domTree = xml.dom.minidom.parse(path)
    rootNode = domTree.documentElement
    reviews = rootNode.getElementsByTagName('Review')

    data = []
    for review in reviews:
        all_sentences = review.getElementsByTagName("sentences")
        for sentences in all_sentences:
            sentence_list = sentences.getElementsByTagName("sentence")
            for sentence in sentence_list:
                id = sentence.getAttribute("id")
                text = sentence.getElementsByTagName("text")[0].firstChild.data
                pre_target = None
                for opinions in sentence.getElementsByTagName("Opinions"):
                    opinion_list = opinions.getElementsByTagName("Opinion")
                    for opinion in opinion_list:
                        polarity = opinion.getAttribute("polarity")
                        # category = opinion.getAttribute("category")
                        target = opinion.getAttribute("target")
                        if target != pre_target:
                            data.append([id, text, target, polarity])
                        else:
                            data.pop()
    df = pandas.DataFrame(data, columns=['id', 'text', 'target', 'polarity'])
    return df
