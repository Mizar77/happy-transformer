"""
Tests for text classification
"""

from happytransformer.happy_text_classification import HappyTextClassification


def test_qa_train_train():
    """
    Tests
    HappyQuestionAnswering.eval()
    HappyQuestionAnswering.train()

    """
    happy_tc = HappyTextClassification()
    happy_tc.train("../data/test_tc_trainer/tc-train-eval.csv",)


def test_qa_train_eval():
    """
    Tests
    HappyQuestionAnswering.eval()
    HappyQuestionAnswering.train()

    """
    happy_tc = HappyTextClassification()
    happy_tc.eval("../data/test_tc_trainer/tc-train-eval.csv", "../data/test_tc_trainer/results/test-eval.csv")
