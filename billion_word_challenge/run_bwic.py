from billion_word_challenge import data_collection_bwic
from happy_transformer.happy_roberta import HappyRoBERTa

def bwic_roberta():
    """
    Uses HappyRoBERTa to solve the Billion Word Imputation Challenge
    Writes the result to bwic_output.txt

    The run time is too long for a standard computer to complete.
    """
    happy_roberta = HappyRoBERTa()
    data = data_collection_bwic.get_data_bwic()
    output = open("bwic_output.txt", 'w')
    output.write("id,\"sentence\"\n")

    test_case = 0
    for case in data:
        test_case += 1
        case_array = case.split(" ")
        top_softmax = 0
        top_word = ""
        top_index = 0

        for i in range(1, len(case_array)-1):
            mask_list = case_array.copy()
            mask_list.insert(i, "<mask>")
            mask_sentence = " ".join(mask_list)
            predictions = happy_roberta.predict_k_masks(mask_sentence, 20)
            j = 0

            while j < len(predictions):
                temp_word = predictions[j][0]
                temp_softmax = predictions[j][1]
                if temp_word.isalpha():
                    if temp_softmax > top_softmax:
                        top_index = i
                        top_softmax = temp_softmax
                        top_word = temp_word
                    break
                j += 1

        case_array.insert(top_index, top_word)
        final_answer = " ".join(case_array)
        final_answer = str(test_case) + ",\"" + final_answer+"\"\n"
        output.write(final_answer)
        print(final_answer)

