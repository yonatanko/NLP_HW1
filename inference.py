from preprocessing import read_test
from tqdm import tqdm
import numpy as np


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    n = len(sentence)
    tags = feature2id.feature_statistics.tags
    tags_indices_dict = create_indices_of_tags(tags)
    pi_dict = {}
    bp_dict = {}
    for i in range(n):
        pi_dict[i] = np.zeros(len(tags),len(tags))
        bp_dict[i] = np.zeros(len(tags),len(tags))

    pi_dict[0][tags_indices_dict["~"]][tags_indices_dict["~"]] = 1

    for k in range(1,n):
        for u in range(len(tags)):
            for v in range(len(tags)):
                for t in range(len(tags)):
                    pass


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()

def create_indices_of_tags(tags):
    tag_to_index = {}
    for index,tag in enumerate(tags):
        tag_to_index[tag] = index
    return tag_to_index

