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
    sentence_tags = [""] * len(sentence)
    pi = {}
    bp = {}
    for i in range(n):
        for u in tags:
            for v in tags:
                pi[(i, u, v)] = 0
                bp[(i, u, v)] = ""

    pi[(0, "*", "*")] = 1

    for k in range(1, n):
        for u in tags:
            for v in tags:
                max_prob = -np.inf
                arg_max_prob = ""
                for t in tags:
                    current = pi[(k - 1, t, u)] * q(v, t, u, pre_trained_weights, sentence,
                                                    feature2id, k, tags)
                    if current > max_prob:
                        max_prob = current
                        arg_max_prob = t

                pi[(k, u, v)] = max_prob
                bp[(k, u, v)] = arg_max_prob

    max_u_v = ("", "")
    max_num = -np.inf
    for key in pi.keys():
        if key[0] == n-1:
            if pi[key] > max_num:
                max_num = pi[key]
                max_u_v = (key[1], key[2])

    (sentence_tags[n - 2], sentence_tags[n - 1]) = (max_u_v[0], max_u_v[1])
    print(sentence_tags)

    for k in range(n - 3, -1, -1):
        print(k)
        sentence_tags.append(bp[(k + 2, sentence_tags[k + 1], sentence_tags[k + 2])])

    return sentence_tags


def q(v, t, u, pre_trained_weights, sentence, feature2id, k, tags):
    """
    q function implementation
    """
    # building history of :"(c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word)"
    nominator = 0
    denominator = 0
    # calculating the nominator and the denominator
    for tag in tags:
        if k == 1:
            history = (sentence[k], tag, sentence[k - 1], u, "*", t, sentence[k + 1])
        elif k == len(sentence) - 1:
            history = (sentence[k], tag, sentence[k - 1], u, sentence[k - 2], t, "~")
        else:
            history = (sentence[k], tag, sentence[k - 1], u, sentence[k - 2], t, sentence[k + 1])

        if history not in feature2id.histories_features.keys():
            current_calc = 1
        else:
            current_calc = np.exp(np.dot(pre_trained_weights,
                                         features_list_to_binary_vector(feature2id.histories_features[history],
                                                                        feature2id.n_total_features)))
        if tag == v:
            nominator = current_calc
        denominator += current_calc

    return nominator / denominator


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
    for index, tag in enumerate(tags):
        tag_to_index[tag] = index
    return tag_to_index


def features_list_to_binary_vector(features_indices, total_features):
    binary_vector = np.zeros(total_features)
    for feature_idx in features_indices:
        binary_vector[feature_idx] = 1
    return binary_vector
