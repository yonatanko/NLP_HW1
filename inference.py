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
    print(tags)
    sentence_tags = [""]*len(sentence)
    tag_to_idx = create_indices_of_tags(tags)
    pi = {}
    bp = {}
    for i in range(n):
        pi[i] = np.zeros((len(tags),len(tags)))
        bp[i] = np.zeros((len(tags),len(tags)))

    pi[0][tag_to_idx["*"]][tag_to_idx["*"]] = 1

    for k in range(1, n-1):
        for u in tags:
            for v in tags:
                max_prob = -np.inf
                arg_max_prob = ""
                for t in tags:
                    current = pi[k-1][tag_to_idx[t]][tag_to_idx[u]] * q(v, t, u, pre_trained_weights, sentence, feature2id, k, tags)
                    if current > max_prob:
                        max_prob = current
                        arg_max_prob = t
                pi[k][tag_to_idx[u]][tag_to_idx[v]] = max_prob
                bp[k][tag_to_idx[u]][tag_to_idx[v]] = arg_max_prob
    (sentence_tags[n-1], sentence_tags[n]) = np.unravel_index(np.argmax(pi[n]), pi[n].shape)

    for k in range(n-2,0,-1):
        sentence_tags.append(bp[k+2][sentence_tags[k+1]][sentence_tags[k+2]])
    print(sentence_tags)
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
        history = (sentence[k], tag, sentence[k - 1], u, sentence[k - 2], t, sentence[k + 1])
        print(history)
        current_calc = np.exp(np.dot(pre_trained_weights, feature2id.histories_features[history]))
        if tag == v:
            nominator = current_calc
        denominator += current_calc

    return nominator/denominator


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

