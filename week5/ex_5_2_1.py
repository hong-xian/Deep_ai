import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_glove_vec(glove_file):
    with open(glove_file, "r", encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return words, word_to_vec_map


def cosine_similarity(u, v):
    """
    compute the cosine similarity between two vector
    """
    inner_pro = np.dot(u, v)
    norm_u = np.sqrt(np.sum(u ** 2))
    norm_v = np.sqrt(np.sum(v ** 2))
    cosine = inner_pro / (norm_u * norm_v)
    return cosine


# words_set, word_to_vec_maps = read_glove_vec("/Users/xiujing/Desktop/DL/deep_ai/ex5_2/data/glove.6B.50d.txt")
print("*********", device)
words_set, word_to_vec_maps = read_glove_vec("/home/liushuang/PycharmProjects/lab/mydata/ex5_2/glove.6B.50d.txt")
# print(len(words_set), len(word_to_vec_maps["the"]))
# # 400000, 50
# father = word_to_vec_maps["father"]
# mother = word_to_vec_maps["mother"]
# ball = word_to_vec_maps["ball"]
# crocodile = word_to_vec_maps["crocodile"]
# france = word_to_vec_maps["france"]
# italy = word_to_vec_maps["italy"]
# paris = word_to_vec_maps["paris"]
# rome = word_to_vec_maps["rome"]
# print("cosine_similarity(father, mother)=", cosine_similarity(father, mother))
# print("cosine_similarity(ball, crocodile)=", cosine_similarity(ball, crocodile))
# print("cosine_similarity(france-paris, rome-italy)=", cosine_similarity(france - paris, rome - italy))


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    :param word_a: word of string type
    :param word_b: word of string type
    :param word_c: word of string type
    :param word_to_vec_map: dictionary, maps of words to Glove vector
    :return:
        best_word -- the word satisfies that (v_b - v_a) and (v_best_word - v_c) are closet under cosine similarity
    """
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    words = word_to_vec_map.keys()
    max_cosine_sim = -100
    best_word = None

    for word in words:
        if word in [word_a, word_b, word_c]:
            continue
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[word] - e_c))

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word

    return best_word


# triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'),
#                  ('man', 'woman', 'boy'), ('small', 'smaller', 'big')]
# for triad in triads_to_try:
#     print('{} -> {} <====> {} -> {}'.format(*triad, complete_analogy(*triad, word_to_vec_maps)))


g = word_to_vec_maps["woman"] - word_to_vec_maps['man']
# name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']
# for w in name_list:
#     print(w, cosine_similarity(word_to_vec_maps[w], g))
# word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist',
#              'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
# for w in word_list:
#     print(w, cosine_similarity(word_to_vec_maps[w], g))


def neutralize(word, gender, word_to_vec_map):
    """
    projecting "word" to the orthogonal space of bias space, eliminate the bias of "word"
    :param word: string of word
    :param gender: aixs of bias, shape of (50, )
    :param word_to_vec_map: dictionary, maps from words to GloVe vector
    :return:
        e_debiased -- vector without bias
    """
    e = word_to_vec_map[word]
    e_bias_component = (np.dot(e, gender) / np.sum(gender ** 2)) * gender
    e_debiased = e - e_bias_component
    return e_debiased


# example = "receptionist"
# print("Before neutralizing, the cosine similarity between \"{0}\" and g is：{1}"
#       .format(example, cosine_similarity(word_to_vec_maps["receptionist"], g)))
#
# ex_debiased = neutralize("receptionist", g, word_to_vec_maps)
# print("After neutralizing, the cosine similarity between \"{0}\" and g is：{1}"
#       .format(example, cosine_similarity(ex_debiased, g)))


def equalize(pair, bias_axis, word_to_vec_map):
    """
        Debias gender specific words by following the equalize method described in the figure above.

        Arguments:
        pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
        bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
        word_to_vec_map -- dictionary mapping words to their corresponding vectors

        Returns
        e_1 -- word vector corresponding to the first word
        e_2 -- word vector corresponding to the second word
    """
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    u = (e_w1 + e_w2) / 2
    u_b = (np.dot(u, bias_axis) / np.sum(bias_axis ** 2)) * bias_axis
    u_ort = u - u_b

    e_w1_b = (np.dot(e_w1, bias_axis) / np.sum(bias_axis ** 2)) * bias_axis
    e_w2_b = (np.dot(e_w2, bias_axis) / np.sum(bias_axis ** 2)) * bias_axis

    e_w1_b_corrected = np.sqrt(np.abs(1 - np.sum(u_ort ** 2))) * ((e_w1_b - u_b) / np.linalg.norm(e_w1_b - u_b))
    e_w2_b_corrected = np.sqrt(np.abs(1 - np.sum(u_ort ** 2))) * ((e_w2_b - u_b) / np.linalg.norm(e_w2_b - u_b))

    e_w1 = e_w1_b_corrected + u_ort
    e_w2 = e_w2_b_corrected + u_ort

    return e_w1, e_w2


# print("==========before equalizing==========")
# print("cosine_similarity(word_to_vec_maps[\"man\"], gender) = ", cosine_similarity(word_to_vec_maps["man"], g))
# print("cosine_similarity(word_to_vec_maps[\"woman\"], gender) = ", cosine_similarity(word_to_vec_maps["woman"], g))
# print("cosine_similarity(word_to_vec_maps[\"actor\"], gender) = ", cosine_similarity(word_to_vec_maps["actor"], g))
# print("cosine_similarity(word_to_vec_maps[\"actress\"], gender) = ",
# cosine_similarity(word_to_vec_maps["actress"], g))
# e1, e2 = equalize(("man", "woman"), g, word_to_vec_maps)
# e3, e4 = equalize(("actor", "actress"), g, word_to_vec_maps)
# print("\n==========after equalizing==========")
# print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
# print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
# print("cosine_similarity(e3, gender) = ", cosine_similarity(e3, g))
# print("cosine_similarity(e4, gender) = ", cosine_similarity(e4, g))

