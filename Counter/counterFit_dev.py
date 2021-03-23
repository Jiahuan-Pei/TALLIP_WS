# encoding=UTF-8
"""
    @author: ppsunrise
    @email: p_sunrise@mail.dlut.edu.cn
    @function: counter-fitting调参
"""
from __future__ import division
from gensim.models import KeyedVectors
from Com import macro, utils
from Single.linearcombination import *
import codecs
import numpy
import random
import math
from copy import deepcopy
from numpy.linalg import norm
from numpy import dot
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
# import seaborn as sns


class ExperimentRun:
    """
    This class stores all of the data and hyperparameters required for a counterfitting run.
    """

    def __init__(self, hypers):
        """
        To initialise the class, we need to supply the config file, which contains the location of
        the pretrained word vectors, of the vocabulary to use, the location of (potentially many)
        collections of linguistic constraints (one pair per line), the location of the dialogue
        domain ontology to inject (optional, needs to respect DSTC format), as well as the six
        hyperparameters of the counterfitting procedure (as detailed in the NAACL paper).
        """
        pretrained_vectors_filepath = r"%s/fml_org_bdnews_xieso_w2v.bin" % macro.DICT_DIR
        # vocabulary_filepath = r"D:\MyData\NLPCC_DATA\corpus\vocab.txt"
        # eval_fname = 'test_4.csv'   # 评价文件名
        self.eval_fname = macro.NLPCC_FML_FILE  # 评价文件名
        # eval_fname = 'MC30.txt'
        # eval_fname = 'SemEval50.csv'
        self.read2wordlist = utils.read2wordlist([(macro.CORPUS_DIR, self.eval_fname)], 'tag')
        self.id_list, self.word1_list, self.word2_list, self.manu_sim_list, self.headline = self.read2wordlist
        # load list of filenames for synonyms and antonyms.
        # 同义和反义词典(固定的)
        antonym_list = [r'%s/merged.txt' % macro.DICT_DIR]
        synonym_list = [r'%s/tongyis.txt' % macro.DICT_DIR]
        # 根据single_sim们计算的词对（可变的）
        antonym_value_list = [r'%s/fanyi_value.txt' % macro.DICT_DIR]
        synonym_value_list = [r'%s/tongyi_value.txt' % macro.DICT_DIR]
        df = single_sims([(macro.CORPUS_DIR, self.eval_fname)], 'PKU500_single_sims')
        linear_max_auto_sims = [row[4:].max() for row in df.values]
        fw1 = codecs.open(antonym_value_list[0], 'w')
        fw2 = codecs.open(synonym_value_list[0], 'w')
        th = 7.5
        for auto_sim, w1, w2 in zip(linear_max_auto_sims, self.word1_list, self.word2_list):
            if auto_sim < 1 - th:
                fw1.write(w1.encode('utf-8') + '\t' + w2.encode('utf-8') + '\n')
            elif auto_sim > th:
                fw2.write(w1.encode('utf-8') + '\t' + w2.encode('utf-8') + '\n')
        fw1.close()
        fw2.close()
        # vocabulary = []
        # with open(vocabulary_filepath, "r+") as f_in:
        #     for line in f_in:
        #         vocabulary.append(line.strip())
        #
        # vocabulary = set(vocabulary)
        word_list = []

        with open('%s/%s' % (macro.CORPUS_DIR, self.eval_fname), 'r') as fr:
            for line in fr.readlines()[1:]:
                id, w1, w2, score = line.strip().split('\t')
                word_list.append(w1)
                word_list.append(w2)
        vocabulary = sorted(set(word_list))


        # load pretrained word vectors and initialise their (restricted) vocabulary.
        self.pretrained_word_vectors = load_word_vectors(pretrained_vectors_filepath, vocabulary)

        # if no vectors were loaded, exit gracefully:
        if not self.pretrained_word_vectors:
            return

        self.vocabulary = set(self.pretrained_word_vectors.keys())

        self.synonyms = set()
        self.antonyms = set()
        self.synonyms_value = set()
        self.antonyms_value = set()

        # and we then have all the information to collect all the linguistic constraints:
        for syn_filepath in synonym_list:
            self.synonyms = self.synonyms | load_constraints(syn_filepath, self.vocabulary)

        for ant_filepath in antonym_list:
            self.antonyms = self.antonyms | load_constraints(ant_filepath, self.vocabulary)

        for syn_v_filepath in synonym_value_list:
            self.synonyms_value = self.synonyms_value | load_constraints(syn_v_filepath, self.vocabulary)

        for ant_v_filepath in antonym_value_list:
            self.antonyms_value = self.antonyms_value | load_constraints(ant_v_filepath, self.vocabulary)

        # finally, load the experiment hyperparameters:
        self.hyper_k1, self.hyper_k2, self.hyper_k3, self.hyper_k4, self.hyper_k5,\
        self.delta, self.gamma, self.rho, self.theta, self.eta = hypers


def load_word_vectors(file_destination, vocabulary):
    """
    This method loads the word vectors from the supplied file destination.
    It loads the dictionary of word vectors and prints its size and the vector dimensionality.
    """
    print "Loading pretrained word vectors from", file_destination
    word_dictionary = {}
    try:
        w2v_model = KeyedVectors.load_word2vec_format(file_destination, binary=True)
        for v in vocabulary:
            try:

                word_dictionary[v] = w2v_model[v.decode('utf-8')]
            except:
                print 'Cannot find the vector of', v, 'in pretreained word vectoer model'
    except:
        print "Word vectors could not be loaded from:", file_destination
        return {}
    # try:
    #     with open(file_destination, "r") as f:
    #         for line in f:
    #             line = line.split(" ", 1)
    #             key = line[0].lower()
    #             if key in vocabulary:
    #                 word_dictionary[key] = numpy.fromstring(line[1], dtype="float32", sep=" ")
    # except:
    #     print "Word vectors could not be loaded from:", file_destination
    #     if file_destination == "word_vectors/glove.txt" or file_destination == "word_vectors/paragram.txt":
    #         print "Please unzip the provided glove/paragram vectors in the word_vectors directory.\n"
    #     return {}

    print len(word_dictionary), "vectors loaded from", file_destination
    return normalise_word_vectors(word_dictionary)


def print_word_vectors(word_vectors, write_path):
    """
    This function prints the collection of word vectors to file, in a plain textual format.
    """
    print "Saving the counter-fitted word vectors to", write_path, "\n"
    with open(write_path, "wb") as f_write:
        for key in word_vectors:
            print >> f_write, key, " ".join(map(str, numpy.round(word_vectors[key], decimals=6)))


def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word] ** 2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors


def load_constraints(constraints_filepath, vocabulary):
    """
    This methods reads a collection of constraints from the specified file, and returns a set with
    all constraints for which both of their constituent words are in the specified vocabulary.
    """
    constraints_filepath.strip()
    constraints = set()
    with open(constraints_filepath, "r+") as f:
        for line in f:
            word_pair = line.split()
            if word_pair[0] in vocabulary and word_pair[1] in vocabulary and word_pair[0] != word_pair[1]:
                constraints |= {(word_pair[0], word_pair[1])}
                constraints |= {(word_pair[1], word_pair[0])}

    print constraints_filepath, "yielded", len(constraints), "constraints."

    return constraints


def extract_antonyms_from_dialogue_ontology(dialogue_ontology, vocabulary):
    """
    Returns a list of antonyms for the supplied dialogue ontology, which needs to be provided as a dictionary.
    The dialogue ontology must follow the DST Challenges format: we only care about goal slots, i.e. informables.
    """
    # We are only interested in the goal slots of the ontology:
    dialogue_ontology = dialogue_ontology["informable"]

    slot_names = set(dialogue_ontology.keys())

    # Forcing antonymous relations between different entity names does not make much sense.
    if "name" in slot_names:
        slot_names.remove("name")

    # Binary slots - we do not know how to handle - there is no point enforcing antonymy relations there.
    binary_slots = set()
    for slot_name in slot_names:
        current_values = dialogue_ontology[slot_name]
        if len(current_values) == 2 and "true" in current_values and "false" in current_values:
            binary_slots |= {slot_name}

    if binary_slots:
        print "Removing binary slots:", binary_slots
    else:
        print "There are no binary slots to ignore."

    slot_names = slot_names - binary_slots

    antonym_list = set()

    # add antonymy relations between each pair of slot values for each non-binary slot.
    for slot_name in slot_names:
        current_values = dialogue_ontology[slot_name]
        for index_1, value in enumerate(current_values):
            for index_2 in range(index_1 + 1, len(current_values)):
                # note that this will ignore all multi-value words.
                if value in vocabulary and current_values[index_2] in vocabulary:
                    antonym_list |= {(value, current_values[index_2])}
                    antonym_list |= {(current_values[index_2], value)}

    return antonym_list


def distance(v1, v2, normalised_vectors=True):
    """
    Returns the cosine distance between two vectors.
    If the vectors are normalised, there is no need for the denominator, which is always one.
    """
    if normalised_vectors:
        return (1 - dot(v1, v2))/2.0
    else:
        return (1 - dot(v1, v2) / (norm(v1) * norm(v2)))/2.0



def compute_vsp_pairs(word_vectors, vocabulary, rho=0.2):
    """
    This method returns a dictionary with all word pairs which are closer together than rho.
    Each pair maps to the original distance in the vector space.

    In order to manage memory, this method computes dot-products of different subsets of word
    vectors and then reconstructs the indices of the word vectors that are deemed to be similar.
    """
    print "Pre-computing word pairs relevant for Vector Space Preservation (VSP). Rho =", rho

    vsp_pairs = {}

    threshold = 1 - rho
    vocabulary = list(vocabulary)
    num_words = len(vocabulary)

    step_size = 1000  # Number of word vectors to consider at each iteration.
    vector_size = random.choice(word_vectors.values()).shape[0]

    # ranges of word vector indices to consider:
    list_of_ranges = []

    left_range_limit = 0
    while left_range_limit < num_words:
        curr_range = (left_range_limit, min(num_words, left_range_limit + step_size))
        list_of_ranges.append(curr_range)
        left_range_limit += step_size

    range_count = len(list_of_ranges)

    # now compute similarities between words in each word range:
    for left_range in range(range_count):
        for right_range in range(left_range, range_count):

            # offsets of the current word ranges:
            left_translation = list_of_ranges[left_range][0]
            right_translation = list_of_ranges[right_range][0]

            # copy the word vectors of the current word ranges:
            vectors_left = numpy.zeros((step_size, vector_size), dtype="float32")
            vectors_right = numpy.zeros((step_size, vector_size), dtype="float32")

            # two iterations as the two ranges need not be same length (implicit zero-padding):
            full_left_range = range(list_of_ranges[left_range][0], list_of_ranges[left_range][1])
            full_right_range = range(list_of_ranges[right_range][0], list_of_ranges[right_range][1])

            for iter_idx in full_left_range:
                vectors_left[iter_idx - left_translation, :] = word_vectors[vocabulary[iter_idx]]

            for iter_idx in full_right_range:
                vectors_right[iter_idx - right_translation, :] = word_vectors[vocabulary[iter_idx]]

            # now compute the correlations between the two sets of word vectors:
            dot_product = vectors_left.dot(vectors_right.T)

            # find the indices of those word pairs whose dot product is above the threshold:
            indices = numpy.where(dot_product >= threshold)

            num_pairs = indices[0].shape[0]
            left_indices = indices[0]
            right_indices = indices[1]

            for iter_idx in range(0, num_pairs):

                left_word = vocabulary[left_translation + left_indices[iter_idx]]
                right_word = vocabulary[right_translation + right_indices[iter_idx]]

                if left_word != right_word:
                    # reconstruct the cosine distance and add word pair (both permutations):
                    score = 1 - dot_product[left_indices[iter_idx], right_indices[iter_idx]]
                    vsp_pairs[(left_word, right_word)] = score
                    vsp_pairs[(right_word, left_word)] = score

    # print "There are", len(vsp_pairs), "VSP relations to enforce for rho =", rho, "\n"
    return vsp_pairs


def vector_partial_gradient(u, v, normalised_vectors=True):
    """
    This function returns the gradient of cosine distance: \frac{ \partial dist(u,v)}{ \partial u}
    If they are both of norm 1 (we do full batch and we renormalise at every step), we can save some time.
    """

    if normalised_vectors:
        gradient = u * dot(u, v) - v
    else:
        norm_u = norm(u)
        norm_v = norm(v)
        nominator = u * dot(u, v) - v * numpy.power(norm_u, 2)
        denominator = norm_v * numpy.power(norm_u, 3)
        gradient = nominator / denominator

    return gradient


def one_step_SGD(word_vectors, synonym_pairs, antonym_pairs, vsp_pairs, current_experiment, synonyms_value_pairs = None,antonyms_value_pairs = None):
    """
    This method performs a step of SGD to optimise the counterfitting cost function.
    """
    new_word_vectors = deepcopy(word_vectors)

    gradient_updates = {}
    update_count = {}
    oa_updates = {}
    vsp_updates = {}

    # AR term:
    for (word_i, word_j) in antonym_pairs:

        current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

        if current_distance < current_experiment.delta:

            gradient = vector_partial_gradient(new_word_vectors[word_i], new_word_vectors[word_j])
            gradient = gradient * current_experiment.hyper_k1

            if word_i in gradient_updates:
                gradient_updates[word_i] += gradient
                update_count[word_i] += 1
            else:
                gradient_updates[word_i] = gradient
                update_count[word_i] = 1

    # SA term:
    for (word_i, word_j) in synonym_pairs:

        current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

        if current_distance > current_experiment.gamma:

            gradient = vector_partial_gradient(new_word_vectors[word_j], new_word_vectors[word_i])
            gradient = gradient * current_experiment.hyper_k2

            if word_j in gradient_updates:
                gradient_updates[word_j] -= gradient
                update_count[word_j] += 1
            else:
                gradient_updates[word_j] = -gradient
                update_count[word_j] = 1

    # VSP term:
    for (word_i, word_j) in vsp_pairs:

        original_distance = vsp_pairs[(word_i, word_j)]
        new_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

        if original_distance <= new_distance:

            gradient = vector_partial_gradient(new_word_vectors[word_i], new_word_vectors[word_j])
            gradient = gradient * current_experiment.hyper_k3

            if word_i in gradient_updates:
                gradient_updates[word_i] -= gradient
                update_count[word_i] += 1
            else:
                gradient_updates[word_i] = -gradient
                update_count[word_i] = 1

    # SA_V term:
    if synonyms_value_pairs:
        for (word_i, word_j) in synonyms_value_pairs:

            current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

            if current_distance > current_experiment.theta:

                gradient = vector_partial_gradient(new_word_vectors[word_j], new_word_vectors[word_i])
                gradient = gradient * current_experiment.hyper_k4

                if word_j in gradient_updates:
                    gradient_updates[word_j] -= gradient
                    update_count[word_j] += 1
                else:
                    gradient_updates[word_j] = -gradient
                    update_count[word_j] = 1

    #AR_V term:
    if antonyms_value_pairs:
        for (word_i, word_j) in antonym_pairs:

            current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

            if current_distance < current_experiment.eta:

                gradient = vector_partial_gradient(new_word_vectors[word_i], new_word_vectors[word_j])
                gradient = gradient * current_experiment.hyper_k5

                if word_i in gradient_updates:
                    gradient_updates[word_i] += gradient
                    update_count[word_i] += 1
                else:
                    gradient_updates[word_i] = gradient
                    update_count[word_i] = 1

    for word in gradient_updates:
        # we've found that scaling the update term for each word helps with convergence speed.
        update_term = gradient_updates[word] / (update_count[word])
        new_word_vectors[word] += update_term

    return normalise_word_vectors(new_word_vectors)


def counter_fit(current_experiment):
    """
    This method repeatedly applies SGD steps to counter-fit word vectors to linguistic constraints.
    """
    word_vectors = current_experiment.pretrained_word_vectors
    vocabulary = current_experiment.vocabulary
    antonyms = current_experiment.antonyms
    synonyms = current_experiment.synonyms
    synonyms_value = current_experiment.synonyms_value
    antonyms_value = current_experiment.antonyms_value
    idl, w1l, w2l, score, headline = current_experiment.read2wordlist
    current_iteration = 0

    vsp_pairs = {}

    if current_experiment.hyper_k3 > 0.0:  # if we need to compute the VSP terms.
        vsp_pairs = compute_vsp_pairs(word_vectors, vocabulary, rho=current_experiment.rho)

    # Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
    for antonym_pair in antonyms:
        if antonym_pair in synonyms:
            synonyms.remove(antonym_pair)
        if antonym_pair in vsp_pairs:
            del vsp_pairs[antonym_pair]

    max_iter = 20
    print "\nAntonym pairs:", len(antonyms), "Synonym pairs:", len(synonyms), "VSP pairs:", len(vsp_pairs), "Syn_value:", len(synonyms_value)
    print "Running the optimisation procedure for", max_iter, "SGD steps..."
    spearmans = []
    spearmans2 = []
    times = []
    times.append(0)
    word_vectors2 = word_vectors.copy()
    _, _, _, sims, _, _ = analysis(word_vectors, current_experiment.read2wordlist)
    _, _, _, sims2, _, _ = analysis(word_vectors, current_experiment.read2wordlist)

    spearmans.append(spearmanr(score, sims))
    spearmans2.append(spearmanr(score, sims2))
    while current_iteration < max_iter:
        current_iteration += 1
        times.append(current_iteration)
        word_vectors = one_step_SGD(word_vectors, synonyms, antonyms, vsp_pairs, current_experiment, synonyms_value,antonyms_value) # improved
        word_vectors2 = one_step_SGD(word_vectors2, synonyms, antonyms, vsp_pairs, current_experiment)  # basic

        _,_,_,sims, _, _ = analysis(word_vectors, current_experiment.read2wordlist)
        _,_,_,sims2, _, _ = analysis(word_vectors2, current_experiment.read2wordlist)
        spearmans.append(spearmanr(sims, score))
        spearmans2.append(spearmanr(sims2, score))

    dataset = {
        'times':times,
        'rho':spearmans
    }
    # frame = DataFrame(dataset)

    plt.plot(times,spearmans,color='red',label='improved counter-fitting',linewidth=2)
    plt.plot(times,spearmans2,color = 'blue',label='basic counter-fitting',linewidth=2)
    plt.xlabel('iteration times')
    plt.grid(True)
    plt.ylabel("Spearman's rho coefficient")
    plt.xlim([0,21])
    plt.ylim([0.3,0.6])
    plt.legend()
    plt.savefig('%s/%s.pdf' % (macro.PICS_DIR, 'counter-fitting-iteration'))

    return word_vectors


def analysis(word_vectoers, read2wordlist):
    '''
    :param word_vectoers: dict类型，词向量
    :return: id列表，词的列表，以及以word_vectors计算的相似度list
    '''
    idl, w1l, w2l, score, headline = read2wordlist
    auto_sims = []
    print 'len of pairs:', len(idl)

    for w1, w2,s in zip(w1l, w2l, score):
        sim =0
        try:
            sim = (1 - distance(word_vectoers[w1.encode('utf-8')], word_vectoers[w2.encode('utf-8')]))
        except:
            sim = 0
       # print w1,w2,sim
        auto_sims.append(4.5*sim+5.5)
        # print w1,w2,sim,s
    p = spearmanr(auto_sims, score)
    r = pearsonr(auto_sims, score)
    print 'spearmanr:', p, '\tpearson:', r
    return idl,w1l,w2l,auto_sims, p, r



def simlex_analysis(word_vectors):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors.
    The method also prints the gold standard SimLex-999 ranking to results/simlex_ranking.txt,
    and the ranking produced using the counter-fitted vectors to results/counter_ranking.txt
    """
    fread_simlex = open("linguistic_constraints/SimLex-999.txt", "rb")
    pair_list = []

    line_number = 0
    for line in fread_simlex:
        if line_number > 0:
            tokens = line.split()
            word_i = tokens[0]
            word_j = tokens[1]
            score = float(tokens[3])
            if word_i in word_vectors and word_j in word_vectors:
                pair_list.append(((word_i, word_j), score))
        line_number += 1

    pair_list.sort(key=lambda x: - x[1])

    f_out_simlex = open("results/simlex_ranking.txt", "wb")
    f_out_counterfitting = open("results/counter_ranking.txt", "wb")

    extracted_list = []
    extracted_scores = {}

    for (x, y) in pair_list:
        (word_i, word_j) = x
        current_distance = distance(word_vectors[word_i], word_vectors[word_j])
        extracted_scores[(word_i, word_j)] = current_distance
        extracted_list.append(((word_i, word_j), current_distance))

    extracted_list.sort(key=lambda x: x[1])

    # print both the gold standard ranking and the produced ranking to files in the results folder:
    def parse_pair(pair_of_words):
        return str(pair_of_words[0] + ", " + str(pair_of_words[1]))

    for idx, element in enumerate(pair_list):
        clean_elem = str(parse_pair(element[0])) + " : " + str(round(element[1], 2))
        print >> f_out_simlex, idx, ":", clean_elem

    for idx, element in enumerate(extracted_list):
        clean_elem = str(parse_pair(element[0])) + " : " + str(round(element[1], 2))
        print >> f_out_counterfitting, idx, ":", clean_elem

    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)

    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)
    return round(spearman_rho[0], 3)


def run_experiment(hypers):
    """
    This method runs the counterfitting experiment, printing the SimLex-999 score of the initial
    vectors, then counter-fitting them using the supplied linguistic constraints.
    We then print the SimLex-999 score of the final vectors, and save them to a .txt file in the
    results directory.
    """

    current_experiment = ExperimentRun(hypers)
    if not current_experiment.pretrained_word_vectors:
        return

    print "Spearman's rho coefficient of initial vectors is:"
    # simlex_analysis(current_experiment.pretrained_word_vectors), "\n"
    _, _, _, _, pp, rr = analysis(current_experiment.pretrained_word_vectors, current_experiment.read2wordlist)
    transformed_word_vectors = counter_fit(current_experiment)

    #
    print "\nSpearman's rho coefficient of the counter-fitted vectors is:"
    idl,w1l,w2l,sims, p, r = analysis(transformed_word_vectors, current_experiment.read2wordlist)

    # 参数、性能字符串
    hypers_str = ','.join([str(h) for h in hypers]) + ',' + str(pp[0]) + ',' + str(p[0])

    # 写文件
    outfile = codecs.open('%s/hypers/counter-fitting-result(%s).txt' % (macro.RESULTS_DIR, hypers_str),'w','utf-8')
    i=0
    _, __, ___, score, ____ = utils.read2wordlist([(macro.CORPUS_DIR, current_experiment.eval_fname)])
    outfile.write('ID\tWord1\tWord2\tSimilarity\r\n')
    for id,w1,w2,s,ss in zip(idl,w1l,w2l,score,sims):
        outfile.write(id+'\t'+w1+'\t'+w2+'\t'+str(s)+'\t'+str(ss)+'\r\n')

    outfile.close()

    print_word_vectors(current_experiment.pretrained_word_vectors, macro.RESULTS_DIR+r'/dev_org_vectors.txt')
    print_word_vectors(transformed_word_vectors, macro.RESULTS_DIR+r"/dev_counter_fitted_vectors.txt")
    return hypers_str


def train():
    """
    The user can provide the location of the config file as an argument.
    If no location is specified, the default config file (experiment_parameters.cfg) is used.
    """
    k1 = 0.1
    k2 = 0.1
    k3 = 0.1
    k4 = 0.1 # 论文里面的k3
    k5 = 0

    delta = 0.5 # 固定
    gamma = 0   # 固定
    rho = 0.2
    theta = 0 # 相似下界
    eta = 1

    outfile = codecs.open('%s/hypers/counter-fitting-hypers.csv' % macro.RESULTS_DIR, 'w', 'utf-8')

    grid = (numpy.arange(21))/100
    for k4 in grid:
        hypers = [k1, k2, k3, k4, k5, delta, gamma, rho, theta, eta]
        hstr = run_experiment(hypers)
        print hstr
        outfile.write(hstr+'\n')

    # grid = (numpy.arange(41)) / 100
    # grid = (numpy.arange(10)) / 10 # rho
    # for rho in grid:
    #     hypers = [k1, k2, k3, k4, k5, delta, gamma, rho, theta, eta]
    #     hstr = run_experiment(hypers)
    #     print hstr
    #     outfile.write(hstr + '\n')

    outfile.close()


def train2():
    k1 = 0.1
    k2 = 0.1
    k3 = 0.1
    k4 = 0.1 # 论文里面的k3
    k5 = 0

    delta = 0.5 # 固定
    gamma = 0   # 固定
    rho = 0.2
    theta = 0.25 # 相似下界
    eta = 1

    outfile = codecs.open('%s/hypers/counter-fitting-hypers.csv' % macro.RESULTS_DIR, 'w', 'utf-8')

    grid = (numpy.arange(11))/50 # [0,0.2],步长0.02,11^3次
    for k4 in grid:
        for k2 in grid:
            for k1 in grid:
                hypers = [k1, k2, k3, k4, k5, delta, gamma, rho, theta, eta]
                hstr = run_experiment(hypers)
                print hstr
                outfile.write(hstr+'\n')

    outfile.close()
    return


def train3():
    k1, k2, k3, k4, k5 = 0.02, 0.08, 0.1, 0.14, 0.0     # others
    delta, gamma, rho, theta, eta = 0.5, 0.0, 0.2, 0.25, 1

    outfile = codecs.open('%s/hypers/counter-fitting-hypers.csv' % macro.RESULTS_DIR, 'w', 'utf-8')

    grid = (numpy.arange(10))/10
    for rho in grid:
        hypers = [k1, k2, k3, k4, k5, delta, gamma, rho, theta, eta]
        hstr = run_experiment(hypers)
        print hstr
        outfile.write(hstr+'\n')

    outfile.close()
    return

if __name__ == '__main__':
    train()
    # train2()
    # train3()
