from collections import Counter, defaultdict
import math
import nltk

def load_counts(word_list, min_count=0, stopwords=set(), class_name="class"):
    result = defaultdict(int)
    word_counts = Counter(word_list)

    for word, count in word_counts.items():
        if count >= min_count and word not in stopwords and len(word)>1:
            result[word] = count

    print('# of keys in', class_name, len(result.keys()))
    return result

def compute_log_odds(counts1, counts2, prior):
    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in prior.keys(): prior[word] = int(prior[word] + 0.5)

    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0: prior[word] = 1
    # end for

    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0: prior[word] = 1
    # end for

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())
    nprior = sum(prior.values())

    for word in prior.keys():
        if prior[word] > 0:
            l1 = float(counts1[word] + prior[word]) / ((n1 + nprior) - (counts1[word] + prior[word]))
            l2 = float(counts2[word] + prior[word]) / ((n2 + nprior) - (counts2[word] + prior[word]))
            sigmasquared[word] = 1 / (float(counts1[word]) + float(prior[word])) + 1 / (
                    float(counts2[word]) + float(prior[word]))
            sigma[word] = math.sqrt(sigmasquared[word])
            delta[word] = (math.log(l1) - math.log(l2)) / sigma[word]
    # end if
    # end for
    return delta


def calc_log_odds_on_word_list(class_a_words, class_b_words, num_of_top_words, class_a_name="class_a",
                               class_b_name="class_b"):
    class_a_counts = load_counts(class_a_words,min_count=30, class_name=class_a_name)
    class_b_counts = load_counts(class_b_words,min_count=30, class_name=class_b_name)
    prior = class_a_words + class_b_words
    prior_counts = load_counts(prior, min_count=10, class_name="prior")
    log_odds = compute_log_odds(class_a_counts, class_b_counts, prior_counts)
    for word, log_odd in sorted(log_odds.items(), key=lambda x: x[1]):
        print("{}\t{:.3f}\n".format(word, log_odd))

    sorted_log_odds = sorted(log_odds.items(), key=lambda x: x[1])
    N = num_of_top_words
    words = [x[0] for x in sorted_log_odds]
    top_words_a = words[-N:]
    top_words_b = words[:N]

    print(f'top words that are most associated with {class_a_name}: ')
    for item in top_words_a:
        print(item)

    print(f'top words that are most associated with {class_b_name}: ')
    for item in top_words_b:
        print(item)


def calc_word_log_odds(class_a_sentences_list, class_b_sentences_list, num_of_top_words=10,
                       normalize_by_num_of_sentences=False, class_a_name="class_A", class_b_name="class_B"):
    class_a = class_a_sentences_list  # List of sentences in class A
    class_b = class_b_sentences_list  # List of sentences in class B

    # Tokenize Hebrew sentences into words
    tokenizer = nltk.RegexpTokenizer(r'\w+|[^\w\s]+')
    class_a_words = [word for sentence in class_a for word in tokenizer.tokenize(sentence)]
    class_b_words = [word for sentence in class_b for word in tokenizer.tokenize(sentence)]

    # Calculate word counts for each class
    class_a_counts = Counter(class_a_words)
    class_b_counts = Counter(class_b_words)

    # Calculate number of sentences in each class
    num_sentences_a = len(class_a)
    num_sentences_b = len(class_b)

    # Calculate total number of words in each class
    total_a = sum(class_a_counts.values())
    total_b = sum(class_b_counts.values())
    if normalize_by_num_of_sentences:
        normalized_factor_a = num_sentences_a
        normalized_factor_b = num_sentences_b
    else:
        normalized_factor_a = total_a
        normalized_factor_b = total_b
    # Calculate log odds ratio for each word
    log_odds = []
    for word in class_a_counts:
        count_a = class_a_counts[word]
        count_b = class_b_counts[word]
        log_odds.append((word, math.log((count_a / normalized_factor_a) / (count_b + 1 / normalized_factor_b + 1))))

    # Sort log odds list in descending order
    log_odds.sort(key=lambda x: x[1], reverse=True)

    # Return top N words with highest log odds ratio
    N = num_of_top_words  # Change this to the desired number of top words
    top_words = log_odds[:N]

    print(f'top words that are most associated with {class_a_name}: {top_words}')
    print(f'top words that are most associated with {class_a_name}: ')
    only_words_top_words = []
    for item in top_words:
        only_words_top_words.append(item[0])
        print(item[0])

    # Sort log odds list in ascending order
    log_odds.sort(key=lambda x: x[1])

    # Return top N words with lowest log odds ratio (i.e., most associated with class B)
    top_words = log_odds[:N]

    print(f'top words that are most associated with {class_b_name}: {top_words}')
    print(f'top words that are most associated with {class_b_name}: ')
    only_words_top_words = []
    for item in top_words:
        only_words_top_words.append(item[0])
        print(item[0])