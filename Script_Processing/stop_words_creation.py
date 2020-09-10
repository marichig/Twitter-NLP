from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
from nltk.corpus import stopwords as nltk_stop_words
import string
from Script_Processing.stem_and_lemmatize import StemLemmaWrapper
import pickle

sl_wrapper = StemLemmaWrapper()

def create_and_get_stop_words(save = False):
    try:
        base_stop_words = pickle.load( open("../stop_words_dict.pkl", "rb"))
        return base_stop_words
    except:
        pass
    base_stop_words = []
    base_stop_words += nltk_stop_words.words("english")
    base_stop_words += spacy_stop_words


    contractions_file = open('../contractions.txt', 'r')
    contractions = contractions_file.readlines()
    contractions_file.close()
    contractions = [contr.replace("\n","") for contr in contractions]

    base_stop_words += contractions
    base_stop_words.sort()

    stem_stop = [sl_wrapper.stem(x) for x in base_stop_words]
    lemma_stop = [sl_wrapper.lemmatize(x) for x in base_stop_words]
    lemma_stem_stop = [sl_wrapper.stem(x) for x in lemma_stop]
    stop_words_dict = {"clean":base_stop_words,
                "stem": stem_stop,
                "lemma": lemma_stop,
                "lemma_stem":lemma_stem_stop
               }

    if save:
        pickle.dump(stop_words_dict, open("../stop_words_dict.pkl", "wb"))
    return stop_words_dict

def create_and_get_domain_stop_words(domain_stop_words, save = False):
    try:
        base_stop_words = pickle.load( open("../stop_words_dict.pkl", "rb"))
    except:
        base_stop_words = create_and_get_stop_words(save = save)
        
    domain_base_stop_words = set(base_stop_words["clean"]+domain_stop_words)

    domain_stem_stop = [sl_wrapper.stem(x) for x in domain_base_stop_words]
    domain_lemma_stop = [sl_wrapper.lemmatize(x) for x in domain_base_stop_words]
    domain_lemma_stem_stop = [sl_wrapper.stem(x) for x in domain_lemma_stop]

    domain_stop_words_dict = {"clean":domain_base_stop_words,
                "stem": domain_stem_stop,
                "lemma": domain_lemma_stop,
                "lemma_stem": domain_lemma_stem_stop
               }

    if save:
        pickle.dump(domain_stop_words_dict, open("../domain_stop_words_dict.pkl", "wb"))
    
    return domain_stop_words_dict