from gensim.models import CoherenceModel
from gensim import corpora
from gensim.models.phrases import Phrases, Phraser
import pyLDAvis.gensim
from gensim.models import LdaModel
import pickle

class LDA_Modeler:
    def __init__(self, corpus, stop_words_dict = None, text_mode = "clean"):
        self.corpus = corpus
        self.stop_words_dict = stop_words_dict
        self.bag_of_texts = None
        self.dictionary = None
        self.doc_term_matrix = None
        self.lda_model = None
        self.text_mode = text_mode
        

    def perform_topic_modeling(self, num_topics):
        '''
        :param corpus: list of texts, in the associated text_mode
        :param num_topics: assumed number of topics to extract
        :param text_mode: one of "clean", "lemma", "stem", "lemma_stem"
        :param stop_words: list of stop words, in the associated text_mode
        '''
        
        self.bag_of_texts = []
        for doc in self.corpus:
            self.bag_of_texts.append([word for word in doc.split()
            if (word not in self.stop_words_dict[self.text_mode])])
        self.build_doc_term_matrix(self.bag_of_texts)
        self.lda_model = self.get_LDA_model(num_topics)

        self.print_LDA_topics(self.lda_model, num_topics)

        return self.lda_model


    def build_doc_term_matrix(self, bag_of_texts):
        if not self.doc_term_matrix:
            if len(bag_of_texts) == 0:
                return None
            phrases = Phrases(bag_of_texts, min_count=1)
            bigram = Phraser(phrases)
            data_words_bigrams = [bigram[doc] for doc in bag_of_texts]
            self.dictionary = corpora.Dictionary(data_words_bigrams)
            self.doc_term_matrix = [self.dictionary.doc2bow(text) for text in data_words_bigrams]
            
        return self.doc_term_matrix

    def get_LDA_model(self, num_topics):
        return LdaModel(corpus=self.doc_term_matrix,
                             id2word=self.dictionary,
                             num_topics=num_topics,
                             random_state=42,
                             update_every=1,
                             chunksize=2000,
                             passes=1,
                             alpha='auto',
                             per_word_topics=True)

    def print_LDA_topics(self, topic_model, num_topics, num_words = 10):
        x = topic_model.show_topics(num_topics=num_topics, num_words= num_words,formatted=False)
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

        #Below Code Prints Topics and Words
        for topic,words in topics_words:
            print("Topic", str(topic +1)+ ": "+ " ".join(words))
        print()
