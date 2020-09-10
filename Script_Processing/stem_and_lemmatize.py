from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import SnowballStemmer

class StemLemmaWrapper:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer('english')

    def stem(self, sentence):
        tokens = word_tokenize(sentence)
        result = []
        for token in tokens:
            result.append(self.stemmer.stem(token))
        return " ".join(result)

    def get_wordnet_pos(self, tag):
        '''
        Thanks to
        https://stackoverflow.com/questions/15586721/
        wordnet-lemmatization-and-pos-tagging-in-python.

        Wordnet uses Treebank POS tags, found:
        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        '''
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    def lemmatize(self, sentence):
        tokens = word_tokenize(sentence)
        result = []
        tagged = pos_tag(tokens)
        for token, tag in tagged:
            pos = self.get_wordnet_pos(tag)
            if pos == "":
                result.append(token)
            else:
                result.append(self.lemmatizer.lemmatize(token, pos))
        return " ".join(result)

    def lemmatize_then_stem(self, sentence):
        tokens = word_tokenize(sentence)
        result = []
        tagged = pos_tag(tokens)
        for token, tag in tagged:
            pos = self.get_wordnet_pos(tag)
            if pos == "":
                result.append(self.stemmer.stem(token))
            else:
                lemmatized_token = self.lemmatizer.lemmatize(token, pos)
                result.append(self.stemmer.stem(lemmatized_token))
        return " ".join(result)
