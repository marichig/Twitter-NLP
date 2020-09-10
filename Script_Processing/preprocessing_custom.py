from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import demoji
import html, re
import string
# Demoji package - https://pypi.org/project/demoji/

if demoji.last_downloaded_timestamp() is None:
    demoji.download_codes() # one-time run

urlPattern = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+))')
MENTION_PATTERN = re.compile(r'@\w*')
HASHTAG_PATTERN = re.compile(r'#\w*')
non_alphanumeric_or_twitter_characters = re.compile(r'[^a-zA-Z0-9\s|#|@|\'|\.|’|\"|”]')
sentence_ending_punctuation = [".", "!", "?"]
#weather_channels = ["weatherchannel", "weather channel", "accuweather"]

def remove_non_ascii(s):
    #thanks to https://stackoverflow.com/questions/1342000/
    #how-to-make-the-python-interpreter-correctly-handle-non-ascii-characters-in-stri
    return "".join(i for i in s if ord(i)<128)

#def remove_weather_channels(text):
#    t = text.lower()
#    for channel in weather_channels:
#        if channel in t:
#            t = t.replace(channel, "")
#    return t

def remove_URL(text):
    return re.sub(urlPattern," ",text).strip() #strips leading/trailing whitespace

def remove_mentions(text): # new function for preprocessor replacement
#    text = text.replace("@realdonaldtrump", "trump")
    return re.sub(MENTION_PATTERN, " ", text)

def clean_text(text):
    '''
    Removes mentions, non alphanumeric/twitter characters, and emojis
    '''
    t = text.lower()
    t = remove_mentions(t)
    t = re.sub(non_alphanumeric_or_twitter_characters," ",t)
    return demoji.replace(t) # remove emojis with " "

def clean_html(text):
    t = text.encode('ascii', 'ignore')
    t = html.unescape(text)
    t = t.replace('\n'," ")
    t = t.replace('&', 'and')
    t = remove_non_ascii(t)
    t = " ".join(t.split())
    return t

def remove_hashtagged_words(text):
    return re.sub(HASHTAG_PATTERN, " ", text)

def remove_hashtag_symbol(text):
    return text.replace('#',"")

def preprocess_minimal_clean(text):
    return remove_URL(clean_html(text))

def preprocess_basic_clean(text):
    return clean_text(remove_URL(clean_html(text)))

def preprocess_keep_hashtag_text(text):
    return remove_hashtag_symbol(clean_text(remove_URL(clean_html(text))))

def preprocess_remove_hashtag_text(text):
    return remove_hashtagged_words(clean_text(remove_URL(clean_html(text))))

def is_hashtag(token):
    return token[0] == "#"

def is_end_of_sentence(token):
    return token[-1] in sentence_ending_punctuation

def first_hashtag_to_discard(reversed_iter):
    try:
        this_tok = next(reversed_iter)
        if not is_hashtag(this_tok):
            return None

        next_tok = next(reversed_iter)
        if not (is_hashtag(next_tok) or is_end_of_sentence(next_tok)): #i.e. if is word
            return None
        elif is_end_of_sentence(next_tok):
            return this_tok
        #from this point on, previous two tokens are #'s'
        last_tok = this_tok
        this_tok = next_tok
        next_tok = next(reversed_iter)
        while(True): #Python does not have a .has_next() function for iterators...
            if is_end_of_sentence(next_tok):
                return this_tok
            if not is_hashtag(next_tok):
                return last_tok

            last_tok = this_tok
            this_tok = next_tok
            next_tok = next(reversed_iter)

    except StopIteration:
        pass

def remove_trailing_hashtags(text):
    toks = text.split()
    reversed_iter = iter(reversed(toks))
    first_trailing_hashtag = first_hashtag_to_discard(reversed_iter)
    if first_trailing_hashtag is None:
        return text
    else:
        return " ".join(toks[:toks.index(first_trailing_hashtag)])

def bio_preprocess(text):
    if text is not None:
        s = remove_hashtag_symbol(
        clean_text(
        remove_URL(
        clean_html(text))))
        s = s.translate(str.maketrans('.', ' ', string.punctuation.replace('.','')))
        #clear punctuation, replaces '.' with ' ' to handle ellipses
        return " ".join(s.split())
        #strip leading, in-between, and trailing whitespace
    else:
        return None

def full_preprocess(text):
    if text is not None:
        s = remove_hashtag_symbol(
        remove_trailing_hashtags(
        clean_text(
#        remove_weather_channels(
        remove_URL(
        clean_html(text)))))#)

        s = s.translate(str.maketrans('.', ' ', string.punctuation.replace('.','')))
        #clear punctuation, replaces '.' with ' ' to handle ellipses
        return " ".join(s.split())
        #strip leading, in-between, and trailing whitespace
    else:
        return ""
