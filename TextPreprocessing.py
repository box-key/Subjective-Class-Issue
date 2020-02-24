# Preprocess texts:
# Removing lower text, remove digits, remove stopwords, lemmatize, punctuations
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re, string

def fillEmpty(sentence):
    if not isinstance(sentence, str):
        return 'N/A'
    else:
        return sentence

def removePunctuation(sentence):
    return sentence.translate(str.maketrans(' ', ' ',string.punctuation))

def removeDigit(sentence):
    digit_removed = re.sub(r'\d+', '', sentence)
    return digit_removed

nltk.download('stopwords')
def removeStopWords(sentence):
    stop_words = set(stopwords.words('english'))
    return list(filter(lambda x: x not in stop_words, sentence.split()))

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def lemmatizeDoc(sentence):
    return list(map(lambda x: lemmatizer.lemmatize(x), sentence.split()))

def convertWordsToString(word_list):
    sentence = ' '.join(word_list)
    return sentence

def clean_corpora(doc_list, isLower=True, doesRemoveDigit=True, doesRemovePunc=True, doesRemoveStopWords=True, doesLemmatize=True):
    # Fill out missing cells
    doc_list = doc_list.apply(fillEmpty)
    # Lower all texts
    if isLower:
        doc_list = doc_list.str.lower()
    # Remove all digits
    if doesRemoveDigit:
        doc_list = doc_list.apply(removeDigit)
    # Remove punctuations
    if doesRemovePunc:
        doc_list = doc_list.apply(removePunctuation)
    # Remove Stopwords
    if doesRemoveStopWords:
        doc_list = doc_list.apply(removeStopWords)
        doc_list = doc_list.apply(convertWordsToString)
    # Lemmatize text
    if doesLemmatize:
        doc_list = doc_list.apply(lemmatizeDoc)
        doc_list = doc_list.apply(convertWordsToString)
    return doc_list
