import re
import contractions

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# load stopwords default nltk list
stop_words = stopwords.words('english')

def normalize_document(doc):
    """Normalize the document by performing basic text pre-processing tasks."""
    # remove special characters
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    # remove trailing whitespace
    nowhite = doc.strip()
    # expand contractions
    expanded = contractions.fix(nowhite)
    # tokenize document
    tokens = word_tokenize(expanded)
    # remove stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from tokens
    doc = ' '.join(filtered_tokens)

    return doc


def simple_text_preprocessor(document):
    """Perform basic text pre-processing tasks."""
    # load up a simple porter stemmer - nothing fancy
    ps = PorterStemmer()

    # lower case
    document = str(document).lower()

    # expand contractions
    document = contractions.fix(document)

    # remove unnecessary characters
    document = re.sub(r'[^a-zA-Z]',r' ', document)
    document = re.sub(r'nbsp', r'', document)
    document = re.sub(' +', ' ', document)

    # simple porter stemming
    document = ' '.join([ps.stem(word) for word in document.split()])

    # stopwords removal
    document = ' '.join([word for word in document.split() if word not in stop_words])

    return document
