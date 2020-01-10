"""
Takes document as input and performs chunking accoring to a pattern to return a list of Candidate Keywords.

Input :
    document - text file
Output :
    list of possible keyphrases

"""
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize
from nltk.chunk import RegexpParser


def extract_candidate_keywords(document):

    #Get the words in the document
    words = word_tokenize(document)

    # Chunk first to get 'Candidate Keywords'
    tagged = nltk.pos_tag(words)
    chunkGram = r""" PHRASE: 
                        {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}
                """

    chunkParser = RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)

    candidate_keywords = []
    for tree in chunked.subtrees():
        if tree.label() == 'PHRASE':
            candidate_keyword = ' '.join([x for x,y in tree.leaves()])
            candidate_keywords.append(candidate_keyword)

    candidate_keywords = [w for w in candidate_keywords if len(w) > 3 and  len(w.split(' ')) < 6]
    #print("Data XYZ:",candidate_keywords) 
    return candidate_keywords