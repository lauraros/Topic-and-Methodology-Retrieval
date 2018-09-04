__doc__ = """This script collects the modules to perfom NLP tasks necessary for the retrieval process.
It includes a function to annotate text by the Stanford CoreNLP library, functions to extract noun phrases
 and compute noun phrases similarity by the NLTK Wordnet corpus."""

import re
import json

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer
from nltk.tree import Tree
from pycorenlp import StanfordCoreNLP
from nltk.stem.wordnet import WordNetLemmatizer

stemmer = SnowballStemmer("english", ignore_stopwords=True)
"""get stemmer"""

lemmatizer = WordNetLemmatizer()
"""get lemmatizer"""

stopwords= ['about', '-lrb-', '-rrb-', 'recent', 'every', 'whole', 'proposed', 'mentioned', 'these', 'of', "'s", "'",
            'past', 'different', 'latest', 'such', 'or', 'other', 'aforementioned', 'there', 'its', 'previous',
            'latter', 'former', 'next', 'following', 'et', 'me', 'him', 'her', 'us', 'them', 'i', 'you', 'we', 'he',
            'she', 'they', 'it', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'only',
            'alii', 'less', 'more', 'as', 'whose', 'another', 'an', 'my', 'mine', 'your', 'yours', 'our', 'ours',
            'their', 'theirs', 'his', 'hers', 'all', 'the', 'a', 'this', 'that', 'these', 'those', 'some', 'any',
            'each', 'few', 'many', 'much', 'most', 'several']
"""get stopwords"""


def annotate(text):
    """Annotate the text with tokens, sentence split, POS tags and constituency and dependency parsing."""
    nlp = StanfordCoreNLP('http://localhost:9000')# depparse,natlog,openie
    output = nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,pos,parse', 'outputFormat': 'json'})
    #print('output' ,output)
    if type(output) is str:
        output = json.loads(output, strict=False)
    return output


def clean_NP_1(np):
    """Process each NP and filters by stopwords and digits"""

    NP = []
    if not np:
        pass
    else:
        new_np = ' '.join([word for word in np if word.lower() not in stopwords and not re.search('\d', word)])
        NP = [new_np]
        NP = list(filter(None, NP))

    return (NP)


def clean_NP(np):
    """Process each NP , splitting on comma and 'and' and filters by stopwords and digits"""

    nouns=[]
    if not np:
        pass
    else:
        NPs = [l.split(',') for l in ','.join(np).lower().split('and')]
        if len(NPs)>1:
            for n in NPs:
                n=' '.join(n).split(',')
                if len(n)>1:
                    for t in n:
                        NP=t[0].split()
                        NP = ' '.join([word for word in NP if word not in stopwords and not re.search('\d', word)])
                        nouns.append(NP)
                else:
                    NP=n[0].split()
                    NP = ' '.join([word for word in NP if word not in stopwords and not re.search('\d', word)])
                    nouns.append(NP)
        else:
            NP=NPs[0]
            NP = ' '.join([word for word in NP if word not in stopwords and not re.search('\d', word)])
            nouns.append(NP)
    NP=list(filter(None, nouns))
    return(NP)


def find_np_of_np(np):
    """Process a subtree to find and return any noun phrase"""

    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    top_level_trees = [np[i] for i in range(len(np)) if type(np[i]) is Tree]
    #   search for a top-level noun
    top_level_nouns = [t for t in top_level_trees if t.label() in noun_tags]
    if len(top_level_nouns) > 0:
        #   if you find some, take the leaves
        return np.leaves()
    else:
        #   otherwise search for a top-level np
        top_level_nps = [t for t in top_level_trees if t.label()=='NP']
        if len(top_level_nps) > 0:
            #   if you find some take the leaves
            for n in top_level_nps:
                find_np_of_np(n)

        else:
            #   search for any noun
            nouns = [p[0] for p in np.pos() if p[1] in noun_tags]
            if len(nouns) > 0:
                #
                return nouns
            else:
                return np.leaves() #   if you find some take the leaves


def find_noun_phrases(tree):
    """Finds and returns all the subtrees representing an NP"""

    return [subtree for subtree in tree.subtrees(lambda t: t.label()=='NP')]


def get_nps_from_sent_no_clean(sent):
    """Retrieves and returns NPs of a sentence"""

    parsed_sent = sent['parse']
    tree = Tree.fromstring(parsed_sent)
    nounPh = find_noun_phrases(tree)
    nps = [clean_NP_1(find_np_of_np(np)) for np in nounPh] # finds and filters sub noun phrases
    # print (nps)
    nps = list(filter(None, nps))
    # print(nps)
    nps = [n for np in nps for n in np]
    or_sent = ' '.join([t['word'] for t in sent['tokens']])
    return nps


def get_nps_from_sent(sent):
    """Finds and returns the NPs of a sentence"""

    parsed_sent=sent['parse']
    tree = Tree.fromstring(parsed_sent)
    nounPh=find_noun_phrases(tree) #finds and returns all the subtrees representing an NP
    nps=[clean_NP(find_np_of_np(np)) for np in nounPh] #finds and filters all sub noun phrases
    nps=[n for np in nps for n in np]
    or_sent=' '.join([t['word'] for t in sent['tokens']])
    if 'peer review' in or_sent:
        nps=['peer review' if x=='review' else x for x in nps]
    return nps


def NPs_from_text(output, doc_id):
    """Finds and returns the NPs from the text"""

    nps=[get_nps_from_sent(sent) for sent in output['sentences']]
    return nps


def penn_to_wn_tags(pos_tag):
    """Translates Penn treebanks tags into Wordnet abbreviations for POS"""

    if pos_tag.startswith('J'):
        return wn.ADJ
    elif pos_tag.startswith('V'):
        return wn.VERB
    elif pos_tag.startswith('N'):
        return wn.NOUN
    elif pos_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def tagged_to_synset(word, tag):
    """Checks if the POS tag of a word is a Wordnet Category too and eventually retrieves the first
    of the synsets"""

    wn_tag = penn_to_wn_tags(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

     
def np_similarity(np1, np2):
    """Computes and return a semantic similarity measure between 2 phrases
                according to hyperonomy/hyponomy relations in Wordnet."""

    np1 = pos_tag(word_tokenize(np1))
    np2 = pos_tag(word_tokenize(np2))
    synsets1 = [tagged_to_synset(*word_tag) for word_tag in np1] # np1 =[('hello', 'NN'), ('boy', 'NNP')]
    synsets2 = [tagged_to_synset(*word_tag) for word_tag in np2] # np2 =[('clean', 'NN'), ('cat', 'NNP')
    synsets1 = [ss for ss in synsets1 if ss] # [Synset('house.n.01'), Synset('boy.n.01')]
    synsets2 = [ss for ss in synsets2 if ss] # [Synset('clean.n.01'), Synset('cat.n.01')]
    """checks if there are words which do not have a synset"""
    # print (np1, np2,synsets1, synsets2)
    score, count = 0.0, 0
    for synset in synsets1: # [Synset('house.n.01'),
        best = [synset.path_similarity(ss) for ss in synsets2 if synset.path_similarity(ss) != None]
        """calculates the path_similarity for each synset in synsets1 to each synset in synsets2 """                                             #[Synset('clean.n.01'), Synset('house.n.01')]
        if best:
            best_score = max(best)
            score += best_score
            count += 1
    if count:
        score /= count
        return score
    else:
        return 0
