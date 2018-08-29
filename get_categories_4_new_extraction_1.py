__doc__ = """This script collects the modules to retrieve the topics and methodologies.
The path for the log file needs to be changed."""

import collections
import itertools
import logging
import re
import time

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from get_categories_4_NLP_1 import get_nps_from_sent_no_clean, np_similarity

logging.basicConfig(filename='D:/Python/get_categories.log', level=logging.DEBUG) #need to change the path

start_time = time.time()
"""start timing"""

wnl = WordNetLemmatizer()
"""get Lemmatizer"""

stopwords = ['about', '-lrb-', '-rrb-', 'recent', 'every', 'whole', 'proposed', 'mentioned', 'these', 'of', "'s", "'",
             'past', 'different', 'latest', 'such', 'or', 'other', 'aforementioned', 'there', 'its', 'previous',
             'latter', 'former', 'next', 'following', 'et', 'me', 'him', 'her', 'us', 'them', 'i', 'you', 'we', 'he',
             'she', 'they', 'it', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'only',
             'alii', 'less', 'more', 'as', 'whose', 'another', 'an', 'my', 'mine', 'your', 'yours', 'our', 'ours',
             'their', 'theirs', 'his', 'hers', 'all', 'the', 'a', 'this', 'that', 'these', 'those', 'some', 'any',
             'each', 'few', 'many', 'much', 'most', 'several']
"""get stopwords"""

stemmer = SnowballStemmer("english", ignore_stopwords=True)
"""get stemmer"""

expsl = ['based on', 'by use of', 'by the use of', 'by using', 'utilizing']
"""get expressions for methodologies"""


def get_semsim_NPs_ranking(stemmed_text_NPs, text_NPs, title_NPs, stemmedNPsbyfr):
    """Rank NPs by similarity to the title"""

    stemmedNOP = zip(stemmed_text_NPs, text_NPs)
    selected = [(stem, np) for stem, np in stemmedNOP if len(np.split()) > 1] #Excludes single nouns
    paired_stem_orNPs = collections.defaultdict(set)
    for stem, NP in selected:
        paired_stem_orNPs[stem].add(NP)
    not_stemmedNPs = [v for stem, fq in stemmedNPsbyfr for v in paired_stem_orNPs[stem]] #list of not-stemmed noun phrases
    """Not-stemmed noun phrases for the 50 most common stemmed ones"""

    combination = [(a, b) for a, b in itertools.product(title_NPs, not_stemmedNPs)]
    """Lists all possible pairs of NP and title"""

    semantics = sorted([[(a, b), (np_similarity(a, b) + np_similarity(b, a)) / 2] for a, b in combination],
                       key=lambda x: x[1], reverse=True)
    """Gets and sort noun phrases by the similarity score"""

    result = [(' '.join([stemmer.stem(n) for n in a[1].split()]), b) for a, b in semantics] # Stemming

    main = itertools.groupby(sorted(result, key=lambda x: x[0]), lambda x: x[0])
    dic = dict()
    for k, g in main:
        dic[k] = list(g)

    result = sorted([sorted(dic[np], reverse=True)[0] for np in dic.keys()], key=lambda x: x[1], reverse=True)
    return result



def get_tf_idf_NPs_ranking(text_per_sec, stemmed_text_NPs):
    """Rank NPs by tf-idf"""

    doc = 0
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 6),
                                 min_df=0, stop_words='english', sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(text_per_sec)
    feature_names = vectorizer.get_feature_names()
    feature_index = tfidf_matrix[doc, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
    m = sorted(
        [(' '.join([stemmer.stem(n) for n in w.split() if n.lower not in stopwords and not re.search('\d', n)]), s) for
         w, s in [(feature_names[i],
                   s) for (i, s) in tfidf_scores] if
         len([stemmer.stem(n) for n in w.split() if n.lower not in stopwords and not re.search('\d', n)]) > 1],
        key=lambda x: x[1], reverse=True)
    filtered= [(np, s) for np, s in m if np in stemmed_text_NPs]
    main = itertools.groupby(sorted(filtered, key=lambda x: x[0]), lambda x: x[0])
    dic = dict()
    for k, g in main:
        dic[k] = list(g)
    result = sorted([sorted(dic[np], reverse=True)[0] for np in dic.keys()], key=lambda x: x[1], reverse=True)
    return result


def get_tf_NPs_ranking(stemmed_text_NPs):
    """Rank NPs by frequency"""

    stemmed_NPs_fr = collections.Counter([np for np in stemmed_text_NPs if len(np.split()) > 1]) # excludes single nouns
    return stemmed_NPs_fr


def get_topic(text_per_sec, stemmed_text_NPs, title_NPs, text_NPs,
              top=20):
    """Returns the top (20) topics of a doc in the form of three vectors, top 20 noun phrases ranked by frequency, tf-idf
    and by similarity to the title"""

    stemmedNPsbyfr = get_tf_NPs_ranking(stemmed_text_NPs)
    stemmedNPsbytfidf = get_tf_idf_NPs_ranking(text_per_sec, stemmed_text_NPs)
    if title_NPs: # If no NP in title is found then it assigns a neutral value to the ranking by semsim
        stemmedNPsbysemsim = get_semsim_NPs_ranking(stemmed_text_NPs, text_NPs, title_NPs,
                                                    stemmedNPsbyfr.most_common(50))
    else:
        stemmedNPsbysemsim = [('empty title', 0)]
    bestNPsbyfr = [np for np, n in stemmedNPsbyfr.most_common(top)]
    bestNPsbytfidf = [np for np, n in stemmedNPsbytfidf[:top]]
    bestNPsbysemsim = [np for np, n in stemmedNPsbysemsim[:top]]
    return (bestNPsbyfr, bestNPsbytfidf, bestNPsbysemsim)


def cluster(depmet,NPsmet):
    """Reorgainize the results eliminating empty lists"""

    if depmet and NPsmet:
        cluster=depmet+NPsmet
        cluster=itertools.groupby(sorted(cluster, key=lambda x:x[0]), key=lambda x:x[0])
        dic = dict()
        for k, g in cluster:
            dic[k] = list(g)
        for tag in dic.keys():
            newlist=[]
            for t, l in dic[tag]:
                newlist=newlist+l
            dic[tag]=newlist
        l=[(tag, dic[tag])for tag in dic.keys()]
        return l
    else:
        cluster=depmet+NPsmet
    return cluster


def get_means_from_NPs(NPs, topic):
    """Selects and returns noun phrases on a lexical base"""

    methodology_NPs = dict()

    ex = [topic, 'case study']
    met_terms = ['method', 'methods', 'methodology', 'methodologies', 'approach', 'approaches', 'procedure',
                 'procedures', 'technique', 'techniques', 'model', 'models', 'analysis']
    methods = set([' '.join([stemmer.stem(n.lower()) for n in np.split()]) for np in NPs if
                   len(np.split()) > 1 and [np.split()[-1] for word in np.split() if
                                            np.split()[-1] in met_terms] and np not in ex])
    methodology_NPs['methods'] = methods
    tool_terms = ['source', 'sources', 'database', 'databases', 'tool', 'tools', 'toolbox', 'toolkit', 'toolkits',
                  'api', 'apis', 'repository', 'repositories', 'instrument', 'instruments', 'package', 'packages',
                  'archive', 'archives', 'libraries', 'library', 'data', 'corpus', 'corpora']
    tools = set([' '.join([stemmer.stem(n.lower()) for n in np.split()]) for np in NPs if
                 len(np.split()) > 1 and [np.split()[-1] for word in np.split() if
                                          np.split()[-1] in tool_terms] and np not in ex and np not in methods])
    methodology_NPs['tools'] = tools
    metric_terms = ['metric', 'measure', 'metrics', 'measures', 'score', 'scores']
    metrics = set([' '.join([stemmer.stem(n.lower()) for n in np.split()]) for np in NPs if
                   len(np.split()) > 1 and [np.split()[-1] for word in np.split() if np.split()[
                       -1] in metric_terms] and np not in ex and np not in methods and np not in tools])
    methodology_NPs['metrics'] = metrics
    ms = [(m, [t for t in methodology_NPs[m]]) for m in methodology_NPs.keys() if methodology_NPs[m]]

    return ms


def get_means_from_dependencies_categories(deps):
    """Categorizes means-purpose relations"""

    deps_by_categories = {}
    tools = []
    methods = []
    metrics = []
    verbs = []
    mes = ['measur', 'calculat', 'comput']

    for mean, head in deps:
        #print (mean, head)
        if not mean[1] and not head[0] and not head[1]:
            verbs.append([mean, head])

        elif any(m for m in mes if m in mean[0] or m in head[0]):
            metrics.append([mean, head])

        elif mean[1]:
            tools.append([mean, head])
        else:
            methods.append([mean, head])
    deps_by_categories['tools'] = tools
    deps_by_categories['methods'] = methods
    deps_by_categories['metrics'] = metrics

    ms = [(m, [t for t in deps_by_categories[m]]) for m in deps_by_categories.keys() if deps_by_categories[m]]

    return ms


def ex_roots(deparse):
    """Return verb if intr v"""

    try:
        ms = []

        required_vs = ['VB','VBZ','VBN','VBG','VBD']
        roots = [tok['dependentGloss'] for tok in deparse if
                         tok['dep'] == 'ROOT' and pos_tag(word_tokenize(tok['dependentGloss']))[0][1] in required_vs]
        if roots:
            for root in roots:
                subs=[tok['dependentGloss'] for tok in deparse if tok['dep']=='nsubj' and tok['governorGloss']==root and tok['dependentGloss'].lower()== 'we']
                dobjs=[tok['dependentGloss'] for tok in deparse if tok['dep']=='dobj' and tok['governorGloss']==root ]
                meths=[]
                if subs and dobjs:

                    meths=[[(root, dobjs[0]), ('', '')]]
                elif subs and not dobjs:
                    meths = [[(root, ''), ('', '')]]
                ms = ms + meths
                return ms
        else:
            return []
    except:
        print('prob with ex mean phrases')
        pass


def ex_means_clauses(deparse, NPs):
    """Extracts and returns sets of four items in the form [(V_1, NP_1)(V_2, NP_2] where the first pair is relative
        to the means clause and the second to the main clause"""

    try:
        ms = []
        verbs = [tok['dependentGloss'] for tok in deparse if tok['dep'] == 'advcl:by']  # verbify
        if verbs:
            main_v = [tok['governorGloss'] for tok in deparse if tok['dep'] == 'advcl:by']
            tups = list(zip(verbs, main_v))
            for v, mv in tups:
                dobj = [tok['dependentGloss'] for tok in deparse if tok['governorGloss'] == mv and tok['dep'] == 'dobj']
                d_dobj = [tok['dependentGloss'] for tok in deparse if
                          tok['governorGloss'] == v and tok['dep'] == 'dobj']
                sub = [tok['dependentGloss'] for tok in deparse if
                       tok['governorGloss'] == mv and tok['dep'] == 'nsubj' or tok['dep'] == 'nsubjpass']
                if dobj and d_dobj:
                    d = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if dobj[0] in np ]
                    do = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if d_dobj[0] in np]
                    if d and do:
                        meths = [[(v, do[0]), (mv, d[0])]]
                        ms = ms + meths
                    else:
                        meths = [[(v, stemmer.stem(d_dobj[0])),((mv, stemmer.stem(dobj[0])))]]
                        ms = ms + meths
                elif dobj and not d_dobj:
                    meths = [[(v, ''), (mv, dobj[0])]]
                    ms = ms + meths
                elif d_dobj and sub:
                    meths = [[(v, stemmer.stem(d_dobj[0])), (mv, sub[0])]]
                    ms = ms + meths
                elif not d_dobj and sub:
                    meths = [[(v, ''), (mv, sub[0])]]
                    ms = ms + meths
                elif d_dobj and not sub:
                    meths = [[(v, stemmer.stem(d_dobj[0])), (mv, '')]]
                    ms = ms + meths
                elif not d_dobj and not sub:
                    meths = [[(v, ''), (mv, '')]]
                    ms = ms + meths

            return ms
        else:
            return []
    except:
        print('prob with ex mean clauses')
        return []


def ex_means_phrases(deparse, NPs):
    """Extracts and returns sets of four items in the form [(V_1, NP_1)(V_2, NP_2] where the first pair is relative
        to the means phrase and the second to the main clause"""
    try:
        ms = []
        t = ['use', 'application', 'mean', 'means']
        compl = [tok['dependentGloss'] for tok in deparse if
                 tok['dep'] == 'nmod:by' and tok['dependentGloss'] in t]  # and dep in list
        if compl:
            main_v = [tok['governorGloss'] for tok in deparse if tok['dep'] == 'nmod:by']
            tups = list(zip(compl, main_v))
            for v, mv in tups:
                dobj = [tok['dependentGloss'] for tok in deparse if tok['governorGloss'] == mv and tok['dep'] == 'dobj']
                d_dobj = [tok['dependentGloss'] for tok in deparse if
                          tok['governorGloss'] == v and tok['dep'] == 'nmod:of']
                sub = [tok['dependentGloss'] for tok in deparse if
                       tok['governorGloss'] in main_v and tok['dep'] == 'nsubj' or tok['dep'] == 'nsubjpass']
                if dobj and d_dobj:
                    d = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if dobj[0] in np]
                    do = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if d_dobj[0] in np]
                    meths = [[(v, do[0]), (mv, d[0])]]
                    ms = ms + meths
                elif dobj and not d_dobj:
                    d = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if dobj[0] in np]

                    meths = [[(v, ''), (mv, d[0])]]
                    ms = ms + meths
                elif d_dobj and sub:
                    s = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if sub[0] in np]
                    do = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if d_dobj[0] in np]
                    meths = [[(v, do[0]), (mv, s[0])]]
                    ms = ms + meths
                elif not d_dobj and sub:
                    s = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if sub[0] in np]
                    meths = [[(v, ''), (mv, s[0])]]
                    ms = ms + meths
                elif d_dobj and not sub:
                    do = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if d_dobj[0] in np]
                    meths = [[(v, do[0]), (mv, '')]]
                    ms = ms + meths
                elif not d_dobj and not sub:
                    meths = [[(v, ''), (mv, '')]]
                    ms = ms + meths
            return ms
        else:
            return []
    except:
        print('prob with ex mean phrases')
        pass


def ex_purpose_clauses(deparse, NPs):
    """Extracts and returns sets of four items in the form [(V_1, NP_1)(V_2, NP_2] where the first pair is relative
    to the main clause and the second to purpose clause"""

    try:
        ms = []
        v_tags = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
        fins = ['advcl:in_order', 'advcl:for']
        main_v = [tok['governorGloss'] for tok in deparse if
                  tok['dep'] in fins and pos_tag([tok['governorGloss']]) in v_tags]
        if main_v:
            verbs = [tok['dependentGloss'] for tok in deparse if tok['dep'] in fins]

            tups = list(zip(main_v, verbs))

            for mv, v in tups:
                dobj = [tok['dependentGloss'] for tok in deparse if tok['governorGloss'] == mv and tok['dep'] == 'dobj']
                d_dobj = [tok['dependentGloss'] for tok in deparse if
                          tok['governorGloss'] == v and tok['dep'] == 'dobj']
                sub = [tok['dependentGloss'] for tok in deparse if
                       tok['governorGloss'] == mv and tok['dep'] == 'nsubj' or tok['dep'] == 'nsubjpass']
                if dobj and d_dobj:
                    d = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if dobj[0] in np]
                    do = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if d_dobj[0] in np]
                    meths = [[(mv, d[0]), (v, do[0])]]
                    ms = ms + meths
                elif dobj and not d_dobj:
                    d = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if dobj[0] in np]
                    meths = [[(mv, d[0]), (v, '')]]
                    ms = ms + meths
                elif d_dobj and sub:

                    do = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if d_dobj[0] in np]
                    meths = [[(mv, sub[0]), (v, do[0])]]
                    ms = ms + meths
                elif not d_dobj and sub:
                    s = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if sub[0] in np]
                    meths = [[(mv, s[0]), (v, '')]]
                    ms = ms + meths
                elif d_dobj and not sub:
                    do = [' '.join([stemmer.stem(n) for n in np.split() if n.lower() not in stopwords
                                    and not re.search('\d', n)]) for np in NPs if d_dobj[0] in np]
                    meths = [[(mv, ''), (v, do[0])]]
                    ms = ms + meths
                elif not d_dobj and not sub:
                    meths = [[(mv, ''), (v, '')]]
                    ms = ms + meths

            return ms
        else:
            return []
    except:
        print('prob with ex purpose clauses')
        pass


def extract_rules(sent, deparse):
    """Extracts and returns four-items sets by dependency parsing rules
                        for means and purpose clauses and complements"""
    
    methods = []
    NPs = get_nps_from_sent_no_clean(sent)

    """looking for means clauses"""
    means_clauses = ex_means_clauses(deparse, NPs)
    if means_clauses:
        methods = methods + means_clauses
    else:
        pass
    
    """looking for means phrases"""
    means_phrases = ex_means_phrases(deparse, NPs)
    if means_phrases:
        methods = methods + means_phrases
    else:
        pass
    
    """looking for final clauses"""
    purpose_clauses = ex_purpose_clauses(deparse, NPs)
    if purpose_clauses:
        methods = methods + purpose_clauses
    else:
        pass
    if methods:
        return methods
    else:
        return []


def get_means_from_dependencies(annotation):
    """Extracts and returns methodologies by the use of dependency parsing
                    as both four-items sets and main predicates lists"""

    pnpns = []
    predicates= []
    for sent in annotation['sentences']:
        deparse = sent['enhancedDependencies']
        pnpn = extract_rules(sent, deparse) #get four-items sets
        verbs_we = ex_roots(deparse) #get verbs
        predicates = predicates + verbs_we
        pnpns = pnpns + pnpn
    if pnpns and predicates:
        methods_by_dep=get_means_from_dependencies_categories(pnpns)
        methods_by_dep.append(('verbs', [list(v) for v in list(set(tuple(row) for row in predicates))]))
        return methods_by_dep
    elif pnpns and not predicates:
        methods_by_dep=get_means_from_dependencies_categories(pnpns)
        return methods_by_dep
    else:
        return pnpns



def get_methods_from_sections(sections, topic):
    """Returns a list of methods grabbed from a list of sections' texts both by dependencies and by lexicon"""

    methods = []
    for sec_id, sec_header, annotation, NPs_per_sent in sections:
        depmet = get_means_from_dependencies(annotation)
        NPs = [np for sent in NPs_per_sent for np in sent]
        NPsmet = get_means_from_NPs(NPs, topic)
        comb = cluster(depmet, NPsmet) #Reorgainize the results
        methods.append([sec_id, sec_header, comb])
    return methods


def clean_method_title(met):
    """Cleans results from empty lists"""

    l=[t for m in met.keys() for t in met[m] if met[m]] #m = dep or lex, t = pnpn or verb
    if l:
        cluster=itertools.groupby(sorted(l, key=lambda x:x[0]), key=lambda x:x[0])
        dic = dict()
        for k, g in cluster:
            dic[k] = list(g)
        for tag in dic.keys():
            newlist=[]
            for t, l in dic[tag]:
                newlist=newlist+l
            dic[tag]=newlist
        c=[(tag, dic[tag])for tag in dic.keys()]
        return c
    else: return ['None']


def get_methods_from_title(annotated_title, NPs, topic):
    """Retrieves methodologies in a text by dependencies and by lexicon"""

    methods = dict()
    met = get_means_from_dependencies(annotated_title)
    methods['deps'] = met
    met = get_means_from_NPs(NPs, topic)
    methods['lex'] = met
    return methods


def get_method(annotated_title, annotated_text_per_sect, topic, title_NPs, NPs_per_sect_per_sent,
               sections):
    """Returns methodologies separately from the title and from the text of the document."""

    methods = []
    met = get_methods_from_title(annotated_title, title_NPs, topic)
    """Extract methodologies from the title"""

    if met:
        m=clean_method_title(met)
        methods.append(m)
    else:
        methods.append(['no method in title'])
    full_text = [(sec1[0], sec1[1], sec2[2], sec3[2]) for sec1, sec2, sec3 in
                 zip(sections, annotated_text_per_sect, NPs_per_sect_per_sent)]
    met = get_methods_from_sections(full_text, topic)
    """Extract methodologies from the text"""

    if met:
        methods.append(met)
    else:
        methods.append(['study'])
    return methods