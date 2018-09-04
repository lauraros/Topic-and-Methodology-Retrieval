__doc__ = """
This is the main script for the project T&M.

It loads the corpus collection on disk (pickle files),
and retrieves information about the main topic and the methodology of each document,
reported in 2 output text files.

Before running the script, the server for the StanfordCoreNLP library needs to be called from the command prompt/terminal:
java -mx6g -cp "D:\Java\*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -threads 8 -timeout 500000 
-maxCharLength -1 -preload tokenize,ssplit,pos,parse -outputFormat json -quiet

Moreover, some variables need to be changed:
    - directory and filename for the logfile 
    - collection_directory : directory of corpus collection (requires pickle files, main function) 
    - directory and filename for the 2 output text files (doc_stats  function)
"""

import concurrent.futures
import itertools
import logging
import os
import time

from nltk.stem.snowball import SnowballStemmer

from utils import get_data
from nlp import annotate, NPs_from_text
from extraction import get_method, get_topic


logging.basicConfig(filename='D:/Python/get_categories.log', level=logging.DEBUG)
"""log_file"""

start_time = time.time()
"""starts timing"""

stemmer = SnowballStemmer("english", ignore_stopwords=True)
"""load stemmer"""

def cluster(seclist):
    """Changes the format of the output for the report"""

    m=[tag for sec in seclist for tag in sec[2] if sec[2]]
    if m:
        cluster=itertools.groupby(sorted(m, key=lambda x:x[0]), key=lambda x:x[0])
        dic = dict()
        for k, g in cluster:
            dic[k] = list(g)
        f=[]
        for ta in dic.keys():
            newlist=[]
            for t, l in dic[ta]:
                newlist=newlist+l
            dic[ta]=newlist
            f.append((ta,newlist))
        return f
    else:
        cluster=['study']
    return cluster



def get_c(method):
    """Finds results from conclusion section"""

    full = method[1]
    c = [[secid, secheader, list] for secid, secheader, list in full
         if 'conclusion' in secheader.lower()]
    if c:
        return c
    else:
        return []


def get_m(method):
    """Finds results from methodology section"""

    full=method[1]
    exc = ['analysis of results', 'further study']
    ms = ['methodology', 'approach', 'procedure', 'technique', 'study', 'analysis', 'method']
    m = [[secid, secheader, list] for secid, secheader, list in full for term in ms
         if term in secheader.lower() and secheader.lower() not in exc]
    if m:
        return m
    else:
        return []





def doc_stats(doc_id, title, title_NPs, text_NPs, sections, topic, method):
    """Reports the output of the analysis per document to the log file and 2 other text files,
    for topics and one for methodologies.

    The report about topics contains information about DOC ID, TITLE LENGTH, NPs IN TITLE, TEXT LENGTH,
    and NPs IN TEXT	SECTIONS, additionally to 3 lists of NPs ranked by frequency, TF-IDF and similarity to the title.

    The report about methodologies contains information about DOC ID, TITLE LENGTH, NPs IN TITLE, TEXT LENGTH,
    and NPs IN TEXT	SECTIONS, additionally to the list of methodologies retrieved in the title, the whole text
    and their membership to the different sections of the text.
    """

    if not topic:
        topic = ['scientometrics']
        """assigns a neutral topic in case of null topic"""

    if not method:
        method = ['study']
        """assigns a neutral methodology in case of null method"""

    logging.info('Doc:\t{}\n'.format(doc_id))
    logging.info('ASSIGNED TOPIC:\t{}\n'.format(topic))
    logging.info('ASSIGNED METHOD:\t{}\n'.format(method))
    logging.info('Title:\t{}\n'.format(title))
    titleL = len(title)
    logging.info('Title lenght:\t{}\n'.format(titleL))
    title_NPsL = len(title_NPs)
    logging.info('NPs in title:\t{}\n'.format(title_NPsL))
    textL = len(''.join([sec[2] for sec in sections]))
    logging.info('Text lenght:\t{}\n'.format(textL))
    text_NPsL = len(text_NPs)
    logging.info('NPs in text:\t{}\n'.format(text_NPsL))
    sectionsL = len(sections)
    logging.info('Sections:\t{}\n'.format(sectionsL))
    NPsbyFr = topic[0]
    NPsbyTfidf = topic[1]
    NPsbySemsim = topic[2]

    full_collapsed=cluster(method[1])
    """Changes the format of the output for the report"""

    m_sec=get_m(method)
    if m_sec:
        m_collapsed=cluster(m_sec)
    else:
        m_collapsed=['None']
    """Separates results from methodology section"""

    c_sec=get_c(method)

    if c_sec:
        c_collapsed=cluster(c_sec)
    else:
        c_collapsed=['None']
    """Separates results from methodology section"""

    nm=[method[0], method[1], full_collapsed, m_sec, m_collapsed, c_sec, c_collapsed]
    method_headers_list=['Title', 'Full Text','All', 'Meth Section','All', 'Conl Section', 'All']
    x=list(zip(method_headers_list, nm))

    with open('collection_categories/stats_per_doc_topic.txt', 'a') as fout:
        fout.write('_' * 100+'\n')
        fout.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format('DOC ID', 'TITLE LENGTH', 'NPs IN TITLE', 'TEXT LENGTH', 'NPs IN TEXT', 'SECTIONS'))
        fout.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(doc_id, titleL, title_NPsL, textL, text_NPsL, sectionsL))
        fout.write('_' * 100+'\n')
        fout.write('TOPIC\n')
        fout.write('{}\t{}\t{}\n'.format('NPs by Frequency', 'NPs by Tf-idf', 'NPs by Semsim'))
        result = list(itertools.zip_longest(NPsbyFr, NPsbyTfidf, NPsbySemsim, fillvalue='_'))
        for quadle in result:
            fout.write('{}\t{}\t{}\n'.format(quadle[0], quadle[1], quadle[2]))
        fout.write('\n')

    with open('collection_categories/stats_per_doc_method.txt', 'a') as fout:
        fout.write('_' * 100+'\n')
        fout.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format('DOC ID', 'TITLE LENGTH', 'NPs IN TITLE', 'TEXT LENGTH', 'NPs IN TEXT', 'SECTIONS'))
        fout.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(doc_id, titleL, title_NPsL, textL, text_NPsL, sectionsL))
        fout.write('_' * 100+'\n')
        fout.write('METHODOLOGY\n')
        fout.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('Title', 'Full Text','All', 'Meth Section','All', 'Conl Section', 'All'))
        r = list(itertools.zip_longest(method[0], method[1], full_collapsed, m_sec, m_collapsed, c_sec, c_collapsed, fillvalue='_'))
        for q in r:
            fout.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(q[0], q[1], q[2], q[3], q[4], q[5], q[6]))
        fout.write('\n')



def check_doc(doc):
    """Performs the analysis of a document retrieving its topic and its methodology"""

    doc_id = doc[0]
    print(doc_id)
    if int(doc_id) > 0:
        title = doc[1]
        sections = doc[2]
        print(title)
        try:
            text_per_sec = [sec[2] for sec in sections]
            annotated_title = annotate(title) # Annotates the title
            title_NPs_per_sent = NPs_from_text(annotated_title, doc_id) # Gets NPs in the title
            title_NPs = [np for sent in title_NPs_per_sent for np in sent] # not stemmed
            """Gets NPs from title"""

            annotated_text_per_sect = [[sec[0], sec[1], annotate(sec[2])] for sec in sections] # Annotates the text
            NPs_per_sect_per_sent = [[sec[0], sec[1], NPs_from_text(sec[2], doc_id)] for sec in annotated_text_per_sect]
            # Gets NPs in the text
            sex_NPs = [sec[2] for sec in NPs_per_sect_per_sent]
            text_NPs = [np for sec_NPs in sex_NPs for sent in sec_NPs for np in sent] # not stemmed
            stemmed_text_NPs = [' '.join([stemmer.stem(n) for n in np.split()]) for sec_NPs in sex_NPs for sent in
                                sec_NPs for np in sent]
            """Get NPs from sections"""

            topic = get_topic(text_per_sec, stemmed_text_NPs, title_NPs, text_NPs, 20)
            """Gets topic"""

            method = get_method(annotated_title, annotated_text_per_sect, topic, title_NPs,
                                NPs_per_sect_per_sent, sections)
            """Gets method"""

            stats = doc_stats(doc_id, title, title_NPs_per_sent, text_NPs, sections, topic, method)
            """Reports the results of the analysis to the log-file and 2 text files"""

        except:
            logging.ERROR('cannot parse')


def check_collection(directory):
    """Analyzes the whole collection using 8 threads"""

    filelist = os.listdir(directory)
    for fname in filelist:
        if fname.endswith('.pickle'):  # not fname.endswith(".log"):
            collection_part = get_data(directory + '/' + fname)
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=8) as executor:  # change the index according to usable threads
                for doc, doc_info in zip(collection_part, executor.map(check_doc, collection_part)):
                    print(f"Categories for doc {doc[0]} were saved as {doc_info}")


def main():
    """analyzes the collection"""

    collection_directory = 'D:/Python/collection' #change with corpus directory
    check_collection(collection_directory)


if __name__ == '__main__':
    """init script"""

    main()


print("--- %s seconds ---" % (time.time() - start_time))
"""prints ending time"""
logging.info('end time: {}s seconds '.format(time.time() - start_time))
"""logs end time"""
