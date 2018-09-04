# Topics and Methodologies Retrieval
This repository collects the code necessary to retrieve topics and methodologies for each document in a collection.
The topics are represented by NPs automatically extracted and ranked according to frequency, tf-idf and semantic similarity to the title's NPs .
The methodologies are represented by NPs, selected by lexicon, mean-purpose relations and predicates.

## Pre-requisites
The code relies on Python 3 and its libraries, mainly the Standford CoreNLP library, in the wrapper pycorenlp and Wordnet corpus from NLTK library.

## Code Description
The code loads the corpus collection on disk (pickle files),
and retrieves information about the main topic and the methodology of each document,
reported in 2 output text files.

- topic-methodology-retrieval.py is the main script and manages the whole retrieval process,
- utils.py manages the reading and writing of the data,
- extraction.py manages the extraction rules,
- NLP.py manages the annotation and the subtasks necessary for the extraction procedure.

## Running
Before running the script, the server for the StanfordCoreNLP library needs to be called from the command prompt/terminal:
```
java -mx6g -cp "D:\Java\*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -threads 8 -timeout 500000 
-maxCharLength -1 -preload tokenize,ssplit,pos,parse -outputFormat json -quiet
```


Moreover, some variables need to be changed in the get_categories_4.py :
    - directory and filename for the logfile 
    - collection_directory : directory of corpus collection (requires pickle files, main function) 
    - directory and filename for the 2 output text files (doc_stats  function)
    
Finally, only the main script needs to be run from the command line:

```
python3 topic-methodology-retrieval.py
```


