import nltk
import sys
import string
import os
import math
from collections import Counter

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dictionary={}
    for file in os.listdir(directory):
        dictionary[file]=open(directory+os.sep+file,encoding='utf-8').read()
    return dictionary
        


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens=nltk.word_tokenize(document.lower())
    punctuations=set(string.punctuation)
    stopwords=set(nltk.corpus.stopwords.words('english'))
    result=[]
    for x in tokens:
        if x not in punctuations and x not in stopwords:
            result.append(x)
    return result


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    word_file_count={}
    no_documents=len(documents.keys())
    for file in documents:
        words=set(documents[file])
        for word in words:
            if word in word_file_count:
                word_file_count[word]+=1
            else:
                word_file_count[word]=1
    idf={word:math.log(no_documents/word_file_count[word]) for word in word_file_count}
    return idf
            
            
            


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs=[]
    for file in files:
        tf_idf=0
        for q in query:
            tf_idf+=idfs[q]*files[file].count(q)
        tf_idfs.append((tf_idf,file))
    import heapq
    res=list(heapq.nlargest(n,tf_idfs))
    res=[x[1] for x in res]
    return res
    
    


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    res=[]
    for sentence in sentences:
        idf=0
        count_word_found=0
        words_in_sentence=set(tokenize(sentence))
        for q in query:
            if q in words_in_sentence:
                idf+=idfs[q]
                count_word_found+=1
        density=count_word_found/len(words_in_sentence)
        res.append((sentence,idf, density))
    res.sort(key=lambda x:(x[1],x[2]),reverse=True)
    
    res=[x[0] for x in res[:n]]
    return res
    
    
    
    
    


if __name__ == "__main__":
    main()
