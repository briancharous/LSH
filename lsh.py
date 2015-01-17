""" 
lsh.py by Brian Charous and Yawen Chen
for CS324 Winter 2015

This program takes in a text file with word counts and 
computes the jaccard similarity between 2 documents and
estimates the jaccard similarity by generating a signature matrix

To run: argument 1 is the filename containing the word counts
        argument 2 is the number of lines to read from said file; 'all' is equivalent to the number of lines in the file
        argument 3 is number of has functions to generate for the signature matrix
        arguments 4 and 5 are the document ids of of which to compute the jaccard similarity

"""

from __future__ import division
import sys
import random

def get_data(filename, lines):
    """ read in data from filename """
    docs = dict()
    all_words = set()
    with open(filename, 'r') as f:
        for i in range(3):
            f.next() # skip first 3 lines
        linenum = 0
        for line in f:
            if linenum == lines:
                break
            components = line.split()
            components_int = [int(c) for c in components]
            if components_int[0] in docs:
                docs[components_int[0]].add(components_int[1])
            else:
                docs[components_int[0]] = set([components_int[1]])
            all_words.add(components_int[1])
            linenum+=1
    return docs, all_words

def gen_hash_func(n):
    """ generate has function of the form h(x) = ax + b mod n
    where and n is the number of words in the dataset 
    and a & b are chosen pseudorandomly in the range 0 to n-1"""
    a = random.randint(0, n-1)
    b = random.randint(0, n-1)
    def hash(x):
        return (a*x+b)%n
    return hash

def compute_jaccard(doc_id_1, doc_id_2, docs):
    """ compute jaccard similarity between 2 documents 
    using formula |A intersect B|/|A union B|"""
    d1 = docs[doc_id_1]
    d2 = docs[doc_id_2]
    return len(d1 & d2)/len(d1 | d2)

def init_sig_matrix(rows, cols):
    """ init signature matrix of rows and cols with vals all infinity """
    return [[float('inf') for i in xrange(cols)] for j in xrange(rows)]

def signature_matrix(functions, docs): 
    """ Return a signature matrix with given hash function lists and attribute lists
    """
    rows = len(functions)
    columns =  len (docs)
    matrix = init_sig_matrix(rows, columns)     #initialiing the matrix
    for i, function in enumerate(functions): 
        for doc in docs:
            words = docs.get(doc)   # get a list of words for each document
            for word in words:
                cur_value = function(word)
                if cur_value < matrix[i][doc-1]:    # replace if current value is smaller than the stored value
                    matrix[i][doc-1] = cur_value

    return matrix

def jarccard_probability(matrix, doc_id_1, doc_id_2, n):
    """ Returns the approximated jaccard similarity of two documents 
         computed through computing the probrability of having the same min-hash 
         given n number of fash functions
    """
    same_count = 0
    for i in range(n):      #loop though n rows of the matrix
        if matrix[i][doc_id_1] == matrix[i][doc_id_2]:      #check if the min-hash values are the same
            same_count += 1
    return same_count/n

def main():
    if len(sys.argv) != 6:
        print "Usage: pypy {0} input_file num_lines num_hash_functions doc_1_id doc_id_2".format(sys.argv[0])
        exit(0)

    num_docs = None
    if sys.argv[2] == 'all':
        num_docs = sys.maxint
    else:
        num_docs = int(sys.argv[2])
    docs, all_words = get_data(sys.argv[1], num_docs)
    random.seed(27945)
    num_hash_funcs = int(sys.argv[3])
    funcs = [gen_hash_func(len(all_words)) for i in xrange(num_hash_funcs)]
    m = signature_matrix(funcs, docs)
    doc_id_1 = int(sys.argv[4])
    doc_id_2 = int(sys.argv[5])
    jaccard_estimate = jarccard_probability(m, doc_id_1, doc_id_2, num_hash_funcs)
    actual_jaccard = compute_jaccard(doc_id_1, doc_id_2, docs)
    print "Estimated jaccard: {0}, actual jaccard: {1}".format(jaccard_estimate, actual_jaccard)

if __name__ == '__main__':
    main()