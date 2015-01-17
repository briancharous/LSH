from __future__ import division
import sys
import random

def get_data(filename, lines):
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
    d1 = docs[doc_id_1]
    d2 = docs[doc_id_2]
    return len(d1 & d2)/len(d1 | d2)

def init_sig_matrix(rows, cols):
    return [[float('inf') for i in xrange(cols)] for j in xrange(rows)]

def signature_matrix(functions, docs): 
    """ Return a signature matrix with given hash function lists and attribute lists
    """
    rows = len(functions)
    columns =  len (docs)
    matrix = init_sig_matrix(rows, columns)
    for i, function in enumerate(functions): 
        for doc in docs:
            words = docs.get(doc)   # get a list of words for each document
            for word in words:
                cur_value = function(word)
                if cur_value < matrix[i][doc-1]:  
                    matrix[i][doc-1] = cur_value

    return matrix

def main():
    num_docs = None
    if sys.argv[2] == 'all':
        num_docs = sys.maxint
    else:
        num_docs = int(sys.argv[2])
    docs, all_words = get_data(sys.argv[1], num_docs)
    funcs = [gen_hash_func(len(all_words)) for i in xrange(int(sys.argv[3]))]
    m = signature_matrix(funcs, docs)
    print m
    print len(m[0])
    # print compute_jaccard(1, 108, docs)
    # h = gen_hash_func(len(docs))

if __name__ == '__main__':
    main()