from __future__ import division
import sys
import random

def get_data(filename, lines):
    docs = dict()
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
            linenum+=1
    return docs

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

""" Return a signature matrix with given hash function lists and attribute lists
"""
def signature_matrix(functions, docs): 
    functions = functions
    # For each function, generates the signature matrix
    for function in functions:  
        for doc in docs:

            words = doc.get(doc_id)


def main():
    num_docs = None
    if sys.argv[2] == 'all':
        num_docs = sys.maxint
    else:
        num_docs = int(sys.argv[2])
    docs = get_data(sys.argv[1], num_docs)
    print compute_jaccard(1, 108, docs)
    h = gen_hash_func(len(docs))

if __name__ == '__main__':
    main()