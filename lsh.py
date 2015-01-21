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
import marshal
import argparse
import heapq
from collections import defaultdict

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
    columns = len(docs)
    matrix = init_sig_matrix(rows, columns)     #initialiing the matrix
    for i, function in enumerate(functions): 
        for doc in docs:
            words = docs.get(doc)   # get a list of words for each document
            for word in words:
                cur_value = function(word)
                if cur_value < matrix[i][doc-1]:    # replace if current value is smaller than the stored value
                    matrix[i][doc-1] = cur_value

        # update user on progress
        percent = (i/rows)*100
        sys.stdout.write('\r[{0}] {1}%'.format('#'*int(percent/5), percent))
        sys.stdout.flush()

    sys.stdout.write('\r[{0}] {1}%\n'.format('#'*int(100/5), 100))
    sys.stdout.flush()
    return matrix

def jarccard_probability(matrix, doc_id_1, doc_id_2, n):
    """ Returns the approximated jaccard similarity of two documents 
         computed through computing the probrability of having the same min-hash 
         given n number of fash functions
    """
    same_count = 0
    for i in range(n):  #loop though n rows of the matrix
        if matrix[i][doc_id_1-1] == matrix[i][doc_id_2-1]: #check if the min-hash values are the same
            same_count += 1
    return same_count/n

def brute_force_nearest_neighbors(k, ref_doc_id, docs):
    """ brute force out k nearest neighbors for a given document id"""
    nearest_neighbors = []
    for doc_id in docs:
        if ref_doc_id != doc_id:
            jaccard_compare = compute_jaccard(ref_doc_id, doc_id, docs)
            if len(nearest_neighbors) < k:
                nearest_neighbors.append((jaccard_compare, ref_doc_id))
                if len(nearest_neighbors) == k:
                    heapq.heapify(nearest_neighbors)
            else:
                heapq.heappush(nearest_neighbors, (jaccard_compare, doc_id))
                heapq.heappop(nearest_neighbors)
    return nearest_neighbors

def brute_force_jaccard_all(k, docs):
    """ brute force an average jaccard score over all documents using k nearest neighbors """
    jsum = 0
    d = len(docs)
    index = 0
    for doc in docs:
        neighbors = brute_force_nearest_neighbors(k, doc, docs)
        average = sum([i[0] for i in neighbors])/k
        jsum += average

        percent = (index/d)*100
        sys.stdout.write('\r[{0}] {1}%'.format('#'*int(percent/5), round(percent,2)))
        sys.stdout.flush()
        index+=1

    sys.stdout.write('\r[{0}] {1}%\n'.format('#'*int(100/5), 100))
    return jsum/len(docs)

def save_sig_matrix(matrix, filename):
    """ dump signature matrix to file so it doesn't constantly have to be recreated """
    with open(filename, 'w') as f:
        marshal.dump(matrix, f, 2)

def load_sig_matrix(filename):
    """ load signature matrix from file """
    with open(filename, 'r') as f:
        matrix = marshal.load(f)
        return matrix
    return None

def create_band_hashes(matrix, b):
    """ Do LSH on matrix using b bands """
    hashes = []
    for i in xrange(0, len(matrix), b):
        rows = matrix[i:i+b]
        tohash = []
        for j in xrange(0, len(rows[0])):
            # get a list containing 1 element from each row in a band at index j (doc id)
            column_elem = []
            for k in xrange(0, len(rows)):
                column_elem.append(rows[k][j])
            tohash.append((tuple(column_elem), j+1)) # j+1 should be doc id     
        d = defaultdict(set)
        for key, value in tohash:
            d[key].add(value)
        hashes.append(d)
    return hashes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--words', required=True, help='File containing document word counts')
    parser.add_argument('-l', '--lines', required=False, help='Number of lines to read from the words file')
    parser.add_argument('-n', '--number_of_hashes', required=False, help='Number of has functions to include in the signature matrix')
    parser.add_argument('-d1', '--document1', required=True, help='First document ID to compare')
    parser.add_argument('-d2', '--document2', required=True, help='Second document ID to compare')
    parser.add_argument('-o', '--matrix_output', required=False, help='Filename to dump matrix to')
    parser.add_argument('-m', '--matrix_file', required=False, help='Filename of saved signature matrix')
    parser.add_argument('-k', '--k', required=True, help='Number of nearest neighbors to find')

    args = parser.parse_args()

    # read in words
    sys.stdout.write("Reading in words...")
    sys.stdout.flush()
    num_docs = None
    if args.lines in (None, 'all'):
        num_docs = sys.maxint
    else:
        num_docs = int(args.lines)
    docs, all_words = get_data(args.words, num_docs)
    print(" done!")

    matrix = None
    num_hash_funcs = None
    if args.matrix_file is not None:
        # read in sig matrix from file
        sys.stdout.write("Reading matrix from file...")
        sys.stdout.flush()
        with open(args.matrix_file, 'r') as f:
            matrix = marshal.load(f)
            num_hash_funcs = len(matrix)
        print(" done!")
    else:
        print("Generating signature matrix...")      
        random.seed(27945)
        num_hash_funcs = int(args.number_of_hashes)
        if num_hash_funcs is None:
            print "Error: number of hash functions is required when generating a signature matrix"
            exit(0)
        funcs = [gen_hash_func(len(all_words)) for i in xrange(num_hash_funcs)]
        matrix = signature_matrix(funcs, docs)

    # dump matrix
    if args.matrix_output is not None:
        sys.stdout.write("Saving matrix to file...")
        sys.stdout.flush()
        save_sig_matrix(matrix, args.matrix_output)
        print(" done!\nSignature matrix written to {}".format(args.matrix_output))

    # jaccard similarity/estimate between 2 docs
    doc_id_1 = int(args.document1)
    doc_id_2 = int(args.document2)
    jaccard_estimate = jarccard_probability(matrix, doc_id_1, doc_id_2, num_hash_funcs)
    actual_jaccard = compute_jaccard(doc_id_1, doc_id_2, docs)
    print "Estimated jaccard: {0}, actual jaccard: {1}".format(jaccard_estimate, actual_jaccard)

    # brute force nearest neighbors
    # print "Finding average jaccard by using nearest neighbors brute force..."
    # jaccard_all = brute_force_jaccard_all(int(args.k), docs)
    # nearest_neighbors_str = [str(i[1]) for i in nearest_neighbors]
    # print "Average jaccard similarity is {0}".format(jaccard_all)

    band_hashes = create_band_hashes(matrix, 3)

if __name__ == '__main__':
    main()