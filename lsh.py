""" 
lsh.py by Brian Charous and Yawen Chen
for CS324 Winter 2015

This program takes in a text file with word counts and 
computes the average jaccard similarity between all documents.
It also finds k nearest neighbors do said document using brute force
and locality sensitive hashing. 

To run: -w is the filename containing the word counts
        -l is the number of lines to read from said file; 'all' is equivalent to the number of lines in the file
        -n is number of has functions to generate for the signature matrix
        -d is the document id of of which to compute the jaccard similarity with
        -k is the number of nearest neighbors to find (of document d)
        -r is the number of rows in a band for the LSH approach
        -m loads a saved matrix from file
        -o saves a signature matrix to a file

ex: pypy lsh.py -w docword.enron.txt -d 12 -k 10 -l 500000 -r 6 -n 100
(to find documents k=10 documents similar to document 12, reading in 500000 lines, using 6 rows/band and 100 hashes in the signature matrix)
"""

from __future__ import division
import sys
import random
import marshal
import argparse
import heapq
from collections import defaultdict
import time

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

def create_band_hashes(matrix, r):
    """ Do LSH on matrix using b bands (n/r bands) """
    hashes = []
    for i in xrange(0, len(matrix), r):
        rows = matrix[i:i+r]
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

def get_candidate_set(hashes, ref_doc_id):
    """ returns a set of all candidate pairs of given doc_id with the data in the given dictionaries
    """
    candidates_set = set()
    for dict in hashes:        # loop through all dictionaries in hashes
        for key, s in dict.iteritems():   
            if ref_doc_id in s:
                for doc_id in s:  
                    candidates_set.add(doc_id)   
    return candidates_set

def find_k_neighbors_of_set(k, ref_doc_id, s, docs):
    ''' returns k nearest neighbors of a given set and parameter k 
    '''
    nearest_neighbors = []
    for neighbor in s:
        if neighbor != ref_doc_id:
            jaccard_compare = compute_jaccard(ref_doc_id, neighbor, docs)
            if len(nearest_neighbors) < k:
                nearest_neighbors.append((jaccard_compare, neighbor))
                if len(nearest_neighbors) == k:
                    heapq.heappush(nearest_neighbors, (jaccard_compare, neighbor))
                    heapq.heappop(nearest_neighbors)
    return nearest_neighbors

def lsh_k_neighbors(matrix, k, b, ref_doc_id, docs):
    ''' call a series of helper function to return k nearest neighbor of a given doc_id and signature matrix
    '''
    nearest_neighbors = []
    starth = time.time()
    hashes = create_band_hashes(matrix, b)
    endh = time.time()
    print "Band hashes computed in {0}ms".format(endh-starth)

    startc = time.time()
    candidates_set = get_candidate_set(hashes, ref_doc_id)
    nearest_neighbors = find_k_neighbors_of_set (k, ref_doc_id, candidates_set, docs)
    while len(nearest_neighbors) < k:
        # put some documents in there at random if k docs not found
        rand_doc = random.randint(1, len(matrix))
        j = compute_jaccard(ref_doc_id, rand_doc, docs)
        nearest_neighbors.append((j, rand_doc))
    endc = time.time()
    print "Found nearest neighbors in {0}ms using LSH".format(endc-startc)

    return nearest_neighbors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--words', required=True, help='File containing document word counts')
    parser.add_argument('-l', '--lines', required=False, help='Number of lines to read from the words file')
    parser.add_argument('-n', '--number_of_hashes', required=False, help='Number of has functions to include in the signature matrix')
    parser.add_argument('-d', '--document1', required=True, help='First document ID to compare')
    parser.add_argument('-o', '--matrix_output', required=False, help='Filename to dump matrix to')
    parser.add_argument('-m', '--matrix_file', required=False, help='Filename of saved signature matrix')
    parser.add_argument('-k', '--k', required=True, help='Number of nearest neighbors to find')
    parser.add_argument('-r', '--number_of_rows_in_band', required = True, help ='Number of bands for LSH approach')

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

    # brute force nearest neighbors
    doc_id_1 = int(args.document1)
    k = int(args.k)
    r = int(args.number_of_rows_in_band)
    sys.stdout.write("\nFinding nearest {0} neighbors by brute force...".format(k))
    sys.stdout.flush()
    start = time.time()
    brute_force_neighbors = brute_force_nearest_neighbors(k, doc_id_1, docs)
    end = time.time()
    sys.stdout.write(" done in {0}ms\n\nFound:\n".format(end-start))
    for neighbor in brute_force_neighbors:
        print "Document ID: {0}, Similarity: {1}".format(neighbor[1], neighbor[0])
    print ""
    # print "Finding {0} nearest neighbors by brute force for docement: {1}.".format(str(brute_force_neighbors), doc_id_1)
    
    print "Finding average jaccard similarity among all documents by brute force..."
    jaccard_all = brute_force_jaccard_all(k, docs)
    print "Average jaccard similarity is {0}\n".format(jaccard_all)

    # call k neighbors with LSH approach
    sys.stdout.write("Finding nearest {0} neighbors by LSH...".format(k))
    sys.stdout.flush()
    lsh_neighbors = lsh_k_neighbors(matrix, k,r, doc_id_1, docs)
    print "\nFound:"
    for neighbor in lsh_neighbors:
        print "Document ID: {0}, Similarity: {1}".format(neighbor[1], neighbor[0])
    # print "Finding {0} nearest neighbors by LSH approach".format(str(lsh_neighbors))

if __name__ == '__main__':
    main()