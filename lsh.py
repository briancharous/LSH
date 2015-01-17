from __future__ import division
import sys

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

def compute_jaccard(doc_id_1, doc_id_2, docs):
    d1 = docs[doc_id_1]
    d2 = docs[doc_id_2]
    return len(d1 & d2)/len(d1 | d2)

def main():
    num_docs = None
    if sys.argv[2] == 'all':
        num_docs = sys.maxint
    else:
        num_docs = int(sys.argv[2])
    docs = get_data(sys.argv[1], num_docs)
    print compute_jaccard(1, 108, docs)

if __name__ == '__main__':
    main()