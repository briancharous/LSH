import sys

def get_data(filename, lines):
    docs = dict()
    with open(filename, 'r') as f:
        for i in range(3):
            f.next() # skip first 3 lines
        # f.readlines(0)
        linenum = 0
        for line in f:
            if linenum == lines:
                break
            components = line.split()
            if components[0] in docs:
                docs[components[0]].add(components[1])
            else:
                docs[components[0]] = set([components[1]])
            linenum+=1
    return docs


def main():
    print get_data(sys.argv[1], int(sys.argv[2]))

if __name__ == '__main__':
    main()