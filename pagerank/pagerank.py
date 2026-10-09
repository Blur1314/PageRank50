import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    dic = {}
    linked = list(corpus[page])
    if not linked:
        prob = 1/len(corpus)
        for pages in corpus.keys():
            dic[pages] = prob
        return dic

    proboflinked = (damping_factor/len(linked) ) + ((1-damping_factor)/len(corpus))
    probofnotlinked = (1-damping_factor)/len(corpus)

    for pagess in corpus.keys():
        if pagess in linked:
            dic[pagess] = proboflinked
        else:
            dic[pagess] = probofnotlinked

    return dic




def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagess = []
    for pages in corpus.keys():
        pagess.append(pages)

    currpage = random.choice(pagess)

    dicc = {}

    for x in corpus.keys():
        dicc[x] = 0

    dicc[currpage] = 1
    for x in range(n-1):
        transdic = transition_model(corpus, currpage, damping_factor)

        pagesss= list(transdic.keys())
        weights = list(transdic.values())

        picked = random.choices(pagesss , weights = weights , k =1 )[0]
        dicc[picked] = dicc[picked] + 1
        currpage = picked


    pgrnk  = {page: count/n for page , count in dicc.items()}
    return pgrnk



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """


    pgrnk = {page: 1 / len(corpus) for page in corpus}

    while True:
        new_pgrnk = {}

        for page in corpus:
            add = 0
            for source in corpus:
                if page in corpus[source]:
                    add += pgrnk[source] / len(corpus[source])
                elif len(corpus[source]) == 0:
                    add += pgrnk[source] / len(corpus)

            new_pgrnk[page] = ((1 - damping_factor) / len(corpus)) + (damping_factor * add)

        # Calculate max change across all pages
        max_change = max(abs(new_pgrnk[p] - pgrnk[p]) for p in corpus)
        pgrnk = new_pgrnk

        if max_change <= 0.001:
            break

    return pgrnk


if __name__ == "__main__":
    main()
