# Retrieval

Retrieval is the first stage of a RAG system: given a query, find the passages
from the corpus most likely to contain the answer. Most RAG failures are
retrieval failures — if the right passage never reaches the model, no amount of
prompting will recover the answer.

A retriever scores every chunk against the query and returns the top-k. Sparse
retrievers such as TF-IDF and BM25 match on overlapping terms and are fast,
transparent, and need no model. Dense retrievers embed query and chunk into the
same vector space and match on cosine similarity, capturing meaning beyond exact
words. Hybrid retrieval fuses both, typically with reciprocal-rank fusion, and a
cross-encoder reranker can reorder the shortlist for higher precision.

Chunk size is a trade-off. Chunks that are too large dilute the relevance signal
and waste context budget; chunks that are too small lose the surrounding meaning
needed to answer. Splitting on paragraph boundaries is a reasonable default that
keeps each chunk a coherent, citable unit.
