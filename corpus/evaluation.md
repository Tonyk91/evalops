# Evaluation

Evaluation turns "the demo looked good" into a number you can defend. For a RAG
system the metrics fall into two families: retrieval quality and answer quality.

Retrieval metrics ask whether the right document reached the top-k. Hit rate is
the fraction of queries with at least one relevant document retrieved. Mean
reciprocal rank rewards placing the relevant document higher. Recall at k is the
fraction of all relevant documents that were retrieved. These are computed
against a gold set of query-to-document labels.

Answer metrics ask whether the response is trustworthy. Groundedness, or
faithfulness, measures whether the answer is supported by the cited context
rather than invented. Answer rate and gate accuracy measure whether the system
answers when it should and refuses when the evidence is too weak.

An evaluation is only as good as its gold set. Each test case pairs a query with
the documents that should answer it, and out-of-corpus queries are included
deliberately so that refusal behaviour is measured, not assumed.
