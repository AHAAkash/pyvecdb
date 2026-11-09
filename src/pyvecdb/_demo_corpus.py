"""Bundled sample corpus used by the demo and retrieval evaluation."""

CORPUS = {'docs': [{'id': 'hnsw',
           'title': 'Hierarchical navigable small world graphs',
           'text': 'HNSW organises vectors in a stack of proximity graphs. The bottom layer '
                   'contains every vector, and each higher layer holds a random, exponentially '
                   'smaller subset. A query enters at the top layer, walks greedily towards the '
                   'closest node, then drops one layer and repeats. On the bottom layer it keeps a '
                   'beam of the best candidates, controlled by the ef parameter. A larger ef '
                   'raises recall but costs time. The parameter M limits how many links each node '
                   'keeps. Choosing neighbours with a diversity heuristic keeps long links between '
                   'clusters, which prevents the graph from splitting into islands.'},
          {'id': 'ivf',
           'title': 'Inverted file indexes',
           'text': 'An inverted file index, or IVF, partitions the vectors with k-means. Each '
                   'vector is stored in the list of its nearest centroid. At query time only the '
                   'nprobe closest lists are scanned exhaustively, so most of the data is skipped. '
                   'Raising nprobe improves recall at the price of speed. IVF trains quickly and '
                   'uses little memory beyond the vectors, but it must be trained on '
                   'representative data first. Vectors that sit near a partition boundary can be '
                   'missed when nprobe is small, which is the main source of its errors.'},
          {'id': 'bm25',
           'title': 'BM25 keyword ranking',
           'text': 'BM25 is a lexical ranking function from the probabilistic retrieval family. A '
                   'document scores higher when it contains rare query terms many times. Term '
                   'frequency saturates through the k1 parameter, so repeating a word ten times '
                   'does not count ten times. The b parameter normalises for document length so '
                   'long documents are not favoured. The inverse document frequency of a term is '
                   'large when few documents contain it. BM25 needs no training and excels at '
                   'exact identifiers, names and error codes that embeddings often blur.'},
          {'id': 'rrf',
           'title': 'Reciprocal rank fusion',
           'text': 'Reciprocal rank fusion merges several ranked lists into one. Each document '
                   'receives the sum of one divided by a constant plus its rank in every list, and '
                   'the constant is usually sixty. Because it only uses ranks, it does not require '
                   'calibrating cosine similarities against BM25 scores, which live on different '
                   'scales. Documents that appear near the top of several lists rise to the front. '
                   'RRF is a common way to build hybrid search that combines keyword and vector '
                   'retrieval.'},
          {'id': 'cosine',
           'title': 'Cosine similarity and normalisation',
           'text': 'Cosine similarity measures the angle between two vectors and ignores their '
                   'length. When all vectors are scaled to unit length, cosine distance equals one '
                   'minus the dot product, and ranking by it is identical to ranking by Euclidean '
                   'distance. Normalising once at insertion time therefore lets an index use fast '
                   'matrix multiplication. Pruning rules that assume a true metric, such as the '
                   'bound used by a KD-tree, do not hold for raw cosine distance, which is why '
                   'such trees return wrong neighbours with it.'},
          {'id': 'chunking',
           'title': 'Chunking documents for retrieval',
           'text': 'Long documents are split into chunks before embedding, because one vector '
                   'cannot represent many topics well. Splitting on sentence boundaries keeps each '
                   'chunk readable. Overlapping neighbouring chunks by a few sentences means an '
                   'answer that crosses a boundary still appears whole in at least one chunk. '
                   'Chunks that are too small lose context, while chunks that are too large dilute '
                   'the embedding and waste the prompt budget. A budget of a few hundred words is '
                   'a typical starting point.'},
          {'id': 'rag',
           'title': 'Retrieval augmented generation',
           'text': 'Retrieval augmented generation answers a question by first fetching relevant '
                   'passages and then asking a language model to respond using them. Grounding the '
                   'answer in retrieved text reduces invented facts and lets the system cite '
                   'sources. A good prompt tells the model to answer only from the excerpts, to '
                   'cite them by number and to say it does not know when the excerpts are '
                   'insufficient. Retrieved documents are untrusted input and may contain '
                   'instructions, so the prompt must tell the model to ignore them.'},
          {'id': 'embeddings',
           'title': 'Text embeddings',
           'text': 'A text embedding model maps a passage to a dense vector so that passages with '
                   'similar meaning land close together. Unlike keyword matching, embeddings can '
                   'connect a question to a passage that shares no words with it. Some retrieval '
                   'models expect a task prefix, one for queries and another for documents. The '
                   'vector dimension, often between 384 and 1536, determines memory use. '
                   'Embeddings from different models are not comparable, so an index must be '
                   'rebuilt whenever the model changes.'},
          {'id': 'pq',
           'title': 'Product quantization',
           'text': 'Product quantization compresses vectors by splitting each into several '
                   'sub-vectors and replacing every sub-vector with the id of its nearest codeword '
                   'from a small learned codebook. A vector of hundreds of floats shrinks to a '
                   'handful of bytes. Distances are then estimated from lookup tables rather than '
                   'from the original numbers. The compression is lossy, so quantised indexes '
                   'usually rerank the best candidates with the exact vectors. It trades some '
                   'accuracy for a large reduction in memory.'},
          {'id': 'metrics',
           'title': 'Measuring retrieval quality',
           'text': 'Recall at k is the fraction of the true nearest neighbours that appear in the '
                   'top k results, and it is the standard way to judge an approximate index '
                   'against exact search. For ranked answers, hit rate at k asks whether a '
                   'relevant document appears in the top k, and mean reciprocal rank averages one '
                   'over the rank of the first relevant result. Recall should always be reported '
                   'together with speed, since any index can reach perfect recall by scanning '
                   'everything.'},
          {'id': 'deletes',
           'title': 'Deleting from graph indexes',
           'text': 'Removing a node from a proximity graph can cut the only path to other nodes. A '
                   'safer approach is a tombstone: the node is flagged as deleted, still helps '
                   'route queries, but is never returned. If many tombstones accumulate, search '
                   'must widen its beam to find enough live results. Periodically rebuilding the '
                   'index without the dead nodes, called compaction, reclaims the space and '
                   'restores speed.'},
          {'id': 'mmr',
           'title': 'Maximal marginal relevance',
           'text': 'Maximal marginal relevance picks results one at a time, scoring each candidate '
                   'by its relevance to the query minus its similarity to results already chosen. '
                   'A weight lambda sets the balance, where one means pure relevance and lower '
                   'values favour diversity. It is useful when the top results are near '
                   'duplicates, for example overlapping chunks of the same page, because it '
                   'spreads the context across more distinct passages.'}],
 'questions': [{'q': 'How does a search in HNSW begin and proceed through the layers?',
                'doc': 'hnsw',
                'answerable': True},
               {'q': 'Why does the neighbour selection heuristic stop the graph from splitting '
                     'into islands?',
                'doc': 'hnsw',
                'answerable': True},
               {'q': 'What does nprobe control in an inverted file index?',
                'doc': 'ivf',
                'answerable': True},
               {'q': 'Why can an IVF index miss vectors near a partition boundary?',
                'doc': 'ivf',
                'answerable': True},
               {'q': 'What does the b parameter do in BM25?', 'doc': 'bm25', 'answerable': True},
               {'q': 'Why is keyword ranking good at finding error codes and exact names?',
                'doc': 'bm25',
                'answerable': True},
               {'q': 'How is the fused score computed when merging ranked lists?',
                'doc': 'rrf',
                'answerable': True},
               {'q': 'Why does rank fusion avoid calibrating BM25 against cosine similarity?',
                'doc': 'rrf',
                'answerable': True},
               {'q': 'How does cosine distance relate to the dot product for unit vectors?',
                'doc': 'cosine',
                'answerable': True},
               {'q': 'Why does a KD-tree return wrong neighbours with cosine distance?',
                'doc': 'cosine',
                'answerable': True},
               {'q': 'Why overlap neighbouring chunks when splitting a long document?',
                'doc': 'chunking',
                'answerable': True},
               {'q': 'What goes wrong if chunks are too large?',
                'doc': 'chunking',
                'answerable': True},
               {'q': 'What should a grounded prompt tell the model to do when excerpts are '
                     'insufficient?',
                'doc': 'rag',
                'answerable': True},
               {'q': 'Why must retrieved documents be treated as untrusted input?',
                'doc': 'rag',
                'answerable': True},
               {'q': 'Why must an index be rebuilt when the embedding model changes?',
                'doc': 'embeddings',
                'answerable': True},
               {'q': 'Can embeddings match a question to a passage that shares no words with it?',
                'doc': 'embeddings',
                'answerable': True},
               {'q': 'How does product quantization shrink a vector to a few bytes?',
                'doc': 'pq',
                'answerable': True},
               {'q': 'Why do quantised indexes rerank candidates with exact vectors?',
                'doc': 'pq',
                'answerable': True},
               {'q': 'What is recall at k for an approximate index?',
                'doc': 'metrics',
                'answerable': True},
               {'q': 'Why should recall be reported together with speed?',
                'doc': 'metrics',
                'answerable': True},
               {'q': 'Why is a tombstone safer than removing a node from the graph?',
                'doc': 'deletes',
                'answerable': True},
               {'q': 'What is compaction and when is it needed?',
                'doc': 'deletes',
                'answerable': True},
               {'q': 'What does the lambda weight control in maximal marginal relevance?',
                'doc': 'mmr',
                'answerable': True},
               {'q': 'When is diversity useful for the retrieved passages?',
                'doc': 'mmr',
                'answerable': True},
               {'q': 'What is the boiling point of tungsten?', 'doc': None, 'answerable': False},
               {'q': 'Who won the 1998 football world cup final?',
                'doc': None,
                'answerable': False},
               {'q': 'How do I bake sourdough bread at home?', 'doc': None, 'answerable': False}]}
