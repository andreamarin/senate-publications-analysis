import os
import spacy
import pathlib
import numpy as np
import pandas as pd
from umap import UMAP
from enum import Enum
from typing import Union
from hdbscan import HDBSCAN
from bertopic import BERTopic
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from nlp_classification.utils.bertopic_evaluator import BerTopicEvaluator


class DocumentRepresentation(str, Enum):
    """Strategy for turning each document into one or more embedding rows.

    Chunking modes split text with spaCy, embed segments, then either aggregate
    back to one vector per source document (pooling) or keep one vector per
    chunk. ``FULL_TEXT`` embeds each document as a single string.

    Attributes
    ----------
    MEAN_POOLING : DocumentRepresentation
        Embed chunks, then take the element-wise mean within each document.
    MAX_POOLING : DocumentRepresentation
        Embed chunks, then take the element-wise maximum within each document.
    CHUNKS : DocumentRepresentation
        One embedding per chunk; BERTopic receives multiple rows per document.
    FULL_TEXT : DocumentRepresentation
        No chunking; one embedding row per input document.
    """
    MEAN_POOLING = "mean_pooling"
    MAX_POOLING = "max_pooling"
    CHUNKS = "chunks"
    FULL_TEXT = "full_text"


@dataclass
class UMAPConfig:
    """Parameters for ``umap.UMAP`` used to reduce embedding dimensionality.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors for the UMAP graph.
    metric : str, default='cosine'
        Distance metric passed to ``umap.UMAP``
    min_dist: float, default=0.1
        Minimum distance between points in the UMAP embedding.
    random_state: int, default=20260415
        Random seed for reproducibility.
    n_components: int, default=10
        Number of components in the UMAP embedding.
    """
    n_neighbors: int
    metric: str = 'cosine'
    min_dist: float = 0.1
    random_state: int = 20260415
    n_components: int = 10


@dataclass
class HDBSCANConfig:
    """Parameters for ``hdbscan.HDBSCAN`` used to cluster embeddings.

    Attributes
    ----------
    min_cluster_size: int
        Minimum number of samples in a cluster.
    metric: str, default='euclidean'
        Distance metric passed to ``hdbscan.HDBSCAN``.
    cluster_selection_method: str, default='eom'
        Method for selecting clusters.
    """
    min_cluster_size: int
    metric: str = 'euclidean'


@dataclass
class EmbeddingConfig:
    """Configuration for sentence embeddings, chunking, and on-disk caches.

    Attributes
    ----------
    embedding_model : str
        Name or path for :class:`sentence_transformers.SentenceTransformer`.
    max_words : int
        Soft limit on words per chunk when chunk-based modes are active.
    spacy_model : str
        spaCy pipeline name for sentence segmentation and chunking.
    document_representation : DocumentRepresentation or str
        How documents map to embedding rows; invalid values mapped to
        ``FULL_TEXT``.
    force_compute : bool, default=False
        If True, recompute embeddings even when a cache file exists.
    embeddings_file : str
        Set in ``__post_init__``: filename for the ``.npy`` embedding matrix.
    chunks_file : str, optional
        Set only when ``document_representation`` is not ``FULL_TEXT``:
        pickle filename for chunk strings.
    doc_ids_file : str, optional
        Set only when ``document_representation`` is not ``FULL_TEXT``:
        ``.npy`` filename for the chunk-to-document id map.

    Notes
    -----
    Chunk-related paths are created only for non-``FULL_TEXT`` modes; see
    :class:`BerTopicModelBuilder`.
    """
    embedding_model: str
    max_words: int
    spacy_model: str
    document_representation: Union[DocumentRepresentation, str]
    force_compute: bool = False

    def __post_init__(self) -> None:
        """Normalize ``document_representation`` and set cache filenames."""
        dr = self.document_representation

        if isinstance(dr, str):
            try:
                self.document_representation = DocumentRepresentation(dr)
            except ValueError:
                self.document_representation = DocumentRepresentation.FULL_TEXT

        if self.document_representation is not DocumentRepresentation.FULL_TEXT:
            self.embeddings_file = f"embeddings_{self.document_representation.value}_{self.max_words}.npy"
            self.chunks_file = f"chunks_{self.max_words}.pkl"
            self.doc_ids_file = f"doc_ids_{self.max_words}.npy"
        else:
            self.embeddings_file = f"embeddings_{self.document_representation.value}.npy"


class BerTopicModelBuilder:
    """Build a :class:`bertopic.BERTopic` model with optional embedding caches.

    Loads or computes sentence-transformer embeddings (with optional
    sentence-level chunking and pooling), then fits BERTopic using precomputed
    vectors. Output paths are rooted at ``nlp_classification/<folder_name>/``.

    Parameters
    ----------
    texts_df : pandas.DataFrame
        Input table; each row is treated as one logical document.
    text_column : str
        Column in ``texts_df`` containing raw text.
    folder_name : str
        Subdirectory under the package ``nlp_classification`` folder used for
        ``coherence_scores/``, ``models/embeddings/``, and ``models/text_chunks/``.
    embedding_config : EmbeddingConfig
        Encoder id, document representation mode, and cache filenames.
    umap_config : UMAPConfig
        UMAP settings kept on the instance as ``umap_model`` (for consumers that
        pass a custom BERTopic pipeline).
    verbose : bool, default=False
        If True, show encoding progress from ``SentenceTransformer``.

    Attributes
    ----------
    umap_model : umap.UMAP
        UMAP instance built from ``umap_config``.
    topic_model : bertopic.BERTopic
        Created in :meth:`fit_transform`.
    topics : numpy.ndarray
        Topic assignment per row passed to BERTopic (document or chunk), set by
        :meth:`fit_transform`.
    probs : numpy.ndarray or None
        Topic probabilities when returned by BERTopic, set by :meth:`fit_transform`.
    embeddings : numpy.ndarray
        Matrix passed to ``BERTopic.fit_transform``, set by :meth:`_load_embeddings`.
    """
    def __init__(
        self,
        texts_df: pd.DataFrame,
        text_column: str,
        folder_name: str,
        embedding_config: EmbeddingConfig,
        umap_config: UMAPConfig,
        hdbscan_config: HDBSCANConfig,
        verbose: bool = False
    ):
        self._document_texts = texts_df[text_column].tolist()
        self._verbose = verbose

        current_path = pathlib.Path(__file__).parent.resolve()
        base_path = current_path.parent.resolve()

        self._images_path = f"{base_path}/{folder_name}/coherence_scores"
        self._models_path = f"{base_path}/{folder_name}/models"
        self._embeddings_path = f"{self._models_path}/embeddings"
        self._chunks_path = f"{self._models_path}/text_chunks"
        self._evaluation_cache_path = f"{self._models_path}/evaluation_cache"
        self._coherence_score_path = f"{self._images_path}/bertopic_coherence_scores.json"

        self._init_folders()

        self._ec = embedding_config

        # save umap model config
        self.umap_model = UMAP(
            n_neighbors=umap_config.n_neighbors,
            metric=umap_config.metric
        )

        # save hdbscan model config
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=hdbscan_config.min_cluster_size,
            metric=hdbscan_config.metric,
            prediction_data=False
        )
        self.evaluation_results = None

    def _init_folders(self) -> None:
        """Create coherence, model, embedding, and chunk directories if missing."""
        os.makedirs(self._images_path, exist_ok=True)
        os.makedirs(self._models_path, exist_ok=True)
        os.makedirs(self._embeddings_path, exist_ok=True)
        os.makedirs(self._chunks_path, exist_ok=True)
        os.makedirs(self._evaluation_cache_path, exist_ok=True)
        
    def _sentence_chunking(self, text, nlp):
        """Split ``text`` into sentence-based chunks up to ``max_words`` each.

        Parameters
        ----------
        text : str
            Source document.
        nlp : spacy.language.Language
            Loaded spaCy pipeline with sentence boundaries.

        Returns
        -------
        list of str
            Chunk strings in order; may be empty if ``text`` yields no tokens.
        """
        doc = nlp(text)

        chunks = []
        current_chunk = []
        current_len = 0

        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_len = len(sent_text.split())

            if current_len + sent_len > self._ec.max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0

            current_chunk.append(sent_text)
            current_len += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _load_chunks(self) -> None:
        """Load or build chunk texts and per-chunk document ids from disk cache."""
        chunks_full_path = os.path.join(self._chunks_path, self._ec.chunks_file)
        doc_ids_full_path = os.path.join(self._chunks_path, self._ec.doc_ids_file)

        if os.path.exists(chunks_full_path) and os.path.exists(doc_ids_full_path):
            self._texts = pd.read_pickle(chunks_full_path)
            self._doc_ids = np.load(doc_ids_full_path)
        else:
            chunk_texts = []
            doc_ids = []

            nlp = spacy.load(self._ec.spacy_model)

            for doc_id, text in enumerate(self._document_texts):
                chunks = self._sentence_chunking(text, nlp)

                chunk_texts.extend(chunks)
                doc_ids.extend([doc_id] * len(chunks))

            self._texts = chunk_texts
            self._doc_ids = np.asarray(doc_ids, dtype=np.int64)

            pd.to_pickle(self._texts, chunks_full_path)
            np.save(doc_ids_full_path, self._doc_ids)

    def _pool_document_embeddings(
        self, chunk_embeddings: np.ndarray, mode: str
    ) -> np.ndarray:
        """Reduce chunk rows to one vector per document using mean or max.

        Uses ``self._doc_ids`` to group rows of ``chunk_embeddings``.

        Parameters
        ----------
        chunk_embeddings : numpy.ndarray, shape (n_chunks, dim)
            Encoder output for each chunk.
        mode : {'mean', 'max'}
            Pooling along the chunk axis.

        Returns
        -------
        numpy.ndarray, shape (n_documents, dim)
            One row per entry in ``self._document_texts``. Rows with no chunks
            remain zero.
        """
        n_docs = len(self._document_texts)
        dim = chunk_embeddings.shape[1]
        out = np.zeros((n_docs, dim), dtype=chunk_embeddings.dtype)
        for doc_id in range(n_docs):
            mask = self._doc_ids == doc_id
            rows = chunk_embeddings[mask]
            if rows.shape[0] == 0:
                continue
            if mode == "mean":
                out[doc_id] = rows.mean(axis=0)
            else:
                out[doc_id] = rows.max(axis=0)
        return out

    def _load_embeddings(self) -> None:
        """Load cached ``.npy`` embeddings or encode texts and save.

        Sets ``self.embeddings`` and aligns ``self._texts`` with the embedding
        rows (per document for ``FULL_TEXT`` and pooling modes, per chunk for
        ``CHUNKS``). When the cache is hit, chunk lists are refreshed only for
        ``CHUNKS`` so ``self._texts`` matches the matrix width.
        """
        embeddings_file_path = os.path.join(self._embeddings_path, self._ec.embeddings_file)

        if os.path.exists(embeddings_file_path) and not self._ec.force_compute:
            self.embeddings = np.load(embeddings_file_path)

            if self._ec.document_representation is DocumentRepresentation.CHUNKS:
                # load chunks as the list of texts
                self._load_chunks()
            else:
                self._texts = list(self._document_texts)

            return
        
        embedding_model = SentenceTransformer(self._ec.embedding_model)

        if self._ec.document_representation is not DocumentRepresentation.FULL_TEXT:
            # create chunks for sentences
            self._load_chunks()

            # encode the chunks
            chunk_embeddings = embedding_model.encode(
                self._texts, show_progress_bar=self._verbose
            )
            if self._ec.document_representation is DocumentRepresentation.MEAN_POOLING:
                # pool the embeddings by the mean and set the texts to the original documents
                self.embeddings = self._pool_document_embeddings(chunk_embeddings, "mean")
                self._texts = list(self._document_texts)
            elif self._ec.document_representation is DocumentRepresentation.MAX_POOLING:
                # pool the embeddings by the max and set the texts to the original documents
                self.embeddings = self._pool_document_embeddings(chunk_embeddings, "max")
                self._texts = list(self._document_texts)
            else:
                self.embeddings = chunk_embeddings
        else:
            self._texts = list(self._document_texts)
            self.embeddings = embedding_model.encode(
                self._texts, show_progress_bar=self._verbose
            )

        np.save(embeddings_file_path, self.embeddings)

    def fit_transform(self):
        """Fit ``BERTopic`` on cached or fresh embeddings.

        Sets ``topic_model``, ``topics``, ``probs``, and delegates embedding
        setup to :meth:`_load_embeddings`.

        Returns
        -------
        topics : numpy.ndarray
            Topic index per row in ``self._texts``.
        probs : numpy.ndarray or None
            Per-document (or per-chunk) topic probabilities when available.
        """
        # 1. load the embeddings
        self._load_embeddings()

        # 2. create the bertopic model
        self.topic_model = BERTopic(umap_model=self.umap_model, hdbscan_model=self.hdbscan_model)

        # fit the data susing the existing embeddings
        self.topics, self.probs = self.topic_model.fit_transform(self._texts, self.embeddings)

        evaluator = BerTopicEvaluator(
            topic_model=self.topic_model,
            texts=self._texts,
            topics=self.topics,
            embeddings=self.embeddings,
            cache_dir=self._evaluation_cache_path,
            document_representation=self._ec.document_representation.value,
        )
        self.evaluation_results = evaluator.evaluate()
        self.coherence_score = self.evaluation_results.get("coherence_c_v")
        evaluator.save(
            results=self.evaluation_results,
            output_path=self._coherence_score_path,
            metadata={
                "document_representation": self._ec.document_representation.value,
            },
        )
