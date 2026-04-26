"""BERTopic quality metrics: coherence, embedding separation, and topic diversity.

Coherence metrics use Gensim's :class:`~gensim.models.coherencemodel.CoherenceModel`.
Documents are tokenized with the *fitted* BERTopic
:class:`~sklearn.feature_extraction.text.CountVectorizer` analyzer
(``topic_model.vectorizer_model.build_analyzer()``) so evaluation stays aligned
with the vocabulary and n-gram settings used for c-TF-IDF topic terms.

Optional corpus caching (see ``cache_dir``) stores tokenized texts, a Gensim
:class:`~gensim.corpora.Dictionary`, and bag-of-words rows; the cache key
includes input texts, ``document_representation``, and the fitted vectorizer
state so it invalidates when documents or representation mode change.
"""

import json
import os
import pickle
import hashlib
from datetime import datetime
from typing import Any

import numpy as np
from bertopic import BERTopic
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import silhouette_score


class BerTopicEvaluator:
    """Evaluate a fitted :class:`~bertopic.BERTopic` model with several metrics.

    Metrics returned by :meth:`evaluate` (higher is better unless noted):

    coherence_c_v
        Semantic coherence: how often the top topic words co-occur in sliding
        windows over the corpus, normalized with NPMI-like terms. Measures
        whether topic words "read" as a single theme in context.
    coherence_u_mass
        Coherence from document-level word co-occurrence under a topic-word
        distribution (Gensim's *u_mass*). Uses the corpus BoW; often more
        sensitive to corpus frequency than *c_v*.
    coherence_c_npmi
        Normalized pointwise mutual information between topic word pairs in
        corpus windows; bounded and comparable across runs. Emphasizes
        statistical association of top terms.
    silhouette_score
        Cluster separation in the **same embedding space** used for BERTopic
        (rows excluded if topic label is ``-1``). High values mean assigned
        topics form compact, well-separated groups in that space. Does **not**
        measure lexical quality of topic words; can be misleading if clusters
        overlap or outliers dominate.
    topic_diversity
        Share of **distinct** terms across all topics' top words divided by
        total top-word slots (after filtering to the vocabulary used for
        coherence). ``1.0`` means no overlap between topics; ``0.0`` means all
        top words are repeats. Rewards topic distinctness, not interpretability.

    Parameters
    ----------
    topic_model
        Fitted BERTopic instance (must have run ``fit`` / ``fit_transform``).
    texts
        Same document strings passed to BERTopic for that fit (same order and
        length as embedding rows).
    topics
        Topic id per row from BERTopic (same length as ``texts``).
    embeddings
        Matrix passed into BERTopic for clustering (same row order as
        ``texts`` and ``topics``).
    cache_dir
        If set, read/write pickled coherence corpus cache under this directory.
    document_representation
        Label for cache keying, e.g. ``full_text`` or ``chunks``, so caches do
        not collide across representation modes.
    """

    def __init__(
        self,
        topic_model: BERTopic,
        texts: list[str],
        topics: np.ndarray,
        embeddings: np.ndarray,
        cache_dir: str | None = None,
        document_representation: str | None = None,
        model_id: str | None = None,
    ):
        self._topic_model = topic_model
        self._texts = texts
        self._topics = np.asarray(topics)
        self._embeddings = embeddings
        self._cache_dir = cache_dir
        self._document_representation = document_representation
        self._model_id = model_id

    def evaluate(self, top_n_words: int = 10) -> dict[str, Any]:
        """Compute coherence, silhouette, and topic-diversity metrics.

        Parameters
        ----------
        top_n_words
            Number of highest-weight terms per topic used for coherence and
            diversity (outlier topic ``-1`` is excluded).

        Returns
        -------
        dict
            Keys include ``coherence_c_v``, ``coherence_u_mass``,
            ``coherence_c_npmi``, ``silhouette_score`` (or ``None`` if not
            defined), ``topic_diversity``, ``top_n_words``, and ``topics_count``.
            See the class docstring for what each score measures.
        """
        tokenized_texts, dictionary, corpus = self._load_or_create_corpus_cache()
        topics_words = self._extract_topic_words(dictionary, top_n_words)

        metrics = {
            "coherence_c_v": self._compute_coherence(
                topics_words=topics_words,
                tokenized_texts=tokenized_texts,
                dictionary=dictionary,
                corpus=corpus,
                coherence_type="c_v",
            ),
            "coherence_u_mass": self._compute_coherence(
                topics_words=topics_words,
                tokenized_texts=tokenized_texts,
                dictionary=dictionary,
                corpus=corpus,
                coherence_type="u_mass",
            ),
            "coherence_c_npmi": self._compute_coherence(
                topics_words=topics_words,
                tokenized_texts=tokenized_texts,
                dictionary=dictionary,
                corpus=corpus,
                coherence_type="c_npmi",
            ),
            "silhouette_score": self._compute_silhouette_score(),
            "topic_diversity": self._compute_topic_diversity(topics_words),
            "top_n_words": top_n_words,
            "topics_count": len(topics_words),
            "outliers_count": int(np.sum(self._topics == -1)),
            "outliers_ratio": float(np.mean(self._topics == -1)),
            "documents_count": int(self._topics.shape[0]),
        }

        return metrics

    def save(self, results: dict[str, Any], output_path: str, metadata: dict[str, Any]) -> None:
        """Write ``results`` and ``metadata`` as JSON (adds ``created_at`` UTC)."""
        payload = {
            **results,
            **metadata,
            "model_id": self._model_id,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    def _tokenize_texts(self) -> list[list[str]]:
        """Tokenize each document with the BERTopic vectorizer analyzer."""
        analyzer = self._topic_model.vectorizer_model.build_analyzer()

        tokenized_texts = []
        for text in self._texts:
            if not isinstance(text, str) or not text.strip():
                continue
            tokens = analyzer(text)
            if tokens:
                tokenized_texts.append(tokens)

        if len(tokenized_texts) == 0:
            raise ValueError("No valid documents available to evaluate BERTopic metrics.")

        return tokenized_texts

    def _extract_topic_words(
        self, dictionary: corpora.Dictionary, top_n_words: int
    ) -> list[list[str]]:
        """Top terms per topic, restricted to tokens present in ``dictionary``."""
        topic_info = self._topic_model.get_topic_info()
        candidate_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()
        topics_words = []
        for topic_id in candidate_topics:
            topic_terms = self._topic_model.get_topic(topic_id)
            if not topic_terms:
                continue
            words = [
                word
                for word, _ in topic_terms[:top_n_words]
                if word in dictionary.token2id
            ]
            if words:
                topics_words.append(words)

        if len(topics_words) == 0:
            raise ValueError("No valid BERTopic topics found to evaluate.")

        return topics_words

    def _compute_coherence(
        self,
        topics_words: list[list[str]],
        tokenized_texts: list[list[str]],
        dictionary: corpora.Dictionary,
        corpus: list[list[tuple[int, int]]],
        coherence_type: str,
    ) -> float:
        """Single Gensim coherence metric (``c_v``, ``u_mass``, or ``c_npmi``)."""
        coherence_model = CoherenceModel(
            topics=topics_words,
            texts=tokenized_texts,
            corpus=corpus,
            dictionary=dictionary,
            coherence=coherence_type,
        )
        return float(coherence_model.get_coherence())

    def _compute_silhouette_score(self) -> float | None:
        """Mean silhouette of non-outlier points in ``embeddings`` by topic id."""
        labels = self._topics
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]

        if valid_labels.size < 2:
            return None

        if np.unique(valid_labels).size < 2:
            return None

        valid_embeddings = self._embeddings[valid_mask]
        if valid_embeddings.shape[0] < 2:
            return None

        return float(silhouette_score(valid_embeddings, valid_labels))

    def _compute_topic_diversity(self, topics_words: list[list[str]]) -> float:
        """Ratio of unique top terms to total top-term slots across topics."""
        total_words = sum(len(topic_words) for topic_words in topics_words)
        if total_words == 0:
            return 0.0

        unique_words = len({word for topic_words in topics_words for word in topic_words})
        return float(unique_words / total_words)

    def _load_or_create_corpus_cache(
        self,
    ) -> tuple[list[list[str]], corpora.Dictionary, list[list[tuple[int, int]]]]:
        """Load cached tokenized texts/dictionary/corpus or create and cache them."""
        cache_file_path = self._get_cache_file_path()
        if cache_file_path and os.path.exists(cache_file_path):
            with open(cache_file_path, "rb") as fp:
                cached = pickle.load(fp)
                return (
                    cached["tokenized_texts"],
                    cached["dictionary"],
                    cached["corpus"],
                )

        tokenized_texts = self._tokenize_texts()
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]

        if cache_file_path:
            with open(cache_file_path, "wb") as fp:
                pickle.dump(
                    {
                        "tokenized_texts": tokenized_texts,
                        "dictionary": dictionary,
                        "corpus": corpus,
                    },
                    fp,
                )

        return tokenized_texts, dictionary, corpus

    def _get_cache_file_path(self) -> str | None:
        if self._cache_dir is None:
            return None

        os.makedirs(self._cache_dir, exist_ok=True)
        cache_key = self._build_cache_key()
        return os.path.join(self._cache_dir, f"coherence_corpus_{cache_key}.pkl")

    def _build_cache_key(self) -> str:
        """Hash texts, document representation mode, and fitted vectorizer state."""
        vectorizer_model = self._topic_model.vectorizer_model
        params = vectorizer_model.get_params(deep=True)
        vocabulary = getattr(vectorizer_model, "vocabulary_", {})

        serializable_payload = {
            "texts": self._texts,
            "document_representation": self._document_representation,
            "vectorizer_params": params,
            "vectorizer_vocabulary": vocabulary,
        }
        payload_str = json.dumps(serializable_payload, sort_keys=True, default=str)
        return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:16]
