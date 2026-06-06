import os
import json
import logging
import shutil
import spacy
import pathlib
import numpy as np
import pandas as pd
from umap import UMAP
from typing import Optional
from hdbscan import HDBSCAN
from bertopic import BERTopic
from dataclasses import asdict
from scipy.cluster import hierarchy as sch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from .bertopic_config import (
    EmbeddingConfig,
    UMAPConfig,
    HDBSCANConfig,
    DocumentRepresentation,
    ComputeConfig,
    OutlierReductionConfig,
    OutlierReductionStrategy,
    CountVectorizerConfig,
)
from .bertopic_evaluator import BerTopicEvaluator

logger = logging.getLogger(__name__)


def setup_builder_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a stream logger for this module when not configured yet.

    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level to set on the module logger.

    Returns
    -------
    logging.Logger
        Configured module logger.
    """
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


class BerTopicModelBuilder:
    """Build a :class:`bertopic.BERTopic` model with cache-first execution.

    For each expensive stage (chunks, embeddings, model outputs, visualizations,
    evaluation metrics), the builder tries to load on-disk artifacts first and
    only recomputes missing pieces. Output paths are rooted at
    ``nlp_classification/<folder_name>/``.

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
    hdbscan_config : HDBSCANConfig
        HDBSCAN settings kept on the instance as ``hdbscan_model``.
    verbose : bool, default=False
        If True, show encoding progress from ``SentenceTransformer``.
    base_path : str, default=None
        Base path for model storage. If not provided, uses the package root.
    compute_config : ComputeConfig, optional
        Per-step cache bypass flags.
    outlier_reduction_config : OutlierReductionConfig, optional
        When enabled, reassigns HDBSCAN outlier documents (topic ``-1``) after
        the initial fit using :meth:`~bertopic.BERTopic.reduce_outliers`.
    load_merged : bool, default=False
        When True, load and persist artifacts under ``{model_id}__merged``.
    run_suffix : str, default=""
        Additional suffix appended after ``load_merged`` (e.g. outlier-reduction
        variant from :meth:`compute_or_suffix`).

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
    evaluation_results : dict or None
        Evaluation payload computed by :class:`BerTopicEvaluator` after
        :meth:`fit_transform`, including coherence, silhouette, and diversity
        metrics.
    coherence_score : float or None
        Backward-compatible alias for ``evaluation_results['coherence_c_v']``.

    Notes
    -----
    The run directory (keyed by ``model_id``) stores BERTopic fit artifacts:
    fitted model, topics/probabilities, visualizations, and evaluation metrics.
    Shared caches under ``models_cache/`` store embeddings/chunks and evaluator
    corpus caches.
    """
    def __init__(
        self,
        texts_df: pd.DataFrame,
        text_column: str,
        folder_name: str,
        embedding_config: EmbeddingConfig,
        umap_config: UMAPConfig,
        hdbscan_config: HDBSCANConfig,
        countvectorizer_config: CountVectorizerConfig,
        verbose: bool = False,
        base_path: str = None,
        compute_config: Optional[ComputeConfig] = None,
        outlier_reduction_config: Optional[OutlierReductionConfig] = None,
        load_merged: bool = False,
        run_suffix: str = "",
    ):
        self._document_texts = texts_df[text_column].tolist()
        self._text_column = text_column
        self._folder_name = folder_name
        self._verbose = verbose
        if self._verbose:
            setup_builder_logging()

        # Save config objects and build model_id.
        self._ec = embedding_config
        self._umap_config = umap_config
        self._hdbscan_config = hdbscan_config
        self._countvectorizer_config = countvectorizer_config
        self._outlier_reduction_config = outlier_reduction_config or OutlierReductionConfig()
        self._compute_config = compute_config or ComputeConfig()
        self._load_merged = load_merged
        self._run_suffix = run_suffix

        self.model_id = self._build_model_id()
        if load_merged:
            self.model_id += "__merged"
        if run_suffix:
            self.model_id += run_suffix

        current_path = pathlib.Path(__file__).parent.resolve()
        base_path = current_path.parent.resolve() if base_path is None else base_path
        self._resolved_base_path = str(base_path)

        self._base_output_path = f"{base_path}/{folder_name}"
        self._runs_path = f"{self._base_output_path}/runs"
        self._cache_path = f"{self._base_output_path}/models_cache"

        # objects re-used by bertopic or the evaluator
        self._embeddings_path = f"{self._cache_path}/embeddings"
        self._chunks_path = f"{self._cache_path}/text_chunks"
        self._evaluation_cache_path = f"{self._cache_path}/evaluation"


        self._run_path = os.path.join(self._runs_path, self.model_id)
        self._visualizations_path = os.path.join(self._run_path, "visualizations")
        self._evaluation_metrics_path = os.path.join(
            self._run_path,
            "evaluation_metrics.json",
        )
        self._saved_model_path = os.path.join(
            self._run_path,
            "bertopic_model",
        )
        self._topics_path = os.path.join(self._run_path, "topics.npy")
        self._probs_path = os.path.join(self._run_path, "probs.npy")
        self._hierarchical_topics_path = os.path.join(
            self._run_path,
            "hierarchical_topics",
        )

        self._init_folders()

        self.topic_model = None
        self.topics = None
        self.probs = None
        self.embeddings = None
        self.coherence_score = None
        self.evaluation_results = None
        self.hierarchical_topics = dict()

    def _init_folders(self) -> None:
        """Create all run and cache directories required for cache-first flow."""
        os.makedirs(self._base_output_path, exist_ok=True)
        os.makedirs(self._runs_path, exist_ok=True)
        os.makedirs(self._cache_path, exist_ok=True)
        os.makedirs(self._run_path, exist_ok=True)
        os.makedirs(self._visualizations_path, exist_ok=True)
        os.makedirs(self._embeddings_path, exist_ok=True)
        os.makedirs(self._chunks_path, exist_ok=True)
        os.makedirs(self._evaluation_cache_path, exist_ok=True)
        os.makedirs(self._hierarchical_topics_path, exist_ok=True)

    def _log(self, message: str) -> None:
        """Emit progress messages only when verbose mode is enabled."""
        if self._verbose:
            logger.info("[BerTopicModelBuilder] %s", message)

    def _build_model_id(self) -> str:
        """Build a human-readable deterministic id from model configs."""
        payload = {
            "embedding_config": asdict(self._ec),
            "umap_config": asdict(self._umap_config),
            "hdbscan_config": asdict(self._hdbscan_config),
            "outlier_reduction_config": asdict(self._outlier_reduction_config),
        }

        embedding_model_slug = str(self._ec.embedding_model).replace("\\", "/").split("/")[-1]
        embedding_model_slug = (
            embedding_model_slug.replace(" ", "_")
            .replace(".", "_")
            .replace("-", "_")
            .lower()
        )

        representation = self._ec.document_representation.value
        max_words = self._ec.max_words
        umap_neighbors = self._umap_config.n_neighbors
        umap_components = self._umap_config.n_components
        hdbscan_min_cluster = self._hdbscan_config.min_cluster_size

        readable_prefix = (
            f"emb_{embedding_model_slug}"
            f"__rep_{representation}"
            f"__mw_{max_words}"
            f"__umap_n{umap_neighbors}_c{umap_components}"
            f"__hdb_mcs{hdbscan_min_cluster}"
        )
        if self._outlier_reduction_config.enabled:
            strategy_slug = self._outlier_reduction_config.strategy.value.replace("-", "")
            threshold_slug = str(self._outlier_reduction_config.threshold).replace(".", "p")
            readable_prefix += f"__or_{strategy_slug}_t{threshold_slug}"
        # Keep file/folder names manageable while still readable.
        readable_prefix = readable_prefix[:120]
        return readable_prefix

    @staticmethod
    def compute_or_suffix(config: OutlierReductionConfig) -> str:
        """Return the run-directory suffix for an outlier-reduction variant."""
        strategy_slug = config.strategy.value.replace("-", "")
        threshold_slug = str(config.threshold).replace(".", "p")
        return f"__or_{strategy_slug}_t{threshold_slug}"

    def _clone_builder(
        self,
        load_merged: Optional[bool] = None,
        run_suffix: Optional[str] = None,
    ) -> "BerTopicModelBuilder":
        """Create a new builder sharing configs but pointing at another run path."""
        texts_df = pd.DataFrame({self._text_column: self._document_texts})
        return BerTopicModelBuilder(
            texts_df=texts_df,
            text_column=self._text_column,
            folder_name=self._folder_name,
            embedding_config=self._ec,
            umap_config=self._umap_config,
            hdbscan_config=self._hdbscan_config,
            countvectorizer_config=self._countvectorizer_config,
            verbose=self._verbose,
            base_path=self._resolved_base_path,
            compute_config=self._compute_config,
            outlier_reduction_config=self._outlier_reduction_config,
            load_merged=self._load_merged if load_merged is None else load_merged,
            run_suffix=self._run_suffix if run_suffix is None else run_suffix,
        )

    def _ensure_fitted(self) -> None:
        """Raise if the builder has no fitted model and topic assignments."""
        if self.topic_model is None:
            raise RuntimeError("Call fit_transform before this operation.")
        if self.topics is None:
            if os.path.exists(self._topics_path):
                self.topics = np.load(self._topics_path)
            else:
                raise RuntimeError(
                    "Topic assignments are missing. Call fit_transform first."
                )
        if self.probs is None and os.path.exists(self._probs_path):
            self.probs = np.load(self._probs_path, allow_pickle=True)

    def _copy_saved_model_to(self, dest_model_path: str) -> None:
        """Copy the on-disk BERTopic model file to ``dest_model_path``."""
        if not os.path.exists(self._saved_model_path):
            raise RuntimeError(
                f"Saved model not found at '{self._saved_model_path}'. "
                "Call fit_transform first."
            )
        shutil.copy(self._saved_model_path, dest_model_path)


    def _persist_variant(
        self,
        builder: "BerTopicModelBuilder",
        topic_model: BERTopic,
        topics: np.ndarray,
        probs,
    ) -> None:
        """Save model, topics/probs, visualizations, and evaluation for a variant run."""
        try:
            topic_model.save(builder._saved_model_path)
            builder._log(f"Saved BERTopic model: {builder._saved_model_path}")
        except Exception as ex:
            logger.warning(
                "Could not save BERTopic model '%s': %s",
                builder._saved_model_path,
                ex,
            )

        try:
            np.save(builder._topics_path, np.asarray(topics))
            np.save(builder._probs_path, np.asarray(probs, dtype=object))
            builder._log(
                f"Saved topics/probs artifacts: {builder._topics_path}, {builder._probs_path}"
            )
        except Exception as ex:
            logger.warning("Could not save topics/probs artifacts: %s", ex)

        builder._save_visualizations(
            topic_model=topic_model,
            output_dir=builder._visualizations_path,
            force_compute=True,
        )
        results = builder._run_and_save_evaluation(
            topic_model=topic_model,
            topics=topics,
            output_path=builder._evaluation_metrics_path,
            model_id=builder.model_id,
        )
        builder.topic_model = topic_model
        builder.topics = np.asarray(topics)
        builder.probs = probs
        builder.evaluation_results = results
        builder.coherence_score = results.get("coherence_c_v")

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
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return []

        text = str(text).strip()
        if not text:
            return []

        doc = nlp(text)

        chunks = []
        current_chunk = []
        current_len = 0

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sent_len = len(sent_text.split())
            if sent_len == 0:
                continue

            if current_len + sent_len > self._ec.max_words and current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = []
                current_len = 0

            current_chunk.append(sent_text)
            current_len += sent_len

        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def _load_chunks(self) -> None:
        """Load or build chunk texts and per-chunk document ids from disk cache."""
        chunks_full_path = os.path.join(self._chunks_path, self._ec.chunks_file)
        doc_ids_full_path = os.path.join(self._chunks_path, self._ec.doc_ids_file)

        def _sanitize_chunks_and_ids(chunk_texts, doc_ids):
            cleaned_chunks = []
            cleaned_doc_ids = []
            for chunk, doc_id in zip(chunk_texts, doc_ids):
                if chunk is None:
                    continue
                chunk_text = str(chunk).strip()
                if not chunk_text:
                    continue
                cleaned_chunks.append(chunk_text)
                cleaned_doc_ids.append(doc_id)
            return cleaned_chunks, np.asarray(cleaned_doc_ids, dtype=np.int64)

        force_compute = self._compute_config.force_chunks

        if os.path.exists(chunks_full_path) and os.path.exists(doc_ids_full_path) and not force_compute:
            self._log("Loading chunk cache from disk.")
            cached_chunks = pd.read_pickle(chunks_full_path)
            cached_doc_ids = np.load(doc_ids_full_path)
            self._texts, self._doc_ids = _sanitize_chunks_and_ids(cached_chunks, cached_doc_ids)
            self._log(f"Loaded {len(self._texts)} chunks.")
        else:
            if force_compute:
                self._log("force_compute=True, recomputing chunks.")
            else:
                self._log("Chunk cache not found. Building chunks with spaCy.")
            chunk_texts = []
            doc_ids = []

            nlp = spacy.load(self._ec.spacy_model)
            self._log(f"Loaded spaCy model: {self._ec.spacy_model}")

            for doc_id, text in enumerate(self._document_texts):
                chunks = self._sentence_chunking(text, nlp)

                chunk_texts.extend(chunks)
                doc_ids.extend([doc_id] * len(chunks))

            self._texts, self._doc_ids = _sanitize_chunks_and_ids(chunk_texts, doc_ids)
            self._log(f"Created {len(self._texts)} chunks from {len(self._document_texts)} documents.")

            pd.to_pickle(self._texts, chunks_full_path)
            np.save(doc_ids_full_path, self._doc_ids)
            self._log("Saved chunk cache to disk.")

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
        force_compute = self._compute_config.force_embeddings

        if os.path.exists(embeddings_file_path) and not force_compute:
            self._log(f"Loading embeddings cache: {self._ec.embeddings_file}")
            self.embeddings = np.load(embeddings_file_path)
            self._log(f"Embeddings loaded with shape {self.embeddings.shape}.")

            if self._ec.document_representation is DocumentRepresentation.CHUNKS:
                # load chunks as the list of texts
                self._load_chunks()
            else:
                self._texts = list(self._document_texts)
                self._log(f"Using {len(self._texts)} full documents as BERTopic texts.")

            return
        
        if force_compute:
            self._log("force_compute=True, recomputing embeddings.")
        else:
            self._log("Embeddings cache not found. Computing embeddings.")

        if self._ec.document_representation is not DocumentRepresentation.FULL_TEXT:
            self._log(f"Document representation: {self._ec.document_representation.value}")
            # create chunks for sentences
            self._load_chunks()

            # encode the chunks
            self._log(f"Encoding {len(self._texts)} chunks.")
            chunk_embeddings = self.embedding_model.encode(
                self._texts, show_progress_bar=self._verbose
            )
            if self._ec.document_representation is DocumentRepresentation.MEAN_POOLING:
                # pool the embeddings by the mean and set the texts to the original documents
                self.embeddings = self._pool_document_embeddings(chunk_embeddings, "mean")
                self._texts = list(self._document_texts)
                self._log("Applied mean pooling to chunk embeddings.")
            elif self._ec.document_representation is DocumentRepresentation.MAX_POOLING:
                # pool the embeddings by the max and set the texts to the original documents
                self.embeddings = self._pool_document_embeddings(chunk_embeddings, "max")
                self._texts = list(self._document_texts)
                self._log("Applied max pooling to chunk embeddings.")
            else:
                self.embeddings = chunk_embeddings
                self._log("Using chunk-level embeddings without pooling.")
        else:
            self._log("Document representation: full_text")
            self._texts = list(self._document_texts)
            self._log(f"Encoding {len(self._texts)} full documents.")
            self.embeddings = self.embedding_model.encode(
                self._texts, show_progress_bar=self._verbose
            )

        np.save(embeddings_file_path, self.embeddings)
        self._log(f"Saved embeddings cache: {self._ec.embeddings_file}")

    def _save_visualizations(
        self,
        topic_model: Optional[BERTopic] = None,
        output_dir: Optional[str] = None,
        force_compute: bool = False,
    ) -> None:
        """Generate and save any missing BERTopic visualization files.

        HTML outputs are required artifacts for each visualization type. Static
        image snapshots are best-effort and depend on Plotly export backends.
        """
        topic_model = topic_model or self.topic_model
        output_dir = output_dir or self._visualizations_path
        if topic_model is None:
            raise RuntimeError("No topic model available for visualizations.")

        visualizations = [
            (f"topics_pyldavis.html", topic_model.visualize_topics),
            (f"heatmap.html", topic_model.visualize_heatmap),
            (f"hierarchy.html", topic_model.visualize_hierarchy),
            (
                f"barchart_top20.html",
                lambda: topic_model.visualize_barchart(top_n_topics=20),
            ),
        ]
        force_compute = force_compute or self._compute_config.force_visualizations

        for file_name, visualize_fn in visualizations:
            output_path = os.path.join(output_dir, file_name)
            png_path = output_path.replace(".html", ".png")
            jpg_path = output_path.replace(".html", ".jpg")
            has_html = os.path.exists(output_path)
            has_snapshot = os.path.exists(png_path) or os.path.exists(jpg_path)

            if has_html and has_snapshot and not force_compute:
                self._log(f"Skipping existing visualization and snapshot: {output_path}")
                continue

            try:
                fig = visualize_fn()
                if not has_html or force_compute:
                    fig.write_html(output_path)
                    self._log(f"Saved visualization: {output_path}")

                if not has_snapshot or force_compute:
                    try:
                        fig.write_image(png_path)
                        self._log(f"Saved visualization snapshot: {png_path}")
                    except Exception:
                        try:
                            fig.write_image(jpg_path)
                            self._log(f"Saved visualization snapshot: {jpg_path}")
                        except Exception as image_ex:
                            logger.warning(
                                "Could not save image snapshot for '%s'. "
                                "Install/enable Plotly static export dependencies (e.g. kaleido). Error: %s",
                                file_name,
                                image_ex,
                            )
            except Exception as ex:
                logger.warning(
                    "Could not save visualization '%s': %s",
                    file_name,
                    ex,
                )

        missing_html = [
            file_name
            for file_name, _ in visualizations
            if not os.path.exists(os.path.join(output_dir, file_name))
        ]
        if missing_html:
            logger.error(
                "Missing BERTopic visualization artifacts after generation attempt: "
                + ", ".join(missing_html)
            )

    def _create_countvectorizer_model(self) -> CountVectorizer:
        return CountVectorizer(
            strip_accents=self._countvectorizer_config.strip_accents,
            lowercase=self._countvectorizer_config.lowercase,
            stop_words=self._countvectorizer_config.stop_words,
            min_df=self._countvectorizer_config.min_df,
            ngram_range=self._countvectorizer_config.ngram_range
        )

    def _load_model(self, force_compute: bool=False) -> bool:
        """Load a fitted BERTopic model from disk or initialize a new one.

        Returns
        -------
        bool
            ``True`` when a model is loaded from disk, ``False`` when a new
            in-memory model is initialized.
        """

        self._log(f"Loading embedding model: {self._ec.embedding_model}")
        self.embedding_model = SentenceTransformer(self._ec.embedding_model)

        if not force_compute and os.path.exists(self._saved_model_path):
            self.topic_model = BERTopic.load(self._saved_model_path)
            self._log(f"Loaded saved BERTopic model: {self._saved_model_path}")
            return True
        else:
            self._log("Initializing BERTopic model with configured UMAP and HDBSCAN.")

            umap_model = UMAP(
                n_neighbors=self._umap_config.n_neighbors,
                n_components=self._umap_config.n_components,
                min_dist=self._umap_config.min_dist,
                metric=self._umap_config.metric,
                random_state=self._umap_config.random_state,
            )

            hdbscan_model = HDBSCAN(
                min_cluster_size=self._hdbscan_config.min_cluster_size,
                metric=self._hdbscan_config.metric,
                prediction_data=self._hdbscan_config.prediction_data,
            )

            calculate_probabilities = (
                self._outlier_reduction_config.enabled
                and self._outlier_reduction_config.requires_probabilities()
            )

            self.countvectorizer_model = self._create_countvectorizer_model()

            self.topic_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=self.countvectorizer_model,
                calculate_probabilities=calculate_probabilities,
                verbose=self._verbose,
                embedding_model=self.embedding_model,
            )
            return False

    def _save_model(self) -> None:
        """Persist the fitted BERTopic model to the run directory."""
        try:
            self.topic_model.save(self._saved_model_path)
            self._log(f"Saved BERTopic model: {self._saved_model_path}")
        except Exception as ex:
            logger.warning("Could not save BERTopic model '%s': %s", self._saved_model_path, ex)

    def _save_topics_probs(self) -> None:
        """Persist topics and probabilities arrays for this run."""
        try:
            np.save(self._topics_path, np.asarray(self.topics))
            np.save(self._probs_path, np.asarray(self.probs, dtype=object))
            self._log(f"Saved topics/probs artifacts: {self._topics_path}, {self._probs_path}")
        except Exception as ex:
            logger.warning("Could not save topics/probs artifacts: %s", ex)

    def _load_saved_evaluation_results(self) -> bool:
        """Load evaluator results from disk into instance attributes."""
        if self._compute_config.force_evaluation:
            self._log("force_compute=True, ignoring cached evaluation results.")
            return False

        if not os.path.exists(self._evaluation_metrics_path):
            return False

        try:
            with open(self._evaluation_metrics_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            self.evaluation_results = payload
            self.coherence_score = payload.get("coherence_c_v")
            self._log(f"Loaded saved evaluation results: {self._evaluation_metrics_path}")
            return True
        except Exception as ex:
            logger.warning(
                "Could not load evaluation results '%s': %s",
                self._evaluation_metrics_path,
                ex,
            )
            return False

    def _run_and_save_evaluation(
        self,
        topic_model: BERTopic,
        topics: np.ndarray,
        output_path: str,
        model_id: str,
    ) -> dict:
        """Run the evaluator and persist results to ``output_path``."""
        if self.embeddings is None:
            self._load_embeddings()

        evaluator = BerTopicEvaluator(
            topic_model=topic_model,
            texts=self._texts,
            topics=topics,
            embeddings=self.embeddings,
            doc_ids=(
                getattr(self, "_doc_ids", None)
                if self._ec.document_representation is DocumentRepresentation.CHUNKS
                else None
            ),
            cache_dir=self._evaluation_cache_path,
            document_representation=self._ec.document_representation.value,
            model_id=model_id,
        )
        results = evaluator.evaluate()
        evaluator.save(
            results=results,
            output_path=output_path,
            metadata={
                "model_id": model_id,
                "document_representation": self._ec.document_representation.value,
                "embedding_config": asdict(self._ec),
                "umap_config": asdict(self._umap_config),
                "hdbscan_config": asdict(self._hdbscan_config),
                "outlier_reduction_config": asdict(self._outlier_reduction_config),
            },
        )
        self._log(f"Saved evaluation results to {output_path}")
        return results

    def _fit_transform_model(self, force_compute: bool=False) -> bool:
        """Load cached BERTopic outputs, or fit and optionally cache them.

        When outlier reduction is enabled, a fresh fit does **not** persist
        topics/probs/model here; :meth:`fit_transform` saves once after
        reduction. Cached topics/probs are always post-reduction assignments.

        Returns
        -------
        bool
            ``True`` when topics/probabilities were loaded from cache,
            ``False`` when a new fit was executed.
        """
        if not force_compute and os.path.exists(self._topics_path) and os.path.exists(self._probs_path):
            self.topics = np.load(self._topics_path)
            self.probs = np.load(self._probs_path, allow_pickle=True)
            self._log("Loaded cached topics/probs arrays.")
            return True
        else:
            self._log("Fitting BERTopic model.")
            self.topics, self.probs = self.topic_model.fit_transform(self._texts, self.embeddings)
            return False

    def _reduce_outliers(
        self,
        topic_model: Optional[BERTopic] = None,
        topics: Optional[np.ndarray] = None,
        probs=None,
        config: Optional[OutlierReductionConfig] = None,
    ) -> np.ndarray:
        """Reassign topic ``-1`` documents using BERTopic's reduce_outliers.

        Returns updated topic assignments without mutating ``self`` unless
        callers use the default arguments (``self.topic_model``, ``self.topics``).
        """
        topic_model = topic_model or self.topic_model
        config = config or self._outlier_reduction_config
        topics_array = np.asarray(topics if topics is not None else self.topics)
        probs = self.probs if probs is None else probs

        if not config.enabled:
            return topics_array

        outliers_before = int(np.sum(topics_array == -1))
        if outliers_before == 0:
            self._log("Outlier reduction enabled but no outlier assignments found.")
            return topics_array

        kwargs = {
            "documents": self._texts,
            "topics": topics_array.tolist(),
            "strategy": config.strategy.value,
            "threshold": config.threshold,
        }
        if config.strategy is OutlierReductionStrategy.EMBEDDINGS:
            if self.embeddings is None:
                self._load_embeddings()
            kwargs["embeddings"] = self.embeddings
        elif config.strategy is OutlierReductionStrategy.PROBABILITIES:
            if probs is None:
                logger.warning(
                    "Outlier reduction strategy 'probabilities' requires topic "
                    "probabilities; skipping reduction."
                )
                return topics_array
            kwargs["probabilities"] = probs
        elif config.strategy is OutlierReductionStrategy.DISTRIBUTIONS:
            kwargs["distributions_params"] = config.distributions_params

        self._log(
            f"Reducing outliers with strategy={config.strategy.value}, "
            f"threshold={config.threshold}. "
            f"Outlier assignments before reduction: {outliers_before}."
        )
        new_topics = topic_model.reduce_outliers(**kwargs)
        new_topics_array = np.asarray(new_topics)

        outliers_after = int(np.sum(new_topics_array == -1))
        reassigned = outliers_before - outliers_after
        self._log(
            f"Outlier reduction complete. Reassigned {reassigned} assignments; "
            f"outliers remaining: {outliers_after}."
        )

        if reassigned > 0:
            if not hasattr(self, "countvectorizer_model"):
                self.countvectorizer_model = self._create_countvectorizer_model()
            topic_model.update_topics(self._texts, topics=new_topics_array.tolist(), vectorizer_model=self.countvectorizer_model)

        return new_topics_array

    def fit_transform(self):
        """Execute BERTopic pipeline with cache-first semantics.

        Cache lookup order:
        1) embeddings/chunks, 2) fitted BERTopic model, 3) topics/probs,
        4) visualizations, 5) evaluation metrics.
        Missing artifacts are computed and persisted.

        When outlier reduction is enabled, cached topics/probs represent
        post-reduction assignments and reduction is skipped on cache hit.
        A fresh run persists topics/probs/model once after reduction completes.

        After this method returns, the builder guarantees:
        - ``topic_model`` is available and usable
        - ``topics``/``probs`` are in memory
        - visualization files were attempted and missing ones generated
        - evaluation metrics are loaded into ``evaluation_results``

        Returns
        -------
        topics : numpy.ndarray
            Topic index per row in ``self._texts``.
        probs : numpy.ndarray or None
            Per-document (or per-chunk) topic probabilities when available.
        """
        self._log("Starting fit_transform.")
        if self._compute_config.force_compute_all:
            self._log("Global force_compute enabled. Recomputing all pipeline artifacts.")
        
        self._load_embeddings()
        self._log(
            f"Prepared {len(self._texts)} texts and embeddings with shape {self.embeddings.shape}."
        )

        model_loaded = self._load_model(
            force_compute=self._compute_config.force_model
        )

        outputs_loaded = self._fit_transform_model(
            force_compute=self._compute_config.force_topics_probs
        )
        if outputs_loaded and not model_loaded:
            # Topics/probs cache without a fitted BERTopic model is inconsistent.
            # Refit once so the builder always ends with a working topic_model.
            self._log(
                "Found cached topics/probs without cached fitted model. "
                "Refitting BERTopic to rebuild model artifact."
            )
            self._load_model(force_compute=True)
            outputs_loaded = self._fit_transform_model(force_compute=True)
        elif model_loaded and not outputs_loaded:
            self._log("Model cache hit but topics/probs cache missing. Recomputed fit outputs.")
            outputs_loaded = self._fit_transform_model(force_compute=True)

        if not outputs_loaded:
            if self._outlier_reduction_config.enabled:
                self.topics = self._reduce_outliers()
            self._save_topics_probs()
            self._save_model()

        if self.topic_model is None or self.topics is None:
            raise RuntimeError("BERTopic fit did not produce a usable topic model and topic assignments.")

        n_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        outliers = int(np.sum(np.asarray(self.topics) == -1))
        self._log(
            f"BERTopic fit complete. Found {n_topics} topics. Outlier assignments: {outliers}."
        )

        self._log("Ensuring BERTopic visualizations are present.")
        self._save_visualizations()

        if not self._load_saved_evaluation_results():
            self._log("Running BERTopic evaluator.")
            self.evaluation_results = self._run_and_save_evaluation(
                topic_model=self.topic_model,
                topics=self.topics,
                output_path=self._evaluation_metrics_path,
                model_id=self.model_id,
            )

        if self.evaluation_results is None:
            raise RuntimeError("Evaluation metrics are missing after fit_transform execution.")

        self.coherence_score = self.evaluation_results.get("coherence_c_v")
        self._log(
            f"Evaluation complete. coherence_c_v={self.coherence_score}"
        )

    def _hierarchical_topics_cache_file(self, linkage_method: str) -> str:
        """Return the on-disk cache path for a linkage method."""
        safe_method = linkage_method.replace(os.sep, "_")
        return os.path.join(
            self._hierarchical_topics_path,
            f"hierarchical_topics_{safe_method}.pkl",
        )

    def _load_hierarchical_topics_cache(
        self, linkage_method: str
    ) -> pd.DataFrame | None:
        """Load cached hierarchical topics for ``linkage_method``, if present."""
        cache_file = self._hierarchical_topics_cache_file(linkage_method)
        if not os.path.exists(cache_file):
            return None

        try:
            result = pd.read_pickle(cache_file)
            self._log(
                f"Loaded cached hierarchical topics ({linkage_method}): {cache_file}"
            )
            return result
        except Exception as ex:
            logger.warning(
                "Could not load hierarchical topics cache '%s': %s",
                cache_file,
                ex,
            )
            return None

    def _save_hierarchical_topics_cache(
        self, linkage_method: str, result: pd.DataFrame
    ) -> None:
        """Persist hierarchical topics for ``linkage_method`` to disk."""
        cache_file = self._hierarchical_topics_cache_file(linkage_method)
        try:
            pd.to_pickle(result, cache_file)
            self._log(
                f"Saved hierarchical topics cache ({linkage_method}): {cache_file}"
            )
        except Exception as ex:
            logger.warning(
                "Could not save hierarchical topics cache '%s': %s",
                cache_file,
                ex,
            )

    def get_hierarchical_topic(self, linkage_method: str = "ward", force_compute: bool = False):
        """Create or load hierarchical topics for the fitted model.

        Lookup order:
        1) in-memory cache on ``self.hierarchical_topics``
        2) on-disk pickle under ``runs/<model_id>/hierarchical_topics/``
        3) BERTopic ``hierarchical_topics`` computation

        Parameters
        ----------
        linkage_method : str, default="ward"
            SciPy linkage method passed to hierarchical clustering.
        force_compute : bool, default=False
            When True, bypass in-memory and on-disk caches and recompute.
        """
        if self.topic_model is None:
            raise RuntimeError(
                "Call fit_transform before computing hierarchical topics."
            )

        force_compute = force_compute or self._compute_config.force_compute_all

        if linkage_method in self.hierarchical_topics and not force_compute:
            return self.hierarchical_topics[linkage_method]

        if not force_compute:
            cached = self._load_hierarchical_topics_cache(linkage_method)
            if cached is not None:
                self.hierarchical_topics[linkage_method] = cached
                return cached

        if force_compute and linkage_method in self.hierarchical_topics:
            self._log(
                f"force_compute=True, recomputing hierarchical topics ({linkage_method})."
            )
        else:
            self._log(f"Computing hierarchical topics ({linkage_method}).")

        if linkage_method == "ward":
            # ward is the default method, we don't need a lambda function
            result = self.topic_model.hierarchical_topics(self._texts)
        else:
            # use provided linkage function
            linkage_function = lambda x: sch.linkage(
                x, linkage_method, optimal_ordering=True
            )
            result = self.topic_model.hierarchical_topics(
                self._texts,
                linkage_function=linkage_function,
            )

        # save in dict
        self.hierarchical_topics[linkage_method] = result
        self._save_hierarchical_topics_cache(linkage_method, result)

        return result

    def merge_topics(self, merge_topic_list: list[list[int]]) -> "BerTopicModelBuilder":
        """Copy the current model, merge topics, and persist to a ``__merged`` run.

        The original builder is not modified. Returns a new builder with
        ``load_merged=True`` and the merged model pre-loaded in memory.

        Parameters
        ----------
        merge_topic_list : list of list of int
            Groups of topic ids to merge (BERTopic ``merge_topics`` format).

        Returns
        -------
        BerTopicModelBuilder
            Builder pointing at ``runs/<base_id>__merged/`` (plus any existing
            ``run_suffix`` on this instance).
        """
        self._ensure_fitted()
        if self.embeddings is None:
            self._load_embeddings()

        merged_builder = self._clone_builder(load_merged=True)
        self._copy_saved_model_to(merged_builder._saved_model_path)

        # merge topics
        merged_model = BERTopic.load(merged_builder._saved_model_path)
        merged_model.merge_topics(self._texts, merge_topic_list)
        merged_topics = np.asarray(merged_model.topics_)
        probs = self.probs

        # update topic representation
        if not hasattr(self, "countvectorizer_model"):
            self.countvectorizer_model = self._create_countvectorizer_model()
        merged_model.update_topics(self._texts, vectorizer_model=self.countvectorizer_model)

        self._persist_variant(
            builder=merged_builder,
            topic_model=merged_model,
            topics=merged_topics,
            probs=probs,
        )
        merged_builder.embeddings = self.embeddings
        if hasattr(self, "_texts"):
            merged_builder._texts = self._texts
        if hasattr(self, "_doc_ids"):
            merged_builder._doc_ids = self._doc_ids
        self._log(f"Merged model persisted under: {merged_builder._run_path}")
        return merged_builder

    def reduce_outliers(
        self, config: OutlierReductionConfig
    ) -> "BerTopicModelBuilder":
        """Copy the current model, apply outlier reduction, and persist to a new run.

        The original builder is not modified. Returns a new builder whose
        ``model_id`` includes ``compute_or_suffix(config)`` appended after any
        existing ``run_suffix`` (and after ``__merged`` when applicable).

        Parameters
        ----------
        config : OutlierReductionConfig
            Outlier reduction settings; ``enabled`` must be True.

        Returns
        -------
        BerTopicModelBuilder
            Builder pointing at the new outlier-reduction variant directory.
        """
        if not config.enabled:
            raise ValueError("Outlier reduction config must have enabled=True.")

        self._ensure_fitted()
        if self.embeddings is None:
            self._load_embeddings()

        or_suffix = self.compute_or_suffix(config)
        or_builder = self._clone_builder(
            load_merged=self._load_merged,
            run_suffix=self._run_suffix + or_suffix,
        )
        self._copy_saved_model_to(or_builder._saved_model_path)

        or_model = BERTopic.load(or_builder._saved_model_path)
        reduced_topics = self._reduce_outliers(
            topic_model=or_model,
            topics=self.topics,
            probs=self.probs,
            config=config,
        )

        self._persist_variant(
            builder=or_builder,
            topic_model=or_model,
            topics=reduced_topics,
            probs=self.probs,
        )
        or_builder.embeddings = self.embeddings
        if hasattr(self, "_texts"):
            or_builder._texts = self._texts
        if hasattr(self, "_doc_ids"):
            or_builder._doc_ids = self._doc_ids
        self._log(
            f"Outlier-reduced model persisted under: {or_builder._run_path}"
        )
        return or_builder