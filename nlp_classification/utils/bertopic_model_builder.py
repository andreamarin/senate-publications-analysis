import os
import json
import hashlib
import logging
import spacy
import pathlib
import numpy as np
import pandas as pd
from umap import UMAP
from dataclasses import asdict
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from .bertopic_config import EmbeddingConfig, UMAPConfig, HDBSCANConfig, DocumentRepresentation
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
        verbose: bool = False,
        base_path: str = None
    ):
        self._document_texts = texts_df[text_column].tolist()
        self._verbose = verbose
        if self._verbose:
            setup_builder_logging()

        # Save config objects and build model_id.
        self._ec = embedding_config
        self._umap_config = umap_config
        self._hdbscan_config = hdbscan_config
        self.model_id = self._build_model_id()

        current_path = pathlib.Path(__file__).parent.resolve()
        base_path = current_path.parent.resolve() if base_path is None else base_path

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

        self._init_folders()

        self.topic_model = None
        self.topics = None
        self.probs = None
        self.embeddings = None
        self.coherence_score = None
        self.evaluation_results = None

    def _init_folders(self) -> None:
        """Create all run and cache directories required for cache-first flow."""
        os.makedirs(self._base_output_path, exist_ok=True)
        os.makedirs(self._runs_path, exist_ok=True)
        os.makedirs(self._cache_path, exist_ok=True)
        os.makedirs(self._run_path, exist_ok=True)
        os.makedirs(self._saved_model_path, exist_ok=True)
        os.makedirs(self._visualizations_path, exist_ok=True)
        os.makedirs(self._embeddings_path, exist_ok=True)
        os.makedirs(self._chunks_path, exist_ok=True)
        os.makedirs(self._evaluation_cache_path, exist_ok=True)

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
        }
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        hash_suffix = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:8]

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
        # Keep file/folder names manageable while still readable.
        readable_prefix = readable_prefix[:120]
        return f"{readable_prefix}__{hash_suffix}"
        
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
            self._log("Loading chunk cache from disk.")
            self._texts = pd.read_pickle(chunks_full_path)
            self._doc_ids = np.load(doc_ids_full_path)
            self._log(f"Loaded {len(self._texts)} chunks.")
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

            self._texts = chunk_texts
            self._doc_ids = np.asarray(doc_ids, dtype=np.int64)
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

        if os.path.exists(embeddings_file_path) and not self._ec.force_compute:
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
        
        if self._ec.force_compute:
            self._log("force_compute=True, recomputing embeddings.")
        else:
            self._log("Embeddings cache not found. Computing embeddings.")

        self._log(f"Loading embedding model: {self._ec.embedding_model}")
        embedding_model = SentenceTransformer(self._ec.embedding_model)

        if self._ec.document_representation is not DocumentRepresentation.FULL_TEXT:
            self._log(f"Document representation: {self._ec.document_representation.value}")
            # create chunks for sentences
            self._load_chunks()

            # encode the chunks
            self._log(f"Encoding {len(self._texts)} chunks.")
            chunk_embeddings = embedding_model.encode(
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
            self.embeddings = embedding_model.encode(
                self._texts, show_progress_bar=self._verbose
            )

        np.save(embeddings_file_path, self.embeddings)
        self._log(f"Saved embeddings cache: {self._ec.embeddings_file}")

    def _save_visualizations(self) -> None:
        """Generate and save any missing BERTopic visualization files.

        HTML outputs are required artifacts for each visualization type. Static
        image snapshots are best-effort and depend on Plotly export backends.
        """
        visualizations = [
            (f"{self.model_id}_topics_pyldavis.html", self.topic_model.visualize_topics),
            (f"{self.model_id}_heatmap.html", self.topic_model.visualize_heatmap),
            (f"{self.model_id}_hierarchy.html", self.topic_model.visualize_hierarchy),
            (
                f"{self.model_id}_barchart_top20.html",
                lambda: self.topic_model.visualize_barchart(top_n_topics=20),
            ),
        ]

        for file_name, visualize_fn in visualizations:
            output_path = os.path.join(self._visualizations_path, file_name)
            png_path = output_path.replace(".html", ".png")
            jpg_path = output_path.replace(".html", ".jpg")
            has_html = os.path.exists(output_path)
            has_snapshot = os.path.exists(png_path) or os.path.exists(jpg_path)

            if has_html and has_snapshot:
                self._log(f"Skipping existing visualization and snapshot: {output_path}")
                continue

            try:
                fig = visualize_fn()
                if not has_html:
                    fig.write_html(output_path)
                    self._log(f"Saved visualization: {output_path}")

                if not has_snapshot:
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
            if not os.path.exists(os.path.join(self._visualizations_path, file_name))
        ]
        if missing_html:
            raise RuntimeError(
                "Missing BERTopic visualization artifacts after generation attempt: "
                + ", ".join(missing_html)
            )

    def _load_model(self, force_compute: bool=False) -> bool:
        """Load a fitted BERTopic model from disk or initialize a new one.

        Returns
        -------
        bool
            ``True`` when a model is loaded from disk, ``False`` when a new
            in-memory model is initialized.
        """

        if not force_compute and os.path.isdir(self._saved_model_path) and len(os.listdir(self._saved_model_path)) > 0:
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
                prediction_data=False,
            )

            self.topic_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                verbose=self._verbose,
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

    def _fit_transform_model(self, force_compute: bool=False) -> bool:
        """Load cached BERTopic outputs, or fit and cache them.

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
            self._save_topics_probs()
            self._save_model()
            return False

    def fit_transform(self):
        """Execute BERTopic pipeline with cache-first semantics.

        Cache lookup order:
        1) embeddings/chunks, 2) fitted BERTopic model, 3) topics/probs,
        4) visualizations, 5) evaluation metrics.
        Missing artifacts are computed and persisted.

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
        
        self._load_embeddings()
        self._log(
            f"Prepared {len(self._texts)} texts and embeddings with shape {self.embeddings.shape}."
        )

        model_loaded = self._load_model()

        outputs_loaded = self._fit_transform_model()
        if outputs_loaded and not model_loaded:
            # Topics/probs cache without a fitted BERTopic model is inconsistent.
            # Refit once so the builder always ends with a working topic_model.
            self._log(
                "Found cached topics/probs without cached fitted model. "
                "Refitting BERTopic to rebuild model artifact."
            )
            self._load_model(force_compute=True)
            self._fit_transform_model(force_compute=True)
        elif model_loaded and not outputs_loaded:
            self._log("Model cache hit but topics/probs cache missing. Recomputed fit outputs.")
            self._fit_transform_model(force_compute=True)


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
            evaluator = BerTopicEvaluator(
                topic_model=self.topic_model,
                texts=self._texts,
                topics=self.topics,
                embeddings=self.embeddings,
                cache_dir=self._evaluation_cache_path,
                document_representation=self._ec.document_representation.value,
                model_id=self.model_id,
            )
            self.evaluation_results = evaluator.evaluate()

            evaluator.save(
                results=self.evaluation_results,
                output_path=self._evaluation_metrics_path,
                metadata={
                    "model_id": self.model_id,
                    "document_representation": self._ec.document_representation.value,
                    "embedding_config": asdict(self._ec),
                    "umap_config": asdict(self._umap_config),
                    "hdbscan_config": asdict(self._hdbscan_config),
                },
            )
            self._log(f"Saved evaluation results to {self._evaluation_metrics_path}")

        if self.evaluation_results is None:
            raise RuntimeError("Evaluation metrics are missing after fit_transform execution.")

        self.coherence_score = self.evaluation_results.get("coherence_c_v")
        self._log(
            f"Evaluation complete. coherence_c_v={self.coherence_score}"
        )
