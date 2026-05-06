from dataclasses import dataclass
from enum import Enum
from typing import Union


class DocumentRepresentation(str, Enum):
    """Strategy for turning each document into one or more embedding rows.

    Chunking modes split text with spaCy, embed segments, then either aggregate
    back to one vector per source document (pooling) or keep one vector per
    chunk. ``FULL_TEXT`` embeds each document as a single string.
    """

    MEAN_POOLING = "mean_pooling"
    MAX_POOLING = "max_pooling"
    CHUNKS = "chunks"
    FULL_TEXT = "full_text"


@dataclass
class UMAPConfig:
    """Parameters for ``umap.UMAP`` used to reduce embedding dimensionality."""

    n_neighbors: int
    metric: str = "cosine"
    min_dist: float = 0.1
    random_state: int = 20260415
    n_components: int = 10


@dataclass
class HDBSCANConfig:
    """Parameters for ``hdbscan.HDBSCAN`` used to cluster embeddings."""

    min_cluster_size: int
    metric: str = "euclidean"


@dataclass
class EmbeddingConfig:
    """Configuration for sentence embeddings, chunking, and on-disk caches."""

    embedding_model: str
    max_words: int
    spacy_model: str
    document_representation: Union[DocumentRepresentation, str]

    def __post_init__(self) -> None:
        """Normalize ``document_representation`` and set cache filenames."""
        dr = self.document_representation

        if isinstance(dr, str):
            try:
                self.document_representation = DocumentRepresentation(dr)
            except ValueError:
                self.document_representation = DocumentRepresentation.FULL_TEXT

        if self.document_representation is not DocumentRepresentation.FULL_TEXT:
            self.embeddings_file = (
                f"embeddings_{self.document_representation.value}_{self.max_words}.npy"
            )
            self.chunks_file = f"chunks_{self.max_words}.pkl"
            self.doc_ids_file = f"doc_ids_{self.max_words}.npy"
        else:
            self.embeddings_file = f"embeddings_{self.document_representation.value}.npy"


@dataclass
class ComputeConfig:
    """Per-step cache bypass flags for BERTopic pipeline execution."""

    force_compute_all: bool = False
    force_chunks: bool = False
    force_embeddings: bool = False
    force_model: bool = False
    force_topics_probs: bool = False
    force_visualizations: bool = False
    force_evaluation: bool = False

    def __post_init__(self) -> None:
        """Apply hierarchical force-compute semantics across pipeline stages."""
        if self.force_compute_all or self.force_chunks:
            self.force_chunks = True
            self.force_embeddings = True
            self.force_model = True
            self.force_topics_probs = True
            self.force_visualizations = True
            self.force_evaluation = True

        elif self.force_embeddings:
            self.force_model = True
            self.force_topics_probs = True
            self.force_visualizations = True
            self.force_evaluation = True

        elif self.force_model:
            self.force_topics_probs = True
            self.force_visualizations = True
            self.force_evaluation = True
