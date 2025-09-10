import re
import spacy
import string
import numpy as np
import pandas as pd
from functools import partial


class NlpProcessor:

    def __init__(
        self,
        texts_df: pd.DataFrame,
        spacy_model_name: str,
        process_text_config: dict,
        extra_processing_steps: list = []
    ):
        self._nlp = spacy.load(spacy_model_name)
        self._texts_df = texts_df

        self._extra_processing_steps = extra_processing_steps

        self._remove_words = process_text_config.get("remove_words", [])
        self._stop_words = process_text_config.get("stop_words", [])

        # create vectorized version of function
        self.process_corpus = np.vectorize(self._process_doc)
        self.extract_adj_only = np.vectorize(
            partial(self._filter_by_pos, pos_tags=["ADJ"])
        )
        self.extract_noun_only = np.vectorize(
            partial(self._filter_by_pos, pos_tags=["NOUN"])
        )
        self.extract_adj_noun = np.vectorize(
            partial(self._filter_by_pos, pos_tags=["ADJ", "NOUN"])
        )

    def _process_doc(self, text: str):
        """
        Clean text (create lemmas, remove special characters, remove stop words)
        """
        for function in self._extra_processing_steps:
            processed_text = function(text)

        # remove specific words (before lemmatizing)
        for word in self._remove_words:
            text = text.replace(word, "")

        # remove multiple spaces
        text = re.sub(r" +", " ", text)
        text = text.strip()

        # remove punctuation marks
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        # lemmatize text
        nlp_text = self._nlp(text)
        lemmas = [word.lemma_ for word in nlp_text]

        normalized_text = " ".join(lemmas)

        # remove special characters
        normalized_text = normalized_text.lower()
        normalized_text = re.sub(r"[^a-záéíóúñ\d\s_]", "", normalized_text, re.A)

        if self._stop_words:
            # remove stopwords
            tokens = normalized_text.split(" ")
            filtered_tokens = [
                token for token in tokens if token not in self._stop_words
            ]

            processed_text = " ".join(filtered_tokens)
        else:
            processed_text = normalized_text

        return processed_text

    def _filter_by_pos(self, text: str, pos_tags: list):
        """
        keep words that match the given part of speech tags
        """
        nlp_text = self._nlp(text)

        # get the words for the given part of speech
        filtered_tokens = [str(word) for word in nlp_text if word.pos_ in pos_tags]

        return " ".join(filtered_tokens)
