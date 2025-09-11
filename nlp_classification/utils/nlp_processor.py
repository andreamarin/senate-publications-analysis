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
        processed_text = text

        # execute functions
        for function in self._extra_processing_steps:
            processed_text = function(processed_text)

        # remove specific words (before lemmatizing)
        for word in self._remove_words:
            processed_text = processed_text.replace(word, "")

        # remove multiple spaces
        processed_text = re.sub(r" +", " ", processed_text)
        processed_text = processed_text.strip()

        # remove punctuation marks
        translator = str.maketrans('', '', string.punctuation)
        processed_text = processed_text.translate(translator)

        # lemmatize text and turn to lower case
        nlp_text = self._nlp(processed_text)
        lemmas = [word.lemma_.lower() for word in nlp_text]

        if self._stop_words:
            # remove stopwords
            filtered_tokens = [
                token for token in lemmas if token not in self._stop_words
            ]

            processed_text = " ".join(filtered_tokens)
        else:
            # join all the tokens
            processed_text = " ".join(lemmas)

        return processed_text.strip()

    def _filter_by_pos(self, text: str, pos_tags: list):
        """
        keep words that match the given part of speech tags
        """
        nlp_text = self._nlp(text)

        # get the words for the given part of speech
        filtered_tokens = [str(word) for word in nlp_text if word.pos_ in pos_tags]

        return " ".join(filtered_tokens)
