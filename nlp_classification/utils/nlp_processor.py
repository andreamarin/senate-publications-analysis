import re
import os
import spacy
import pickle
import pathlib
import numpy as np
import pandas as pd
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import gensim.corpora as corpora
from functools import partial
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel


class NlpProcessor:

    def __init__(
        self,
        texts_df: pd.DataFrame,
        spacy_model_name: str,
        process_text_config: dict,
        extra_processing = None
    ):
        self._nlp = spacy.load(spacy_model_name)
        self._texts_df = texts_df

        self._extra_processing_function = extra_processing

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
        if self._extra_processing_function is not None:
            processed_text = self._extra_processing_function(processed_text)

        # remove specific words (before lemmatizing)
        for word in self._remove_words:
            text = text.replace(word, "")

        # remove multiple spaces
        text = re.sub(r" +", " ", text)
        text = text.strip()

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

    def _generate_models(
        self, model_id: str, model_data: pd.Series, force_replacement: bool = False
    ):
        words = list(model_data.apply(str.split))

        base_dir = f"{self._models_path}/{model_id}"

        # dictionary of words
        dictionary_path = f"{base_dir}/dictionary.pkl"
        if not force_replacement and os.path.exists(dictionary_path):
            dictionary = pickle.load(open(dictionary_path, "rb"))
        else:
            dictionary = corpora.Dictionary(words)

            # save dictionary
            pickle.dump(dictionary, open(dictionary_path, "wb"))

        # corpus with bag of words
        corpus_path = f"{base_dir}/corpus.pkl"
        if not force_replacement and os.path.exists(corpus_path):
            corpus = pickle.load(open(corpus_path, "rb"))
        else:
            corpus = [dictionary.doc2bow(record) for record in words]

            # save corpus
            pickle.dump(corpus, open(corpus_path, "wb"))

        self._model_topics = {}
        self._coherence_scores = []

        for num_topics in self._topics_range:
            print(num_topics, end=", ")

            # create folder where the model will be saved
            model_dir = f"{base_dir}/{num_topics}_topics"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = f"{model_dir}/lda"
            if os.path.exists(model_path):
                # load existing lda model
                model = LdaModel.load(model_path)
            else:
                # create lda model
                model = LdaModel(
                    corpus, num_topics=num_topics, id2word=dictionary, passes=300
                )

                # save model to drive
                model.save(model_path)

            # save topics
            self._model_topics[num_topics] = model.print_topics(
                num_topics=num_topics, num_words=10
            )

        # calculate coherence score
        cm = CoherenceModel(
            model=model, texts=words, dictionary=dictionary, coherence="c_v"
        )
        coherence_lda = cm.get_coherence()
        self.coherence_scores.append(coherence_lda)

    def _create_coherence_plot(self, model_id: str):
        plt.figure(figsize=(10, 6))
        plt.plot(self._topics_range, self._coherence_scores, marker="o")
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.title("Number of Topics vs Coherence Score")
        plt.xticks(self._topics_range)
        plt.grid(True)
        plt.savefig(f"{self._images_path}/{model_id}.png", format="png", dpi=300)
        plt.show()

    def build_lda_models(self, text_type: str, filter_type: str = "all"):
        model_id = f"{text_type}_{filter_type}"

        if filter_type != "all":
            model_data = self._texts_df.loc[self._texts_df.type == filter_type]
        else:
            model_data = self._texts_df

        # build models for each num topic
        self._generate_models(model_id, model_data[text_type])

        # get the num of topics with the max coherence score
        optimal_num_topics = self._topics_range[np.argmax(self._coherence_scores)]
        print("\nOptimal number of topics:", optimal_num_topics)
        print(
            "Coherence Score for optimal number of topics:", max(self._coherence_scores)
        )

        # print best topics
        print("")
        for topic in self._model_topics[optimal_num_topics]:
            print(topic)
        print("")

        self._create_coherence_plot(model_id)

    def plot_coherence_scores_comparison(
        self, title: str, coherence_scores_dict: dict, img_name: str
    ):
        plt.figure(figsize=(10, 6))

        for text_type, coherence_scores in coherence_scores_dict.items():
            plt.plot(self._topics_range, coherence_scores, marker="o", label=text_type)

        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.title(title)
        plt.xticks(self._topics_range)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self._images_path}/{img_name}.png", format="png", dpi=300)
        plt.show()

    def load_model_results(self, text_type, num_topics, filter: str = "all"):
        model_id = f"{text_type}_{filter}"
        base_dir = f"{self._models_path}/{model_id}"

        # dictionary of words
        dictionary_path = f"{base_dir}/dictionary.pkl"
        self.dictionary = pickle.load(open(dictionary_path, "rb"))

        # corpus with bag of words
        corpus_path = f"{base_dir}/corpus.pkl"
        self.corpus = pickle.load(open(corpus_path, "rb"))

        # load model
        model_path = f"{base_dir}/{num_topics}_topics/lda"
        self.model = LdaModel.load(model_path)

    def generate_model_data(self, num_topics):
        # create df with topics
        topics_df = pd.DataFrame()
        for num_topic, topic_words in self.model.print_topics(
            num_topics=num_topics, num_words=10
        ):
            topics_df[f"topic_{num_topic}"] = topic_words.split(" + ")

        # generalte topic visualization
        vis = pyLDAvis.gensim.prepare(self.model, self.corpus, self.dictionary)

        return topics_df, vis

    def get_publication_topics(self, text):
        words = text.split()
        text_corpus = self.dictionary.doc2bow(words)

        publication_topics = []
        for topic, score in self.model.get_document_topics(text_corpus):
            publication_topics.append({"topic": topic, "score": float(score)})

        return publication_topics
