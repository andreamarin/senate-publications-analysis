import os
import pickle
import pathlib
import numpy as np
import pandas as pd
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel


class LDAModelBuilder:

    def __init__(
        self,
        texts_df: pd.DataFrame,
        folder_name: str,
        num_topics_config: dict,
        base_path: str = None
    ):
        self._texts_df = texts_df

        self._topics_range = range(
            num_topics_config.get("min_topics", 3),
            num_topics_config.get("max_topics", 15),
            num_topics_config.get("step_size", 3),
        )

        current_path = pathlib.Path(__file__).parent.resolve()
        base_path = current_path.parent.resolve() if base_path is None else base_path

        self._images_path = f"{base_path}/{folder_name}/coherence_scores"
        self._models_path = f"{base_path}/{folder_name}/models"

    def _generate_models(
        self, model_id: str, model_data: pd.Series, force_replacement: bool = False
    ):
        words = list(model_data.apply(str.split))

        base_dir = f"{self._models_path}/{model_id}"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

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
