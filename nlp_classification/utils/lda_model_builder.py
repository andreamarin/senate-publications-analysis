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
        base_path: str = None,
        model_passes: int = 300
    ):
        self._texts_df = texts_df

        self._topics_range = range(
            num_topics_config.get("min_topics", 3),
            num_topics_config.get("max_topics", 15) + 1,
            num_topics_config.get("step_size", 3),
        )

        current_path = pathlib.Path(__file__).parent.resolve()
        base_path = current_path.parent.resolve() if base_path is None else base_path

        self._images_path = f"{base_path}/{folder_name}/coherence_scores"
        self._models_path = f"{base_path}/{folder_name}/models"

        self._model_passes = model_passes

        self.dictionary = None
        self.corpus = None
        self.models = dict()

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
                    corpus, num_topics=num_topics, id2word=dictionary, passes=self._model_passes
                )

                # save model to dictionary
                if model_id not in self.models:
                    self.models[model_id] = dict()

                self.models[model_id][num_topics] = model

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
            self._coherence_scores.append(coherence_lda)

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

    def _load_dictionary(self, model_id: str):
        """
        Load the dictionary and corpus for the given model

        Parameters
        ----------
        model_dir : str
            path where the dictionary and corpus are saved
        """
        # dictionary of words
        dictionary_path = f"{self._models_path}/{model_id}/dictionary.pkl"

        if not os.path.exists(dictionary_path):
            raise ValueError(f"Dictionary for model doesn't exist at: {dictionary_path}")

        self.dictionary = pickle.load(open(dictionary_path, "rb"))

        # corpus with bag of words
        corpus_path = f"{self._models_path}/{model_id}/corpus.pkl"

        if not os.path.exists(corpus_path):
            raise ValueError(f"Corpus for model doesn't exist at: {corpus_path}")
        
        self.corpus = pickle.load(open(corpus_path, "rb"))
    
    def _load_model(self, model_id: str, num_topics: int):
        # load model
        model_path = f"{self._models_path}/{model_id}/{num_topics}_topics/lda"

        if self.models.get(model_id, dict()).get(num_topics, None):
            # model already loaded
            return

        if not os.path.exists(model_path):
            raise ValueError(f"Model doesn't exist at: {model_path}")
        
        if model_id not in self.models:
            self.models[model_id] = dict()

        self.models[model_id][num_topics] = LdaModel.load(model_path)

    def generate_model_vis(self, text_type: str, num_topics: int, filter: str = "all"):

        model_id = f"{text_type}_{filter}"

        # load the needed data
        self._load_model(model_id, num_topics)
        self._load_dictionary(model_id)

        model = self.models[model_id][num_topics]

        # create df with topics
        topics_df = pd.DataFrame()
        for num_topic, topic_words in model.print_topics(
            num_topics=num_topics, num_words=10
        ):
            topics_df[f"topic_{num_topic}"] = topic_words.split(" + ")

        # generalte topic visualization
        vis = pyLDAvis.gensim.prepare(model, self.corpus, self.dictionary)

        return topics_df, vis
    
    def set_model(self, text_type: str, num_topics: int, filter: str = "all"):
        model_id = f"{text_type}_{filter}"
        self._load_model(model_id, num_topics)

        self.model = self.models.get(model_id, dict()).get(num_topics)

        if self.model is None:
            raise ValueError(f"Model not found for specs: {model_id}, {num_topics}")

    def get_publication_topics(self, text):
        words = text.split()
        text_corpus = self.dictionary.doc2bow(words)

        publication_topics = []
        for topic, score in self.model.get_document_topics(text_corpus):
            publication_topics.append({"topic": topic, "score": float(score)})

        return publication_topics

    def print_topics(self, text_type: str, filter: str = "all", num_topics_list: list = []):
        """
        Print the topics for the provided model specs 
        """

        topics_list = [num_topics_list] if num_topics_list else self._topics_range
        model_id = f"{text_type}_{filter}"

        for num_topics in topics_list:
            print("====="*10)
            print(f"Topics for: {model_id}, {num_topics}")

            # load model output
            self._load_model(model_id, num_topics)

            # get the topics
            for topic in self.models[model_id][num_topics].print_topics(num_topics=num_topics, num_words=10):
                print(topic)