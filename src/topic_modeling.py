

import json
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import ast
import matplotlib.pyplot as plt
from pprint import pprint
import pickle
import random

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from stance_detection import *


class TopicModeling():
    def __init__(self,data_source = "covidnews", start=None,end = None, use_embedding=False,method="bertopic"):
        self.root_path = "../data/"
        self.post_process_path = self.root_path+"post_processing/"
        self.data_source = data_source
        self.start = start
        self.end = end
        if self.start == None or self.end == None:
            print("Please input the time window of news articles.")

        if use_embedding:
            self.embedding_file_name = "{data_source}_{start}_{end}_embedding.json".format(data_source=self.data_source,start =self.start.strftime('%Y-%m-%d'),end = self.end.strftime('%Y-%m-%d'))
            self.embedding_data_path = self.post_process_path+"embedding/" + self.embedding_file_name
            self.article_df_embedding = pd.read_json(self.embedding_data_path).reset_index(drop=True)
        
        
        self.tokenized_file_name = "{data_source}_{start}_{end}_tokenized.json".format(data_source=self.data_source,start =self.start.strftime('%Y-%m-%d'),end = self.end.strftime('%Y-%m-%d'))
        self.tokenized_data_path = self.post_process_path+"tokenized/" + self.tokenized_file_name

        # load story
        self.stance_target_path = self.root_path+"/stance_target/"
        self.stance_target_withstance_file = "{data_source}_{start}_{end}_stance_targets_withstance.json".format(data_source=self.data_source,start =self.start.strftime('%Y-%m-%d'),end = self.end.strftime('%Y-%m-%d'))
        f = open(self.stance_target_path+self.stance_target_withstance_file)
        self.selected_story = json.load(f)
        self.ge_article_list()

        entity_types = ["Politician","OfficeHolder","PoliticalParty","Party"]
        self.sd= StanceDetection(self.data_source,start,end)
        self.article_df = self.sd.article_df

        df = self.article_df[self.article_df.id.isin(self.articles_id_list)]

        self.text = [" ".join(a) for a in df.sentences]
        df_data = pd.DataFrame()
        df_data["id"]=df["id"]
        df_data["text"] = self.text
        self.df_data = df_data.drop_duplicates()
    
    def ge_article_list(self):
        articles_id_list = []
        for story_id,story in self.selected_story.items():
            artilce_id = list(story["story_entities"].keys())
            articles_id_list+= artilce_id
        self.articles_id_list = articles_id_list
    
    def generate_random_target_distribution(self,n=10, verbose=True):
        keywords_set = set()
        for words in self.topic_model.get_document_info(self.text).Top_n_words.to_list():
            keywords_set = keywords_set.union(set(words.split(" - ")))

        target_distributions = dict()
        n=10
        random.seed(10)
        for i in range(n):
            target_distribution= dict()

            selected_words = random.sample(keywords_set, n)
            print(selected_words)
            target_distribution["keywords"] = selected_words
            p_distr = self.generate_target_distirbution(selected_words,top_n = 10,verbose=verbose)
            print(sum(p_distr))
            target_distribution["target_distribution"] = p_distr
            target_distributions[i] = target_distribution
        self.target_distributions = target_distributions

    def generate_topic_based_target_distribution(self,n=10,verbose=True):
        # randomly sample seed topics, use topic m keywords to search for the similar top k topics.
        target_distributions = dict()
        n=10 # how many sythetic users to generate
        topic_keywords = self.topic_model.get_topic_info().Representation[1:].to_list()
        topic_ids = [i for i in range(len(topic_keywords))]
        random.seed(10)
        for i in range(n):
            target_distribution= dict()
            selected_topics = random.sample(topic_ids, 5)
            selected_words = set()
            target_distribution["seed_topic"] = selected_topics
            for t_id in selected_topics:
                selected_words = selected_words.union(topic_keywords[t_id][:3])

            target_distribution["keywords"] = list(selected_words)
                
            p_distr = self.generate_target_distirbution(selected_words,top_n = 10,verbose=verbose)
            print(sum(p_distr))
            target_distribution["target_distribution"] = p_distr
            target_distributions[i] = target_distribution
        self.target_distributions = target_distributions


    def save_topic_distribution(self):
        self.topic_distribution_path = self.root_path+ "topic_model/topic_distribution/"
        self.target_distribution_path = self.root_path+ "topic_model/target_distribution/"
        self.topic_distribution_file =  "{data_source}_{start}_{end}_topic_distribution.pkl".format(data_source=self.data_source,start =self.start.strftime('%Y-%m-%d'),end =self.end.strftime('%Y-%m-%d'))
        self.target_distribution_file =  "{data_source}_{start}_{end}_target_distribution.pkl".format(data_source=self.data_source,start =self.start.strftime('%Y-%m-%d'),end =self.end.strftime('%Y-%m-%d'))
        with open(self.topic_distribution_path+self.topic_distribution_file, 'wb') as f:
            pickle.dump(self.distr, f)
        with open(self.target_distribution_path+self.target_distribution_file, 'wb') as f:
            pickle.dump(self.target_distributions, f)

                #save toic results
        df = self.topic_model.get_topic_info()
        self.topic_info_file =  "{data_source}_{start}_{end}_topic_info.csv".format(data_source=self.data_source,start =self.start.strftime('%Y-%m-%d'),end =self.end.strftime('%Y-%m-%d'))
        df.to_csv(self.topic_distribution_path+self.topic_info_file,index=False)

    def bertTopic(self):
        vectorizer_model = CountVectorizer(stop_words="english")
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        representation_model = MaximalMarginalRelevance(diversity=0.2)
        self.topic_model = BERTopic(vectorizer_model=vectorizer_model,ctfidf_model=ctfidf_model,representation_model=representation_model)
        topics, probs = self.topic_model.fit_transform(self.text)
        topic_distr, _ = self.topic_model.approximate_distribution(self.text)
        self.topic_length = len(topic_distr[1])
        self.topic_model.visualize_heatmap()
        self.topic_model.visualize_topics()

        distr = {}
        self.topic_distr = topic_distr
        a_ids = self.df_data.id.to_list()
        for i in range(len(a_ids)):
            distr[a_ids[i]] = topic_distr[i]
        self.distr = distr

        self.keywords_set = set()
        for words in self.topic_model.get_document_info(self.text).Top_n_words.to_list():
            self.keywords_set = self.keywords_set.union(set(words.split(" - ")))
        


    def generate_target_distirbution(self,key_words_set,top_n = 10, decay_similarity=True,target_w=0.7,verbose=False ):
        # for given key words sets, find most related topics and assign values
        p_distr = np.zeros(self.topic_length)
        for key_words in key_words_set:
            similar_topics, similarity = self.topic_model.find_topics(key_words, top_n=top_n)
            lambda_param = 0.5
            x = np.array([i for i in range(top_n)])
            pdf = lambda_param * np.exp(-lambda_param * x)
            pdf = pdf/sum(pdf)
            if decay_similarity:
                weight = similarity * pdf
            else:
                weight = similarity

            # assign values to p_distr
            for i in range(len(weight)):
                p_distr[similar_topics[i]]=weight[i]
        
        p_distr = p_distr/sum(p_distr)

        if verbose:
            plt.plot(p_distr)
            plt.xlabel("Topic id")
            plt.ylabel("Probability")
            plt.title("target distribution of keywords:{}".format(sum(p_distr)))
            plt.show()

        avg_distr = np.zeros(self.topic_length)
        for k,v in self.distr.items():
            avg_distr +=v

        avg_distr = avg_distr/len(self.distr)

        target_distr = target_w*p_distr+ (1-target_w) * avg_distr
        
        target_distr = target_distr/sum(target_distr)
        if verbose:
            plt.plot(target_distr)
            plt.xlabel("Topic id")
            plt.ylabel("Probability")
            plt.title("target distribution:{}".format(sum(target_distr)))
            plt.show()
        return target_distr
