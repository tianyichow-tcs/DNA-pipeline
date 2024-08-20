

import pandas as pd
import json
from datetime import date, timedelta
import os
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
from tqdm import tqdm

class DataProcessing():
    """
    read and process raw data
    """
    def __init__(self,data_source="covidnews",start=None,end=None,test=False):
        self.start = start
        self.end = end
        self.test = test
        
        if self.start == None or self.end == None:
            print("Please input the time window of news articles.")

        if self.test:
            self.start = pd.Timestamp('2019-01-01',tz='UTC')
            self.end = pd.Timestamp('2022-01-31',tz='UTC')

        self.root_path = "../data/"

        self.path_after_process = self.root_path + "post_processing/"
        self.path_raw = self.root_path+"raw_data/"

        self.raw_dir = os.listdir(self.path_raw)
        self.after_process_dir = os.listdir(self.path_after_process)
        
        self.file_name = "{}_{start}_{end}.json".format(data_source,start =start.strftime('%Y-%m-%d'),end = end.strftime('%Y-%m-%d'))
        self.file_path = self.path_raw+self.file_name
        
        if self.file_name not in self.raw_dir:
            self.read_covid_news()
            self.save_data(self.file_path)
        else:
            print("Find raw data, loading....")
            self.df_metadata = pd.read_json(self.file_path)
            print(self.file_name+" loaded!")

    def read_covid_news(self):
        # This function is used to read nessesary data columns from covide news dataset.
        columns = ["id","title","body","keywords","hashtags","published_at","source_domain","entities_body","entities_title","sentiment"] 
        col1 = ["id","title","body","keywords","hashtags","published_at"]
        col2 = ["source_domain"]
        low=100
        upper=400
        get_entity = True
        get_sentiment =True
        meta_data = []
        def read_large_data(file_path):
            count = 0
            read_count = 0
            print("Loading from raw covide news data...")
            with open(file_path, 'r') as json_file:
                while True:
                    row = []
                    data = json_file.readline()
                    if data:
                        count += 1
                        results = json.loads(data)
                        published_at = pd.to_datetime(
                            results["published_at"], dayfirst=False
                        )
                        words_count = results["words_count"]
                        if published_at < self.start or published_at > self.end:
                            continue
                        if words_count < low or words_count > upper:
                            continue  # pass this log
                        for key in col1:
                            row.append(results[key])
                        for key in col2:
                            key1, key2 = key.split("_")
                            row.append(results[key1][key2])
                        read_count+=1
                        # parse entities
                        if get_entity:
                            # extract entities
                            body_entities = dict()
                            for entity in results["entities"]['body']:
                                text = entity["text"]
                                body_entities[text] = entity["types"]
                            row.append(body_entities)

                            title_entities = dict()
                            for entity in results["entities"]['title']:
                                text = entity["text"]
                                title_entities[text] = entity["types"]
                            row.append(title_entities)
                        if get_sentiment:
                            row.append(results["sentiment"])
                    else:
                        print("Total number of news:",count, " Loaded number of news:",read_count," ",100*read_count/count,"%")
                        return
                    meta_data.append(row)
        
        data_path = "aylien_covid_news_data.jsonl" # the data is avialable online
        read_large_data(data_path)
        self.df_metadata = pd.DataFrame(meta_data, columns=columns)
        self.df_metadata.dropna(inplace=True)
        self.df_metadata=self.df_metadata.sort_values(by='published_at').reset_index(drop=True)


    def save_data(self,file_path):
        self.df_metadata["id"] = self.df_metadata.id.astype(str)
        self.df_metadata.to_json(file_path)



class Preprocessing():
    """
      tokenizer of text data, in tikenized path
      embedding for tokenized data, in embedding data
    """

    def __init__(self,data_source,start,end,tokenization=True, embedding=True):
        self.nlp = spacy.load("en_core_web_sm")
        dp = DataProcessing(data_source,start,end)
        self.df_metadata = dp.df_metadata
        self.root_path = "../data/post_processing/"
        self.tokenized_path = self.root_path + "tokenized/"
        self.embedding_path = self.root_path+"embedding/"
        filename = dp.file_name.split(".")[0]
        self.tokenized_filename = filename + "_tokenized.json"
        self.embedding_filename = filename + "_embedding.json"

        self.tokenized_file_dir = os.listdir(self.tokenized_path)
        self.embedding_path_dir = os.listdir(self.embedding_path)
    
        
        self.article_df = pd.DataFrame()
        self.article_df["id"] = self.df_metadata.id
        self.article_df["date"] = self.df_metadata.published_at

        if tokenization:
            if self.tokenized_filename not in self.tokenized_file_dir:
                print("Tokenization in progress...")
                self.get_tokenization()
                self.save_tokenized_data()
            else:
                self.article_df = pd.read_json(self.tokenized_path+self.tokenized_filename)
                print(self.tokenized_filename, " exist, file loaded.")

        if embedding:
            if self.embedding_filename not in self.embedding_path_dir:
                print("Embedding in progress...")
                self.get_embeddings()
                self.save_embedding_data()
            else:
                self.article_df = pd.read_json(self.embedding_path+self.embedding_filename)
                print(self.embedding_filename, " exist, file loaded.")

    def spacy_tokenizer(self, doc):
        tokens = self.nlp(doc)
        return([token.lemma_.lower() for token in tokens if (token.text.isalnum() and not token.is_stop and not token.is_punct and not token.like_num)])
    
    def get_tokenization(self):
        # tokenize news file.
        self.article_df['sentences'] = [[t] for t in self.df_metadata.title]
        self.article_df['sentence_counts'] = ""
        self.article_df['sentence_tokens'] = [[self.spacy_tokenizer(t)] for t in self.df_metadata.title]
        
        all_sentences = []
        all_sentence_tokens = []
        for text in tqdm(self.df_metadata['body'].values):
            parsed = self.nlp(text)
            sentences = []
            sentence_tokens = []
            for s in parsed.sents:
                if len(s) > 1:
                    sentences.append(s.text)
                    sentence_tokens.append([token.lemma_.lower() for token in s if (token.text.isalnum() and not token.is_stop and not token.is_punct and not token.like_num)])
            all_sentences.append(sentences)
            all_sentence_tokens.append(sentence_tokens)

        for i in range(len(all_sentences)):
            self.article_df.at[i,'sentences'] = self.article_df.loc[i].sentences + all_sentences[i]
            self.article_df.at[i,'sentence_tokens'] = self.article_df.loc[i].sentence_tokens + all_sentence_tokens[i]
            self.article_df.at[i,'sentence_counts'] = len(self.article_df.loc[i].sentences)


    def save_tokenized_data(self):
        self.article_df.to_json(self.tokenized_path+self.tokenized_filename)
        print(self.tokenized_filename, " saved!")
    
    def get_embeddings(self,model_name ="sentence-transformers/all-MiniLM-L6-v2"):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Devices using ",device)
        st_model = SentenceTransformer(model_name,device=device) 
        self.article_embedding_df = self.article_df[["id"]]
        embeddings = []
        errors = []
        k=0
        for sentences in tqdm(self.article_df['sentences']):
            try:
                embedding = st_model.encode(sentences)
                embeddings.append(embedding)
            except Exception as e:
                errors.append(k)
                print("error at", self.article_df.id[k], e)
            k = k + 1
        self.article_df['sentence_embds'] = embeddings

    def save_embedding_data(self):
        self.article_df.to_json(self.embedding_path+self.embedding_filename)
        print(self.embedding_filename, " saved!")