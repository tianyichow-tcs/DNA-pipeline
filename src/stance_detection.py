
import json
import pandas as pd
from tqdm import tqdm
import networkx as nx
from itertools import combinations,product
from transformers import T5Tokenizer, T5ForConditionalGeneration
from country_list import countries_for_language
import os

class StanceDetection():
    def __init__(self,data_source='covidnews',start=None,end=None):
        self.data_source = data_source
        self.start = start
        self.end = end
        self.root_path = "../data/"

        if self.start == None or self.end == None:
            print("Please input the time window of news articles.")

        self.root_path = "../data/"
        self.post_processing_path= self.root_path+"post_processing/"


        self.data_path = self.post_processing_path+"tokenized/"
        self.file_name = "{data_source}_{start}_{end}_tokenized.json".format(data_source=data_source,start =start.strftime('%Y-%m-%d'),end = end.strftime('%Y-%m-%d'))


        self.story_path = self.root_path+"/story/"
        self.story_daily_file = "{data_source}_{start}_{end}_story_daily.json".format(data_source=data_source,start =start.strftime('%Y-%m-%d'),end = end.strftime('%Y-%m-%d'))
        self.ustory_file = "{data_source}_{start}_{end}_ustory.json".format(data_source=data_source,start =start.strftime('%Y-%m-%d'),end = end.strftime('%Y-%m-%d'))

        self.stance_target_path = self.root_path+"/stance_target/"
        self.stance_target_file = "{data_source}_{start}_{end}_stance_targets.json".format(data_source=data_source,start =start.strftime('%Y-%m-%d'),end = end.strftime('%Y-%m-%d'))
        self.stance_target_dir = os.listdir(self.stance_target_path)

        if self.stance_target_file in self.stance_target_dir:
            print("Find {} in directory, loading...".format(self.stance_target_file))
            self.read_stance_target(self.stance_target_file)

        self.article_df = pd.read_json(self.data_path+self.file_name)   

    def get_target_entities(self,k=0.95):
        def select_top_keys(dictionary, k):
            # Step 1: Calculate the total value

            total_value = sum(dictionary.values())

            # Step 2: Sort the dictionary items by values in descending order
            sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

            # Step 3: Iterate and accumulate values until reaching the desired percentage
            selected_keys = []
            accumulated_value = 0
            for key, value in sorted_items:
                accumulated_value += value
                selected_keys.append(key)
                if (accumulated_value / total_value) >= k:
                    break

            return selected_keys
        
        candidate_dict = {k:self.target_dict[k] for k in self.entity_types} # make sure it is lower case 
        self.target_for_stance = {}

        for e_type,e_dict in candidate_dict.items():
            selected_keys = select_top_keys(e_dict, k)
            print("In entity type {}, {} out of {} entites are selected, {:.2f} entities cover {} of totoal entity occurence. ".format(e_type,len(selected_keys),len(e_dict),  len(selected_keys)/len(e_dict), k))
            self.target_for_stance[e_type] = selected_keys

    def article_target_entity_statistics(self,stop_words=None):
        if stop_words is not None:
            self.entity_stop_words+= stop_words
        self.article_entities={}
        # self.article_entities_org={}
        for idx,row in tqdm(self.article_df.iterrows(),total=len(self.article_df)):
            id = str(row["id"])
            # self.article_entities_org[id] = list(set(entities_list_transered))
            self.article_entities[id] = row["target_entity"]


    def _stats_artitile_entities(self):
        count = 0 
        counts = []
        subset_article_entities = {}
        for k,v in self.article_entities.items():
            if len(v) > 0:
                count +=1
                counts.append(len(v))
                subset_article_entities[k] = v
        ratio = count/len(self.article_entities.values())
        print("In all {} articles, {}({:.2f}%) articles contains at least one entity in target list".format(len(self.article_entities),count,ratio*100) )

    def article_graphs(self,clusters): 
        if clusters is not None:
            for node, edges in clusters.items():
                #G.add_node(node)
                edges = [str(e) for e in edges]
                self.G.add_edges_from(list(product([node],edges)))


    def story_entitie_mining(self,article_entities, method="alignment",stop_words = None):
        self.story_entities_dict = {}
        if method == "aligenment":
            self.stories = self.aligned_story
        else: 
            cluster_keywords = pd.read_json(self.story_path+self.ustory_file,dtype={"id":object,"cluster":object})
            ustory={}
            for idx,row in cluster_keywords.iterrows():
                story_id = row["cluster"]
                article_id = row["id"]
                if story_id not in ustory.keys():
                    ustory[story_id] = []
                    ustory[story_id].append(str(article_id))
                else:
                    ustory[story_id].append(str(article_id))
    
            self.stories = ustory

        for idx,story in tqdm(self.stories.items(),total=len(self.stories)):
            story_entities = {}
            entities_count ={}
            count_entities = 0
            ratio = len(story)  # number of articles has at least 1 entity in this story.
            article_count = 0
            count = 0
            agreement = 0
            for article in story:
                article_count+=1
                story_entities[article] = article_entities[article]
                count_entities+= len(article_entities[article])
                if len(article_entities[article])>0:
                    count+=1
                for e in article_entities[article]:
                    if e not in entities_count.keys():
                        entities_count[e] = 1
                    else:
                        entities_count[e]+=1
                if not count_entities==0:
                    agreement = max(entities_count.values())/article_count

                entities_count_sorted = sorted(entities_count.items(), key=lambda x:x[1],reverse=True)
            dict = {"ratio":count/ratio,"agreement":agreement,"article_count":article_count,"count_entities":count_entities,"entities_count":entities_count_sorted,"story_entities":story_entities}
            self.story_entities_dict[str(idx)]= dict
    
    def select_stories(self,a=0.9,r=0.6,min_article=2,max_article = 200):
        self.selected_story = {}
        for k,story in self.story_entities_dict.items():
            if story["agreement"]>=a and story["ratio"]>=r and story["article_count"] > min_article and story["article_count"]<max_article:
                self.selected_story[k] = story

        #parse selected stories and 
        average_num_entities = 0
        for k,v in self.selected_story.items():
            target_entites = []
            for e,c in  v["entities_count"]:
                if c/v["article_count"] >a:
                    target_entites.append(e)
            self.selected_story[k]["target_entites"] = target_entites
            average_num_entities+= len(target_entites)
        count=0
        for k,v in self.selected_story.items():
            count+=v["article_count"]
        print("total articles {},{} out of {} stories are selected, average number of target entities per story is {}".format(count,len(self.selected_story),len(self.story_entities_dict),average_num_entities/len(self.selected_story)))
        # self.save_stance_target(self.stance_target_file)
        return average_num_entities/len(self.selected_story)
    def save_stance_target(self,file_name):

        with open(self.stance_target_path+file_name, "w") as outfile:
            json.dump(self.selected_story, outfile)
            print("{} has been saved!".format(file_name))

    def read_stance_target(self,file_name):
        f = open(self.stance_target_path+file_name)
        a = json.load(f)
        self.selected_story = a




    def get_article_stance(self):
        # change it to continue the detection when 
        def stance_detection(target,text):
            input_text= "Given a news article, what is the stance of the news article below with respect to '{target}'?  If we can infer from the news article that the news supports '{target}', please label it as 'in-favor'. If we can infer from the news article that the news is against '{target}', please label it as 'against'. If we can infer from the news article that the news has a neutral stance towards '{target}', please label it as 'neutral-or-unclear'. If there is no clue in the news article to reveal the stance of the news article towards '{target}', please also label is as 'neutral-or-unclear'. Please use exactly one word from the following 3 categories to label it: 'in-favor', 'against', 'neutral-or-unclear'. Here is the news article. '{text}' The stance of the news article is: ".format(target=target,text =text)    
            # print(input_text)
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=1000)
            result = tokenizer.decode(outputs[0])
            return result[6:-4]
        
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl",legacy=False)
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")
        self.stance_target_withstance_file = "{data_source}_{start}_{end}_stance_targets_withstance.json".format(data_source=self.data_source,start =self.start.strftime('%Y-%m-%d'),end = self.end.strftime('%Y-%m-%d'))
        # check if stance_target_withstance_file exist
    
        if self.stance_target_withstance_file in self.stance_target_dir:
            self.read_stance_target(self.stance_target_withstance_file)

        for story_id,v in tqdm(self.selected_story.items(),total=len(self.selected_story)):
            print("Processing {} ...".format(story_id))
            if "story_entities_stance" in v.keys():
                print("{} Done!".format(story_id))
                continue
            target_entites = v["target_entites"]
            story_entities = v["story_entities"]
            story_entities_stance  ={k:{}for k in story_entities.keys() }
            df = self.article_df[self.article_df.id.isin(story_entities.keys())]
            for idx,row in df.iterrows():
                article_id = row["id"]
                text = " ".join(row['sentences'])
                text_len = len(text.split(" "))
                if text_len > 1000:
                    for target in target_entites:
                        story_entities_stance[article_id][target]= "None"
                    continue
                for target in target_entites:
                    if target in story_entities[article_id]:
                        stance = stance_detection(target,text)
                        story_entities_stance[article_id][target]= stance
                    else:
                        story_entities_stance[article_id][target]= "None"

            self.selected_story[story_id]["story_entities_stance"] = story_entities_stance
            
            # every story, save the file.
            self.save_stance_target(self.stance_target_withstance_file)
            print(story_id," are saved!")




    def get_article_stance_all_entity(self):
        # change it to continue the detection when 
        def stance_detection(target,text):
            input_text= "Given a news article, what is the stance of the news article below with respect to '{target}'?  If we can infer from the news article that the news supports '{target}', please label it as 'in-favor'. If we can infer from the news article that the news is against '{target}', please label it as 'against'. If we can infer from the news article that the news has a neutral stance towards '{target}', please label it as 'neutral-or-unclear'. If there is no clue in the news article to reveal the stance of the news article towards '{target}', please also label is as 'neutral-or-unclear'. Please use exactly one word from the following 3 categories to label it: 'in-favor', 'against', 'neutral-or-unclear'. Here is the news article. '{text}' The stance of the news article is: ".format(target=target,text =text)    
            # print(input_text)
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=1000)
            result = tokenizer.decode(outputs[0])
            return result[6:-4]
        
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl",legacy=False)
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")
        self.stance_target_withstance_all_entity_file = "{data_source}_{start}_{end}_stance_targets_withstance_all_entity.json".format(data_source=self.data_source,start =self.start.strftime('%Y-%m-%d'),end = self.end.strftime('%Y-%m-%d'))
        # check if stance_target_withstance_file exist
    
        if self.stance_target_withstance_all_entity_file in self.stance_target_dir:
            self.read_stance_target(self.stance_target_withstance_all_entity_file)

        for story_id,v in tqdm(self.selected_story.items(),total=len(self.selected_story)):
            print("Processing {} ...".format(story_id))
            if "story_entities_stance" in v.keys():
                print("{} Done!".format(story_id))
                continue
            # target_entites = v["target_entites"]
            story_entities = v["story_entities"]
            story_entities_stance  ={k:{}for k in story_entities.keys() }
            df = self.article_df[self.article_df.id.isin(story_entities.keys())]
            for idx,row in df.iterrows():
                article_id = row["id"]
                text = " ".join(row['sentences'])
                target_entites = story_entities[article_id]
                for target in target_entites:
                    stance = stance_detection(target,text)
                    story_entities_stance[article_id][target]= stance

            self.selected_story[story_id]["story_entities_stance"] = story_entities_stance
            
            # every story, save the file.
            self.save_stance_target(self.stance_target_withstance_all_entity_file)
            print(story_id," are saved!")
