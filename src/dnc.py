import json
from tqdm import tqdm
from stance_detection import *
from topic_modeling import *
import omp
import random
import heapq
import math
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def hellinger_distance(p, q):
    """
    Compute the Hellinger distance between two probability distributions.

    Args:
        p (numpy.ndarray): The probability distribution P.
        q (numpy.ndarray): The probability distribution Q.
    Returns:
        float: The Hellinger distance between P and Q.
    """
    if len(p) != len(q):
        raise ValueError("Input distributions must have the same length.")
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)
    h = 0.5 * np.sum((sqrt_p - sqrt_q) ** 2)
    return h

def overlap_measure(p,q):

    if len(p) != len(q):
        raise ValueError("Input distributions must have the same length.")
    sqrt_p = np.sqrt(p)
    sqrt_q = np.sqrt(q)
    Gf = np.multiply(sqrt_p,sqrt_q).sum()
    return Gf


class DiversifyNewsCoverage():
    def __init__(self,data_source="covidnews",start=None,end=None):
        self.data_source = data_source
        self.start = start
        self.end = end
        self.root_path = "../data/"
        self.post_processing_path= self.root_path+"post_processing/"
        self.data_path = self.post_processing_path+"tokenized/"
        self.file_name = "{data_source}_{start}_{end}_tokenized.json".format(data_source=data_source,start =start.strftime('%Y-%m-%d'),end = end.strftime('%Y-%m-%d'))

        self.stance_path = self.root_path +"stance_target/"
        self.selected_story_file_name = "{data_source}_{start}_{end}_stance_targets_withstance.json".format(data_source=self.data_source,start =start.strftime('%Y-%m-%d'),end = end.strftime('%Y-%m-%d')) # stance toward entity 
        
        #load stance file
        f = open(self.stance_path+self.selected_story_file_name)
        self.selected_story = json.load(f)

        entity_types = ["Politician","OfficeHolder","PoliticalParty","Party"]
        if self.data_source=="nela" or self.data_source=="wcep":
            self.article_df = pd.read_json(self.data_path+self.file_name)   

        else:
            self.sd= StanceDetection(start,end,entity_types)
            self.article_df = self.sd.article_df

        # stance detection is not complete, choose stories one with stance.
        # self.get_selected_story()
        self.ge_article_list()
        df = self.article_df[self.article_df.id.isin(self.articles_id_list)]
        self.text = [" ".join(a) for a in df.sentences]
        df_data = pd.DataFrame()
        df_data["id"]=df["id"]
        df_data["text"] = self.text
        self.df_data = df_data.drop_duplicates()

        self.topic_distribution_path = self.root_path+ "topic_model/topic_distribution/"
        self.target_distribution_path = self.root_path+ "topic_model/target_distribution/"
        self.topic_distribution_file =  "{data_source}_{start}_{end}_topic_distribution.pkl".format(data_source=self.data_source,start =self.start.strftime('%Y-%m-%d'),end =self.end.strftime('%Y-%m-%d'))
        self.target_distribution_file =  "{data_source}_{start}_{end}_target_distribution.pkl".format(data_source=self.data_source,start =self.start.strftime('%Y-%m-%d'),end =self.end.strftime('%Y-%m-%d'))
        self.load_distribution()

        self.get_total_articles()

        # bias-sensitive partition
        self.collection_hittingset(collection_method="all")

        self.T = np.asarray(list(self.topic_distribution.values()))
        self.H = self.hittingset2matrix(self.hittingset_collection)
    
    def collection_hittingset(self,collection_method = "story"):
        print("collection construction method:",collection_method)
        if collection_method =="entity":
            self.get_collection_entity()
        elif collection_method =="story":
            self.get_collection_story()
        elif collection_method == "all":
            self.get_collection()
        self.collection2hittingset()


    def get_total_articles(self):
        article_count = 0
        for story_id, story in self.selected_story.items():
            article_count += story["article_count"]
        self.article_count= article_count

    def load_distribution(self):
        with open(self.topic_distribution_path+self.topic_distribution_file, 'rb') as f:
            self.topic_distribution = pickle.load(f)
        with open(self.target_distribution_path+self.target_distribution_file, 'rb') as f:
            self.target_distribution = pickle.load(f)
        self.topic_length = len(self.topic_distribution[list(self.topic_distribution.keys())[1]])

    def ge_article_list(self):
        articles_id_list = []
        for story_id,story in self.selected_story.items():
            artilce_id = list(story["story_entities"].keys())
            articles_id_list+= artilce_id
        self.articles_id_list = articles_id_list

    def get_collection(self):
        collection_id = 0
        collection = {}
        for story_id, story in self.selected_story.items():
            article_ids = list(story["story_entities"].keys())
            df =self.article_df[self.article_df.id.isin(story["story_entities"].keys())]
            group_df_bias = df.groupby(pd.Grouper(key="bias"))
            stance_dict = story["story_entities_stance"]
            for bias,df_bias in group_df_bias:
                # dict_bias[bias] = df_bias.id.to_list()
                if len(df_bias)>0:
                    if df_bias.bias.to_list()[0] != "None":
                        collection[collection_id] = set(df_bias.id.to_list())
                        collection_id+=1
                # collection of 
            for target in story["target_entites"]:
                stance_collection = {"in-favor":[],"neutral-or-unclear":[],"against":[]}
                for k,v in stance_dict.items():
                    if target in v.keys():
                        if v[target] != "None":
                            stance_collection[v[target]].append(k)
                for k,v in stance_collection.items():
                    if len(stance_collection[k])>0:
                        collection[collection_id] = set(stance_collection[k])
                        collection_id+=1
            result = {}
            i = 0
            for key,value in collection.items():
                if value not in result.values():
                    result[i] = value
                    i+=1
        print("Number of collections: {}, number of identical collection {}".format(len(collection), len(result)))
        self.collection = result
    
    def get_collection_story(self):
        collection_id = 0
        collection = {}
        for story_id, story in self.selected_story.items():
            article_ids = list(story["story_entities"].keys())
            df =self.article_df[self.article_df.id.isin(story["story_entities"].keys())]
            group_df_bias = df.groupby(pd.Grouper(key="bias"))
            stance_dict = story["story_entities_stance"]
            for bias,df_bias in group_df_bias:
                # dict_bias[bias] = df_bias.id.to_list()
                if len(df_bias)>0:
                    if df_bias.bias.to_list()[0] != "None":
                        collection[collection_id] = set(df_bias.id.to_list())
                        collection_id+=1
            result = {}
            i = 0
            for key,value in collection.items():
                if value not in result.values():
                    result[i] = value
                    i+=1
        print("Number of collections: {}, number of identical collection {}".format(len(collection), len(result)))
        self.collection = result

    def get_collection_entity(self):
        ## construct collection without story.
        ## cover entity stance, cover entity bias
        all_target_entity = {}
        for story_id, story in self.selected_story.items():
            story_entities_stance = story["story_entities_stance"]
            for a_id,e_stance_pairs in story_entities_stance.items():
                for e,stance in e_stance_pairs.items():
                    # print(stance)
                    if stance == "None":
                        continue
                    if e in all_target_entity.keys():
                        if stance in all_target_entity[e].keys(): 
                            all_target_entity[e][stance].append(a_id)
                        else:
                            all_target_entity[e][stance] = []
                            all_target_entity[e][stance].append(a_id)
                    else:
                        all_target_entity[e] = {}
                        all_target_entity[e][stance] = []
                        all_target_entity[e][stance].append(a_id)
        # print some info
        print ("#Entities:",len(all_target_entity))

        # construct collection
        collection_id = 0
        collection = {}
        
        for e, stance_dict in all_target_entity.items():
            e_id=0
            a_ids = []
            # print("stance cates",len(stance_dict))
            for stance, a_id in stance_dict.items():
                collection[collection_id] = set(a_id)
                collection_id+=1
                a_ids += a_id
                e_id+=1
                
            df =self.article_df[self.article_df.id.isin(a_ids)]
            group_df_bias = df.groupby(pd.Grouper(key="bias"))
            for bias,df_bias in group_df_bias:
                # print("bias category:",len(group_df_bias))
                # dict_bias[bias] = df_bias.id.to_list()
                if len(df_bias)>0:
                    if df_bias.bias.to_list()[0] != "None":                
                        collection[collection_id] = set(df_bias.id.to_list())
                        collection_id+=1
                        e_id+=1
            # print(e," ",e_id)
        
        result = {}
        i = 0
        for key,value in collection.items():
            if value not in result.values():
                result[i] = value
                i+=1
        print("Number of collections: {}, number of identical collection {}".format(len(collection), len(result)))
        self.collection = result



    def collection2hittingset(self):
        uncovered_sets = set(self.collection.keys())
        hittingset_collection = {}
        total_article = set()
        for k,v in self.collection.items():
            total_article= total_article.union(v)
            for article_id in v:
                if article_id not in hittingset_collection.keys():
                    hittingset_collection[article_id]=set()
                    hittingset_collection[article_id].add(k)
                else:
                    hittingset_collection[article_id].add(k)

        self.hittingset_collection = hittingset_collection
        self.uncovered_sets = uncovered_sets
        self.total_article = total_article  #total articles is the number of articles in collections
    
    def plot_collections(self):
        plt.plot(sorted([v["article_count"] for k,v in self.selected_story.items()],reverse=True))
        plt.xlabel("story id")
        plt.ylabel("number of articles")
        plt.title("{} stories, {} articles".format(len(self.selected_story), len(self.df_data)))
        plt.show()

        plt.plot(sorted([len(v) for k,v in self.collection.items()],reverse=True))
        plt.xlabel("collection id")
        plt.ylabel("number of articles")
        plt.title("{} stories, {} articles".format(len(self.selected_story), len(self.df_data)))
        plt.show()

        set_size= sorted([len(v) for k,v in self.hittingset_collection.items()],reverse=True)
        plt.plot(set_size)
        plt.xlabel("article id")
        plt.ylabel("number of collections the article hits")
        plt.title("{} stories, {} articles".format(len(self.selected_story), len(self.df_data)))


    def hittingset2matrix(self, my_dict):
        dict_ids = list(self.df_data.id.to_list())
        # max_index = max(max(val) for val in my_dict.values())
        max_index = len(self.collection)
        # Create a binary matrix filled with zeros
        num_rows = max_index
        num_cols = len(dict_ids)
        binary_matrix = [[0] * num_cols for _ in range(num_rows)]

        # Fill the binary matrix with 1s where values exist
        for idx, dict_id in enumerate(dict_ids):
            if dict_id not in my_dict.keys():
                continue
            for val_idx in my_dict[dict_id]:
                binary_matrix[val_idx][idx] = 1
        return np.asarray(binary_matrix)
    
    def remove_duplicate_columns(self, matrix):
        # Transpose the matrix to work with columns
        transposed_matrix = [list(row) for row in zip(*matrix)]
        
        # Create a dictionary to store column counts
        column_counts = {}
        column_index = []
        # Initialize the result matrix
        result_matrix = []
        i = 0
        for column in transposed_matrix:
            # Convert the column to a tuple for hashability
            column_tuple = tuple(column)
            
            # If the column is not in the dictionary, add it with count 1
            if column_tuple not in column_counts:
                column_counts[column_tuple] = 1
                column_index.append(i)
                result_matrix.append(list(column))
            else:
                # Increment the count in the dictionary
                column_counts[column_tuple] += 1
            i+=1
        # Transpose the result matrix back to original form
        result_matrix = [list(row) for row in zip(*result_matrix)]
        
        return result_matrix, column_counts,column_index
    

    def data2Binarymatrix(self):
        # initialize article id vector 
        self.dict_B = {id:None for id in self.df_data.id}
        self.dict_S = {id:None for id in self.df_data.id}
        dict_f = {}
        dict_b = {}
        
        
        #treat each story independently
        for story_id, story in self.selected_story.items():
            stances = ["in-favor","neutral-or-unclear","against"]
            bias =  ['LEFT-CENTER', 'LEFT', 'RIGHT-CENTER', 'CENTER', 'RIGHT']
            target_entites = story["target_entites"]
            
            #initialize story-bias dictionary
            for b in bias:
                key_bias = "{}_{}".format(story_id, b)
                dict_b[key_bias] = 0
            
            #initialize story-stance dictionary
            for e in target_entites:
                for stance in stances:
                    key = "{}_{}_{}".format(story_id,e,stance)
                    # str(story_id)+"" +e+" "+stance
                    if key not in dict_f.keys():
                        dict_f[key] = 0

        # iterate story to generate stance vector
        for story_id, story in self.selected_story.items():
            stance_dict = story["story_entities_stance"]
            for k,v in stance_dict.items():
                feature = dict_f.copy()
                for e, stance in v.items():
                    if stance != "None":
                        key = "{}_{}_{}".format(story_id,e,stance)
                        feature[key] = 1
                # if feature == dict_f:
                #     print (k, " ", v)
                self.dict_S[k] = list(feature.values())

        # iterate story to generate stance vector
        for story_id, story in self.selected_story.items():
            df = self.article_df[self.article_df.id.isin(story["story_entities"].keys())].replace(np.nan, "None")
            for idx, row in  df.iterrows():
                feature = dict_b.copy()
                id =row['id']
                bias = row['bias']
                if bias !="None":
                    if bias not in ['LEFT-CENTER', 'LEFT', 'RIGHT-CENTER', 'CENTER', 'RIGHT']:
                        print(bias)
                    key_bias = "{}_{}".format(story_id, bias)
                    feature[key_bias] = 1
                self.dict_B[id] = list(feature.values())
        

        # get matrix representation of stance and bias
        self.M_S = np.array([self.dict_S[i] for i in self.dict_S.keys()]).T
        self.M_B = np.array([self.dict_B[i] for i in self.dict_B.keys()]).T

        def get_target(M, a=0.5):
            tau = M.sum(axis=1)/M.shape[0]
            return tau
        # get target distribution of stance and bias.
        self.tau_MB = get_target(self.M_B)
        self.tau_MS = get_target(self.M_S)
        


        result_matrix, column_counts,c_idx = self.remove_duplicate_columns(self.M_S)
        self.M_S_r = np.asarray(result_matrix)
        self.c_s = np.asarray(list(column_counts.values()))
        self.MS_articleset =np.asarray(self.df_data.id.to_list())[c_idx]


        result_matrix, column_counts,c_idx = self.remove_duplicate_columns(self.M_B)
        self.M_B_r = np.asarray(result_matrix)
        self.c_b = np.asarray(list(column_counts.values()))
        self.MB_articleset =np.asarray(self.df_data.id.to_list())[c_idx]

        return self.M_S,self.M_B,self.tau_MS,self.tau_MB,self.M_S_r,self.c_s,self.M_B_r,self.c_b,self.MS_articleset,self.MB_articleset
        # # return dict_B,dict_S,dict_b

    #ALG with parameter tunning
    def greedy_algorithm_both_beta(self,p,w, K,beta,verbose=False):
        vector_dict = self.topic_distribution
        q = np.zeros(self.topic_length)
        hittingset_collection = self.hittingset_collection.copy()
        uncovered_sets = self.uncovered_sets.copy()

        selected_articles = set()
        Gfs = []
        k = 1
        len_set = len(uncovered_sets)

        cover_progress = [0]
        for i in tqdm(range(K)):
            # print ("This is step ",i," :")
            df = pd.DataFrame()
            overlap={}
            for article_id,v in hittingset_collection.items():
                intersection = uncovered_sets.intersection(v)
                overlap[article_id] = len(intersection)
            df["id"] = overlap.keys()
            df["hits"] = overlap.values()
            
            marginal_gains = {}
            for article_id in df.id:  # find the item yield the largest marginal gain.
                q_j = np.asarray(vector_dict[article_id])
                q_temp = q +  w[i]*q_j
                marginal_gain = overlap_measure(p,q_temp) - overlap_measure(p,q)
                marginal_gains[article_id] = marginal_gain
        
            df["gains"] = marginal_gains.values()
   
            df_sorted = df.sort_values(by=['hits','gains'],ascending=False)


            grouped = df_sorted.groupby("hits").first().sort_values("hits",ascending=False)
            ## minimize the sum of the index.
            gains = grouped.gains.to_list()
            gain_max = max(gains)
            hits = [i+1 for i in grouped.index.to_list()]
            hits_max = max(hits)
            grouped["gain_norm"] = np.asarray(gains)/gain_max
            grouped["hits_norm"] = np.asarray(hits)/hits_max
            grouped["score"] = beta*grouped.gain_norm + (1-beta)*grouped.hits_norm
            sorted_grouped = grouped.sort_values("score",ascending=False)
            selected_id = sorted_grouped.id.to_list()[0]

            intersection = uncovered_sets.intersection(hittingset_collection[selected_id])
            uncovered_sets = uncovered_sets-intersection
            c = (len_set - len(uncovered_sets))/len_set
            cover_progress.append(c)
            # if verbose:
            #     print("current coverage of collections:",c)

            hittingset_collection.pop(selected_id)
            selected_articles.add(selected_id)
            q = q +  w[i]*np.asarray(vector_dict[selected_id])
            Gfs.append(overlap_measure(p,q))
            k+=1

        if verbose:
            plt.plot(cover_progress)
            plt.xlabel("number of article selected")
            plt.ylabel("coverage ratio of the collections")
            plt.show()

            plt.xlabel("Number of articles selected")
            plt.ylabel("Overlap measure score")
            plt.plot(Gfs)
            plt.show()

            plt.plot(q)
            plt.plot(p)
            plt.legend(["q:composed","p:target"])
            plt.xlabel("Topic")
            plt.ylabel("Distribution")
            plt.show()

            print ("coverage rate: ", cover_progress[-1], " Overlap measure:",Gfs[-1] )
        return selected_articles, q,Gfs,cover_progress

    #ALG
    def greedy_algorithm_both(self,p,w, K,prioritise="hits",verbose=False):

        vector_dict = self.topic_distribution
        q = np.zeros(self.topic_length)
        hittingset_collection = self.hittingset_collection.copy()
        uncovered_sets = self.uncovered_sets.copy()

        selected_articles = set()
        Gfs = []
        k = 1
        len_set = len(uncovered_sets)
  
        cover_progress = [0]
        for i in tqdm(range(K)):
            df = pd.DataFrame()
            overlap={}
            for article_id,v in hittingset_collection.items():
                intersection = uncovered_sets.intersection(v)
                overlap[article_id] = len(intersection)
            df["id"] = overlap.keys()
            df["hits"] = overlap.values()
            
            marginal_gains = {}
            for article_id in df.id:  # find the item yield the largest marginal gain.
                q_j = np.asarray(vector_dict[article_id])
                q_temp = q +  w[i]*q_j
                marginal_gain = overlap_measure(p,q_temp) - overlap_measure(p,q)
                marginal_gains[article_id] = marginal_gain
        
            df["gains"] = marginal_gains.values()

            if prioritise == "hits":
                df_sorted = df.sort_values(by=['hits','gains'],ascending=False)
            else :
                df_sorted = df.sort_values(by=['gains','hits'],ascending=False)

            grouped = df_sorted.groupby("hits").first().sort_values("hits",ascending=False)

            gains = grouped.gains.to_list()
            gain_sorted = sorted(gains,reverse=True)
            gains_rank_index = [gain_sorted.index(i) for i in gains]
            grouped["gains_rank_index"] = gains_rank_index
            grouped["hit_index"] = [i for i in range(len(gains))] 
            grouped["score"] = grouped.gains_rank_index + grouped.hit_index
            sorted_grouped = grouped.sort_values("score")
            selected_id = sorted_grouped.id.to_list()[0]

            intersection = uncovered_sets.intersection(hittingset_collection[selected_id])
            uncovered_sets = uncovered_sets-intersection
            c = (len_set - len(uncovered_sets))/len_set
            cover_progress.append(c)

            hittingset_collection.pop(selected_id)
            selected_articles.add(selected_id)
            q = q +  w[i]*np.asarray(vector_dict[selected_id])
            Gfs.append(overlap_measure(p,q))
            k+=1

        if verbose:
            plt.plot(cover_progress)
            plt.xlabel("number of article selected")
            plt.ylabel("coverage ratio of the collections")
            plt.show()

            plt.xlabel("Number of articles selected")
            plt.ylabel("Overlap measure score")
            plt.plot(Gfs)
            plt.show()

            plt.plot(q)
            plt.plot(p)
            plt.legend(["q:composed","p:target"])
            plt.xlabel("Topic")
            plt.ylabel("Distribution")
            plt.show()

            print ("coverage rate: ", cover_progress[-1], " Overlap measure:",Gfs[-1] )
        return selected_articles, q,Gfs,cover_progress
    
    #BL1
    def greedy_algorithm_hitting_set(self,p,verbose=False):
        hittingset_collection = self.hittingset_collection.copy()
        uncovered_sets =self.uncovered_sets.copy()
        total_article = self.total_article.copy()

        selected_articles = set()
        len_set = len(uncovered_sets)
        print(len_set)

        cover_progress = [0]
        num_of_hit = []
        selected_articles_hit = []

        while len(uncovered_sets)>0:
            overlap={}
            max_article = 0
            for k,v in hittingset_collection.items():
                intersection = uncovered_sets.intersection(v)
                overlap[k] = len(intersection)
            overlap_sorted =sorted(overlap.items(), key=lambda x:x[1],reverse=True)
            selected_id,_ = overlap_sorted[0]
            intersection = uncovered_sets.intersection(hittingset_collection[selected_id])
            num_of_hit.append(len(intersection))
            selected_articles_hit.append(len(hittingset_collection[selected_id]))
            uncovered_sets = uncovered_sets-intersection
            c = (len_set - len(uncovered_sets))/len_set
            cover_progress.append(c)
            # print ("Number of hits:",len(intersection), " current coverage:",c)
            hittingset_collection.pop(selected_id)
            selected_articles.add(selected_id)

        q_hitting_set = np.zeros(self.topic_length)
        for a_id in selected_articles:
            q_hitting_set+=self.topic_distribution[a_id]
        q_hitting_set= q_hitting_set/len(selected_articles)
        
        print ("{} out of {} articles are selected to compose the hitting set!".format(len(selected_articles),len(total_article)))
        print("Check the cover_progress to see the cover progress.")
        if verbose:
            plt.plot(cover_progress)
            plt.xlabel("number of article selected")
            plt.ylabel("coverage ratio of the collections")
            plt.show()

            plt.plot(selected_articles_hit)
            plt.xlabel("number of article selected")
            plt.ylabel("size of selected set")
            plt.show()
            print ("coverage rate: ", cover_progress[-1], " Overlap measure:",overlap_measure(p,q_hitting_set))

        return selected_articles,cover_progress,num_of_hit,selected_articles_hit,q_hitting_set,[overlap_measure(p,q_hitting_set)], cover_progress
    
    def composed_distribution(self, selected_articles,topic_length):
        q_hitting_set = np.zeros(topic_length)
        for a_id in selected_articles:
            q_hitting_set+=self.topic_distribution[a_id]
        q_hitting_set= q_hitting_set/len(q_hitting_set)
        return q_hitting_set
    

    
    #BL2
    def greedy_algorithm_calibration(self,p,w,K,verbose=False):
        
        news_set = self.total_article.copy()
        news_info_table = self.topic_distribution
        q = np.zeros(self.topic_length)
        hittingset_collection = self.hittingset_collection.copy()
        uncovered_sets = self.uncovered_sets.copy()

        Q = [] # composed recommendation list
        Gfs =[]

        selected_articles = set()
        len_set = len(uncovered_sets)

        cover_progress = [0]
        num_of_hit = []
        selected_articles_hit = []

        for i in range(K): # for each step, chose the one give the largest marginal gain. K = 300
            Gfs.append(overlap_measure(p,q))
            marginal_gains = []
            news_list = list(news_set)
            for new in news_set:  # find the item yield the largest marginal gain.
                q_j = news_info_table[new]
                q_temp = q + w[i] * q_j
                marginal_gain = overlap_measure(p,q_temp) - overlap_measure(p,q)
                marginal_gains.append(marginal_gain)
            marginal_gains_temp = sorted(marginal_gains)
            max_margal_gain_index = marginal_gains.index(max(marginal_gains))
            select_new = news_list[max_margal_gain_index]
            Q.append(select_new)
            news_set.remove(news_list[max_margal_gain_index])
            q = q + w[i] * news_info_table[select_new]
            selected_id=select_new
            intersection = uncovered_sets.intersection(hittingset_collection[selected_id])
            num_of_hit.append(len(intersection))
            selected_articles_hit.append(len(hittingset_collection[selected_id]))
            uncovered_sets = uncovered_sets-intersection
            c = (len_set - len(uncovered_sets))/len_set
            cover_progress.append(c)
            hittingset_collection.pop(selected_id)
            selected_articles.add(selected_id)

        if verbose:
            plt.plot(cover_progress)
            plt.xlabel("number of article selected")
            plt.ylabel("coverage ratio of the collections")
            plt.show()

            plt.plot(q)
            plt.plot(p)
            plt.legend(["q","p"])
            plt.show()

            plt.xlabel("Number of articles selected")
            plt.ylabel("Overlap measure score")
            plt.plot(Gfs)
            plt.show()
            print ("coverage rate: ", cover_progress[-1], " Overlap measure:",Gfs[-1] )
        return Q,q,Gfs,cover_progress

    #BL3:

##BL3: MMR
    def MMR(self,p,w,K,beta,articles_order,S_similarity,B_similarity,verbose =False):
        news_set = self.total_article.copy()
        news_info_table = self.topic_distribution
        q = np.zeros(self.topic_length)
        hittingset_collection = self.hittingset_collection.copy()
        uncovered_sets = self.uncovered_sets.copy()

        Q = [] # composed recommendation list
        Gfs =[]

        selected_articles = set()
        len_set = len(uncovered_sets)

        cover_progress = [0]
        num_of_hit = []
        selected_articles_hit = []
        

        selected_article_index = []
        for i in tqdm(range(K)): # for each step, chose the one give the largest marginal gain. K = 300
            Gfs.append(overlap_measure(p,q))
            marginal_gains = []
            marginal_gains_norm = []
            mmr_list= []
            max_diversities = []
            news_list = list(news_set)
          
            for a_id in news_set:  # find the item yield the largest marginal gain.
                q_j = news_info_table[a_id]
                q_temp = q + w[i] * q_j
                marginal_gain = overlap_measure(p,q_temp) - overlap_measure(p,q)
                marginal_gains.append(marginal_gain)
                if i>=1:
                    a_id_index = articles_order.index(a_id)
                    a_s_simi_vec= S_similarity[a_id_index][selected_article_index]
                    a_b_simi_vec= B_similarity[a_id_index][selected_article_index]
                    a_simi_vec = (a_s_simi_vec+a_b_simi_vec)/2
                    max_diversity = a_simi_vec.max()
                    max_diversities.append(max_diversity)

            if i>=1:
                marginal_gains_norm = np.asarray(marginal_gains)/abs(max(marginal_gains))
                mmr_list= beta* (marginal_gains_norm) - (1-beta) * (np.asarray(max_diversities))
                mmr_list=list(mmr_list)

            if i >=1:
                max_margal_gain_index = mmr_list.index(max(mmr_list))
            else:
                max_margal_gain_index = marginal_gains.index(max(marginal_gains))

            select_new = news_list[max_margal_gain_index]
            Q.append(select_new)
            news_set.remove(news_list[max_margal_gain_index])
            q = q + w[i] * news_info_table[select_new]
            selected_id=select_new
            intersection = uncovered_sets.intersection(hittingset_collection[selected_id])
            num_of_hit.append(len(intersection))
            selected_articles_hit.append(len(hittingset_collection[selected_id]))
            uncovered_sets = uncovered_sets-intersection
            c = (len_set - len(uncovered_sets))/len_set
            cover_progress.append(c)
            hittingset_collection.pop(selected_id)
            selected_articles.add(selected_id)
            selected_article_index.append(articles_order.index(selected_id))

        if verbose:
            plt.plot(cover_progress)
            plt.xlabel("number of article selected")
            plt.ylabel("coverage ratio of the collections")
            plt.show()

            plt.plot(q)
            plt.plot(p)
            plt.legend(["q","p"])
            plt.show()

            plt.xlabel("Number of articles selected")
            plt.ylabel("Overlap measure score")
            plt.plot(Gfs)
            plt.show()
            print ("coverage rate: ", cover_progress[-1], " Overlap measure:",Gfs[-1] )
        return Q,q,Gfs,cover_progress


    #BL4: Multi-object nomp
    def NOMP(self, M,tau,k):
        result = omp.omp(M, tau, ncoef=k)
        return result

    
    def n_largest_elements(self,vector, n):
        # Create a min heap to maintain the n largest elements
        heap = [(value, index) for index, value in enumerate(vector[:n])]
        heapq.heapify(heap)

        # Iterate through the rest of the elements in the vector
        for i, value in enumerate(vector[n:]):
            if value > heap[0][0]:
                heapq.heappop(heap)
                heapq.heappush(heap, (value, i + n))

        # Sort the heap to get the n largest elements in descending order
        n_largest = sorted(heap, reverse=True)

        # Extract the values and indices
        largest_values = [value for value, index in n_largest]
        largest_indices = [index for value, index in n_largest]
        return largest_values, largest_indices
    

    def integer_linear_programming(self,M, m_c, tau, K):
        Loss= []
        S = {}
        for l in tqdm(range(1,K+1)):
            # print ("L = ", l)
            result = self.NOMP(M,tau,l)
            x = result.coef
        
            x_position = [1 if element > 0 else 0 for element in x]
            v = [value for value in x if value > 0]
            v = [i/sum(v) for i in v]
            c = [m_c[i] for i, val in enumerate(x) if val > 0]

            C = sum(c)
  
            error = []
            for N in tqdm(range(1,C+1)):
                s = [-1 for i in range(len(v))]
                s_position = [0 for i in range(len(v))]

                for i,value in enumerate(v):
                    if c[i]<N*v[i]:
                        s[i] = c[i]
                        s_position[i] = 1
                    
                U = sum(list(map(math.ceil,[N*vi for vi in v]) ))
                L = sum(list(map(math.floor,[N*vi for vi in v] )))
                if N <= L:
                    for i,value in enumerate(v):
                        if s_position[i] ==0:
                            s[i] = math.floor(N*v[i])
                            s_position[i] = 1
                if N >= U:
                    for i,value in enumerate(v):
                        if s_position[i] ==0:
                            s[i] = math.ceil(N*v[i])
                            s_position[i] = 1
                else:
                    X = N-L
                    res = [N*value - math.floor(N*value) for value in v]
                    res_position = [val1 * val2 for val1, val2 in zip(res, s_position)]
                    largest_values, largest_indices = self.n_largest_elements(res, X)
                    for idx in largest_indices:
                        s[idx] = math.ceil(N*v[idx])
                        s_position[idx] = 1
                    for idx,value in enumerate(s_position):
                        if value == 0 :
                            s[idx] = math.floor(N*v[idx])
                            s_position[idx] = 1
                idx = 0
                s_new = [0 for i in range(len(x))]
                for i,value in enumerate(x_position):
                    if value ==1:
                        s_new[i] = s[idx]
                        idx +=1
                s = np.asarray(s_new)
                tau_approx = np.dot(M,s)/sum(np.dot(M,s))

                loss = sum((val1 - val2)**2 for val1, val2 in zip(tau_approx, tau))
                error.append(loss)
                S[str(l)+"_"+str(N)] = s
            Loss.append(error)

        return Loss,S

    # check selected stories
    def look_up_selected_articles(self,selected_articles):

        selected_story_article_dict = {}
        for id, v in self.selected_story.items():
            selected_story_article_dict[id] = list(v["story_entities"].keys()) 

        hitting_story_count = {}
        for id,v in  selected_story_article_dict.items():
            hitting_story_count[id] = []
            for a in selected_articles:
                if a in v:
                    hitting_story_count[id].append(a)

        self.hitting_story_count = hitting_story_count
        

    def selected_article_info(self, story_id):
        """
            given a list of selected article, analyze those articles
                * list published time, title, bias, entities, stance.
                * number of articles in each story.
                * distributions?
        """
        article_list = self.hitting_story_count[story_id]
        
        df = self.sd.df_metadata[ self.sd.df_metadata.id.isin(article_list)]
        df_ = self.sd.article_df[ self.sd.article_df.id.isin(article_list)]
        df = pd.merge(df,df_,on="id",how="left")
        for idx, row in  df.iterrows():
            a_id = row["id"]

            print("{}\t {:25s} \t{:25s}\t {:70s} \t{}".format(row["published_at"],row["domain"],row["bias"],str((self.selected_story[story_id]["story_entities_stance"][a_id])),row["title"]))


    def find_entity_story(self, entity):
        for story_id,v in self.selected_story.items():
            if entity in v["target_entites"]:
                print("In story {}, selected articles are as following:".format(story_id))
                self.selected_article_info(story_id)
                print()


    def selected_article_entity_balance(self):
        stance_info_all_story = {}
        entity_count_all_story = {}
        stance_info={}
        entity_count = {}
        for story_id,articles in self.hitting_story_count.items():
            for a_id in articles:
                for k,v in self.selected_story[story_id]["story_entities_stance"][a_id].items():
                    if k not in stance_info.keys():
                        stance_info[k] = {"against":0,"neutral-or-unclear":0,"in-favor":0,"None":0}
                        stance_info[k][v]+=1
                        entity_count[k] = 1
                    else:
                        stance_info[k][v]+=1
                        entity_count[k] +=1

        for story_id, story in self.selected_story.items():
            story_entities_stance = story["story_entities_stance"]
            for a_id, value in story_entities_stance.items():
                for k,v in value.items():
                    if k not in stance_info_all_story.keys():
                        stance_info_all_story[k] = {"against":0,"neutral-or-unclear":0,"in-favor":0,"None":0}
                        stance_info_all_story[k][v]+=1
                        entity_count_all_story[k] = 1
                    else:
                        stance_info_all_story[k][v]+=1
                        entity_count_all_story[k] +=1

        covered_entity_info = {}
        for k,v in stance_info_all_story.items():
            if k in stance_info.keys():
                covered_entity_info[k] = stance_info[k]
            else:
                covered_entity_info[k] = {"against":0,"neutral-or-unclear":0,"in-favor":0,"None":0}

        return stance_info,entity_count,stance_info_all_story,entity_count_all_story,covered_entity_info