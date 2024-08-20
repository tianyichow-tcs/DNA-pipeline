from dnc import  *


def initial_solution(K,tau=None,beta=0.3,method="hs"):
    # generate initial solution of from greedy
    if method=="hs":
        selected_articles, cover_progress, num_of_hit, selected_articles_hit,q_hitting_se = alg.greedy_algorithm_hitting_set()
        
    elif method == "hs_cl":
        w = np.array([1/K for i in range(K)])
        selected_articles,q,Gfs,cover_progress = alg.greedy_algorithm_both(tau,w,K)
        
    elif method =="hs_cl_beta":
        w = np.array([1/K for i in range(K)])
        selected_articles,q,Gfs,cover_progress = alg.greedy_algorithm_both_beta(tau,w, K,beta,verbose=False)

    x = [1 if a in selected_articles else 0 for a in alg.df_data.id.to_list()]

    if method =="hs":
        x = np.asarray(x)
        Gfs = [overlap_measure(np.dot(alg.T.T,x)/x.sum(),tau)]

    return np.asarray(x),Gfs[-1],cover_progress[-1]

def neighborhood(x,method="hs_cl"):
    nbrhood = []     
    for i in range(0,len(x)):
        # if method=="hs_cl":             #delete items
        x_temp = np.copy(x)
        if x_temp[i] ==0:
            continue
        else:
            x_temp[i] = 0
            nbrhood.append(x_temp)
        # else:                           #add items
        #     x_temp = np.copy(x)
        #     if x_temp[i] ==0:
        #         continue
        #     else:
        #         x_temp[i] = 0
        #         nbrhood.append(x_temp)
    return nbrhood

def evaluate_bls(x,method ="hs_cl",tau=None, Gf_t = 0.90, Hr_t = 1 ):
    # check hitting coverage constraints
    r = np.dot(alg.H,x)
    count_non_zero = sum(1 for element in r if element != 0)
    Hr = count_non_zero/len(r)

    Gf = overlap_measure(np.dot(alg.T.T,x)/x.sum(),tau)     

    # if Gf< Gf_t or Hr < Hr_t:
    #     raise ValueError
    # else:
    #     return [Gf, Hr], sum(x)
    if method=="hs_cl":  # if consider both
        # Gf_t = 0.95
        Hr_t = 1

        if Gf< Gf_t or Hr < Hr_t:
            raise ValueError
        else:
            return Gf, Hr, sum(x)
        
    if method =="hs":   # if only consider hitting set
        if Hr < Hr_t:
            raise ValueError
        else:
            return Gf,Hr, Gf
        
    if method == "cl":
        if Gf< 0.95 or Hr < Hr_t: # now Hr_t is th current best Hr_t
            raise ValueError
        else:
            return Gf, Hr, Hr
    
def evaluate(x, tau, Gf_t = 0.90,Hr_t = 1):
    # evaluate the current sulotion w.r.t constriant
    # Gf: overlap of two distribution
    # Hr: collection hitting ratio 
    r = np.dot(alg.H,x)
    count_non_zero = sum(1 for element in r if element != 0)
    Hr = count_non_zero/len(r)
    Gf = overlap_measure(np.dot(alg.T.T,x)/x.sum(),tau)  
    # print("Hr: {} , Gf: {}".format(Hr,Gf))
    if Gf< Gf_t or Hr < Hr_t:
        return False
    else:
        return Gf, Hr, sum(x)


def local_search(K,method="hs_cl",tau=None,Gf_t=0.9,Hr_t=1):
    solutionsChecked = 0
    xs = []
    GfHrs = []
    x_curr,Gf_curr,Hr_curr= initial_solution(K,tau=tau,method=method)  #x_curr will hold the current solution 
    print("x_org: {}, Gf_org:{}, Hr_org: {}".format(x_curr.sum(),Gf_curr,Hr_curr))

    x_best = np.copy(x_curr)     #x_best will hold the best solution 
    Gf_curr,Hr_curr, f_curr = evaluate(x_curr,tau,Gf_t,Hr_t)    #f_curr will hold the evaluation of the current soluton 
    f_best = np.copy(f_curr)
    Gf_best = np.copy(Gf_curr)
    Hr_best = np.copy(Hr_curr)
    #begin local search overall logic ----------------
    done = 0

    while done == 0:
        Neighborhood = neighborhood(x_curr,method)   #create a list of all neighbors in the neighborhood of x_curr
        s_candidates = []
        margin = []         # new solution gap with the previos solution
        for s in tqdm(Neighborhood):                #evaluate every member in the neighborhood of x_curr
            solutionsChecked = solutionsChecked + 1
            #Handling infeasible solution
            try:
                Gf_curr,Hr_curr,eval_s = evaluate(s,tau,Gf_t,Hr_t)
            except:
                #print("Infeasible solution handled")
                continue
            
            if eval_s< f_best:
                s_candidates.append(s)
                margin.append(Gf_curr - Gf_best) # sort and find the fist.

        print("Candidates list:",len(s_candidates))
        if len(s_candidates)==0:
            print("condidate == 0")
            done =1
        else:
            #s = random.choice(s_candidates)
            idx =margin.index(max(margin))
            s = s_candidates[idx]
            Gf_curr,Hr_curr, eval_s = evaluate(s,tau,Gf_t,Hr_t)
            x_best = np.copy(s)            #find the best member and keep track of that solution
            f_best = np.copy(eval_s)       #and store its evaluation 
            Gf_best = np.copy(Gf_curr) 
            Hr_best = np.copy(Hr_curr)

        # if f_best == f_curr:               #if there were no improving solutions in the neighborhood
        #     print("current == 0")
        #     done = 1
        # else:
        x_curr = np.copy(x_best)         #else: move to the neighbor solution and continue
        f_curr = np.copy(f_best)         #evalute the current solution
        Hr_curr = np.copy(Hr_best)
        Gf_curr = np.copy(Gf_best)
        xs.append(list(x_best))
        GfHrs.append([float(Gf_best),float(Hr_best)])
        print ("\nTotal number of solutions checked: ", solutionsChecked)
        print ("Best value found so far: ", Gf_best, Hr_best, f_best)     

    print ("\nFinal number of solutions checked: ", solutionsChecked)
    print ("Best value found: ", f_best)
    print ("Hitting rate: ", Hr_best)
    print ("Overlap: ", Gf_best)
    print ("Total number of items selected: ", np.sum(x_best))
    print ("Best solution: ", x_best)
    return xs,GfHrs
    




def additional_info(alg,selected_articles):
    def balance_metric(stance_stats):
        scores = []
        total = 0
        for entity,stance in stance_stats.items():
            a = stance['against']
            n = stance['neutral-or-unclear']
            f = stance['in-favor']
            v = [a,n,f]
            total +=sum(v)
            for i in v:
                scores.append(i)
        score_distr = [i/total for i in scores]
        uniform_distri = [1/len(score_distr) for i in range(len(score_distr))]
        balance = overlap_measure(score_distr,uniform_distri)
        return balance
    
    #domain cover
    n_domains =len(alg.article_df.dropna().domain.unique())
    domain_cov = len(alg.article_df[alg.article_df.id.isin(selected_articles)].dropna().domain.unique())/n_domains

    # story cover

    alg.look_up_selected_articles(selected_articles)
    count_temp = [len(v) for k,v in alg.hitting_story_count.items()]
    count_temp = [1 if i ==0 else 0 for i in count_temp]
    story_cover= 1- sum(count_temp)/len(alg.selected_story)
    # stance balance
    stance_info,entity_count,stance_info_all_story,entity_count_all_story,covered_entity_info = alg.selected_article_entity_balance()
    #balance metric 
    balance = balance_metric(covered_entity_info)
    
    total = 0
    miss = 0
    for k,v in stance_info_all_story.items():
        for stance in ["against","neutral-or-unclear","in-favor"]:
            if stance_info_all_story[k][stance]>0:
                    total+=1
                    if k not in stance_info.keys():
                        miss +=1
                    else:
                         if stance_info[k][stance]==0:
                              miss+=1
                        
    entity_cover = 1-(miss/total)
    balance_org = balance_metric(stance_info_all_story)

    return domain_cov,story_cover,balance,balance_org, entity_cover




def ALG_exp(alg, Ks,data_source,month,target_distributions,if_store=True):
    BL_both_result = {}
    i = 0
    for tau in target_distributions:
        covers = []
        overlaps = []
        for K in Ks:
            w = np.array([1/K for i in range(K)])
            p_distr = tau
            selected_articles, q,Gfs,cover_progress = alg.greedy_algorithm_both(p_distr,w,K,verbose=False)
            covers.append(cover_progress[-1])
            overlaps.append(Gfs[-1])
            print ("i=",i," ","K=",K,", Target distribution ",i," coverage rate: ", cover_progress[-1], " Overlap measure:",Gfs[-1] )
            key = str(i) + "_" + str(K)
            domain_cov,story_cover,balance,balance_org,entity_cover =  additional_info(selected_articles)

            BL_both_result[key] = {"selected_article":list(selected_articles),"q":q.tolist(),"Gfs":Gfs,"Coverage":cover_progress,"balance":balance,"balance_org":balance_org,"story_cover":story_cover,"domain_cover":domain_cov,"entity_cover":entity_cover}
        i+=1
    if if_store:
        with open('results/{}/{}/ALG.json'.format(data_source,month), 'w') as fp:
            json.dump(BL_both_result, fp)


def ALG_LS_exp(alg, Ks,data_source,month,target_distributions,if_store=True,Gf_t=0.9):
    BL_both_result = {}
    i = 0
    for tau in target_distributions:
        K = Ks[-1] #Ks[11]
        
        xs,GfHrs = local_search(K=K,tau=tau,Gf_t=Gf_t,method="hs_cl")
        a_list = alg.df_data.id.to_list()
        x = list(xs[-1])

        Gfs = []
        Coverage = []
        for ol, c in GfHrs:
            Gfs.append(float(ol))
            Coverage.append(float(c))
        xs = [list(i) for i in xs]
        xs = [[float(element) for element in sublist] for sublist in xs]

        selected_articles = [item for item, flag in zip(a_list, x) if flag == 1]
        domain_cov,story_cover,balance,balance_org,entity_cover=  additional_info(selected_articles)
        BL_both_result = {"K":len(selected_articles),"selected_articles":list(selected_articles),"Gfs":Gfs,"Coverage":Coverage,"xs":xs,"balance":balance,"balance_org":balance_org,"story_cover":story_cover,"domain_cover":domain_cov,"entity_cover":entity_cover}

        with open('results/{}/{}/ALG_LS.json'.format(data_source,month), 'w') as fp:
            json.dump(BL_both_result, fp)

def ALG_beta_exp(alg, Ks,data_source,month,target_distributions,beta, if_store=True):
    BL_both_beta_result = {}
    i = 0
    for tau in target_distributions:
        covers = []
        overlaps = []
        for K in Ks:
            w = np.array([1/K for i in range(K)])
            p_distr = tau
            selected_articles, q,Gfs,cover_progress = alg.greedy_algorithm_both_beta(p_distr,w,K,beta,verbose=False)
            covers.append(cover_progress[-1])
            overlaps.append(Gfs[-1])
            print ("i=",i," ","K=",K,", Target distribution ",i," coverage rate: ", cover_progress[-1], " Overlap measure:",Gfs[-1] )
            key = str(i) + "_" + str(K)
            # BL_both_result[key] = [list(selected_articles),q.tolist(),Gfs,cover_progress]
            domain_cov,story_cover,balance,balance_org,entity_cover =  additional_info(selected_articles)

            BL_both_beta_result[key] = {"selected_article":list(selected_articles),"q":q.tolist(),"Gfs":Gfs,"Coverage":cover_progress,"balance":balance,"balance_org":balance_org,"story_cover":story_cover,"domain_cover":domain_cov,"entity_cover":entity_cover}
        i+=1
    if if_store:
        with open('results/{}/{}/ALG_beta.json'.format(data_source,month), 'w') as fp:
            json.dump(BL_both_beta_result, fp)


def BL1_exp(data_source,month,target_distributions,if_store=True):
    BL1_result = {}
    i=0
    for tau in target_distributions[:1]:
        selected_articles, cover_progress, num_of_hit, selected_articles_hit,q_hitting_set,Gfs,cover_progress = alg.greedy_algorithm_hitting_set(p=tau,verbose=False)
        print ("i=",i," ","K=",len(selected_articles),", Target distribution ",i," coverage rate: ", cover_progress[-1], " Overlap measure:",Gfs[-1] )
        key = str(i)
        domain_cov,story_cover,balance,balance_org,entity_cover=  additional_info(selected_articles)
        BL1_result[key]={"selected_article":list(selected_articles),"Gfs":Gfs,"Coverage":cover_progress,"balance":balance,"balance_org":balance_org,"story_cover":story_cover,"domain_cover":domain_cov,"entity_cover":entity_cover}
        i+=1
    if if_store:
        with open('results/{}/{}/BL1.json'.format(data_source,month), 'w') as fp:
            json.dump(BL1_result, fp)


    

def BL2_exp(alg, Ks,data_source, month, target_distributions,if_store=True):
    BL2_result = {}
    i=0
    for tau in target_distributions[:1]:
        for K in Ks:
            p_distr = tau
            w = np.array([1/K for i in range(K)])
            selected_articles,q,Gfs,cover_progress = alg.greedy_algorithm_calibration(p_distr,w,K)        
            print ("i=",i," ","K=",K,", Target distribution ",i," coverage rate: ", cover_progress[-1], " Overlap measure:",Gfs[-1] )
            key = str(i) + "_" + str(K)
            # BL2_result[key]=[selected_articles_calibration,q.tolist(),Gfs,cover_progress]
            domain_cov,story_cover,balance,balance_org,entity_cover =  additional_info(selected_articles)
            BL2_result[key]={"selected_article":list(selected_articles),"q":q.tolist(),"Gfs":Gfs,"Coverage":cover_progress,"balance":balance,"balance_org":balance_org,"story_cover":story_cover,"domain_cover":domain_cov,"entity_cover":entity_cover}
        i+=1
    if if_store:
        with open('results/{}/{}/BL2.json'.format(data_source,month), 'w') as fp:
            json.dump(BL2_result, fp)


def BL3_exp(alg, Ks,target_distributions,data_source, month,beta,if_store=True):
    BL3_result = {}
    i=0
    news_set = alg.total_article.copy()
    S_vectors =  [alg.dict_S[i] for i in news_set]
    B_vectors = [alg.dict_B[i] for i in news_set]
    articles_order = list(alg.total_article.copy())
    S_similarity = cosine_similarity(S_vectors,S_vectors)
    B_similarity = cosine_similarity(B_vectors,B_vectors)

    for tau in target_distributions:
        for K in Ks:
            p_distr = tau
            w = np.array([1/K for i in range(K)])
            selected_articles,q,Gfs,cover_progress = alg.MMR(p_distr,w,K,beta,articles_order,S_similarity,B_similarity)        
            print ("i=",i," ","K=",K,", Target distribution ",i," coverage rate: ", cover_progress[-1], " Overlap measure:",Gfs[-1] )
            key = str(i) + "_" + str(K)
            domain_cov,story_cover,balance,balance_org,entity_cover =  additional_info(selected_articles)
            BL3_result[key]={"selected_article":selected_articles,"q":q.tolist(),"Gfs":Gfs,"Coverage":cover_progress,"balance":balance,"balance_org":balance_org,"story_cover":story_cover,"domain_cover":domain_cov,"entity_cover":entity_cover}
        i+=1
    if if_store:
        with open('results/{}/{}/BL3.json'.format(data_source,month), 'w') as fp:
            json.dump(BL3_result, fp)


def BL3_beta_exp(alg, K,target_distributions,data_source, month,betas,if_store=True):
    BL3_beta_result = {}
    i=0
    news_set = alg.total_article.copy()
    S_vectors =  [alg.dict_S[i] for i in news_set]
    B_vectors = [alg.dict_B[i] for i in news_set]
    articles_order = list(alg.total_article.copy())
    S_similarity = cosine_similarity(S_vectors,S_vectors)
    B_similarity = cosine_similarity(B_vectors,B_vectors)
    for tau in target_distributions:
        for beta in betas:
            print("beta: ", beta)
            p_distr = tau
            # selected_articles,cover_progress,num_of_hit,selected_articles_hit=greedy_hitting_set(hittingset_collection.copy(),uncovered_sets,total_article)
            w = np.array([1/K for i in range(K)])
            selected_articles,q,Gfs,cover_progress = alg.MMR(p_distr,w,K,beta,articles_order,S_similarity,B_similarity)        
            print ("i=",i," ","beta=",beta,", Target distribution ",i," coverage rate: ", cover_progress[-1], " Overlap measure:",Gfs[-1] )
            key = str(i) + "_" + str(beta)
            domain_cov,story_cover,balance,balance_org,entity_cover=  additional_info(selected_articles)
            BL3_beta_result[key]={"selected_article":selected_articles,"q":q.tolist(),"Gfs":Gfs,"Coverage":cover_progress,"balance":balance,"balance_org":balance_org,"story_cover":story_cover,"domain_cover":domain_cov,"entity_cover":entity_cover}
        i+=1
    if if_store:
        with open('results/{}/{}/BL3_beta.json'.format(data_source,month), 'w') as fp:
            json.dump(BL3_beta_result, fp)