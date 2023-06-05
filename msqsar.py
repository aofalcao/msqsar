
#Local MSQSAR - Localized Metric Space QSAR

#Copyright © 2022-2023 Andre O. Falcao - DI/FCUL

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#and associated documentation files (the "Software"), to deal in the Software without restriction, 
#including without limitation the rights to use, copy, modify, merge, publish, distribute, 
#publicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or 
#substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
#INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
#PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
#FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
#OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
#DEALINGS IN THE SOFTWARE.

 
 
import numpy as np
from math import sqrt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics  import precision_score, recall_score, confusion_matrix
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.svm import SVR, SVC
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import AllChem



def read_smiles(smiles_list):
    #list of smiles
    mols={}
    i=1
    for smiles in smiles_list:
        act="NA"
        signal="="
        molid="M_%d" % i
        mols[molid]=(smiles, act, signal)
        i+=1
    return mols



def read_molecules(fname):
    #This must be the structure or it will output an error: (mol_id, activity, SMILES), separated by tabs
    #the activity need not be a float. If char, then it will be a classification model
    fil=open(fname, "rt")
    lins=fil.readlines()
    fil.close()
    
    mols={}
    regression=False
    classification=False
    bootstrap_r=False
    for lin in lins:
        molid, act, smiles = lin.strip().split("\t")
        try:
            act=float(act)
            signal=""
            regression=True
        except:
            if act[0] in [">", "<"]: 
                act=float(act[1:])
                signal=act[0]
                bootstrap_r=True
            else:
                classification=True
                signal=""
        mols[molid]=(smiles, act, signal)
    if classification ==True: 
        model_type="Classification"
    else: 
        if bootstrap_r==True:
            model_type="Bootstrap"
        else:
            model_type="Regression"

    return mols, model_type

def calc_fingerprints(mols, fp_R, fp_NB):
    #import sys
    #import warnings
    #warnings.simplefilter("ignore")
    mol_fps={}
    i=0
    #parse_log=open("molparse.log", "at")
    #sys.stdout = parse_log
    for mol in mols:
        smiles, act, signal=mols[mol]
        #print(i, "************>", smiles, act, signal)
        mol_ok=True
        try:
            m = Chem.MolFromSmiles(smiles)
            fp=AllChem.GetMorganFingerprintAsBitVect(m, radius=fp_R, nBits = fp_NB)
        except:
            #print(smiles)
            m = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
            if m is None: 
                mol_ok=False
            else:
                m.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(m,Chem.SANITIZE_SYMMRINGS|Chem.SANITIZE_SETCONJUGATION|Chem.SANITIZE_SETHYBRIDIZATION)
                fp=AllChem.GetMorganFingerprintAsBitVect(m, radius=fp_R, nBits = fp_NB)
        if mol_ok==True: mol_fps[mol] = frozenset(fp.GetOnBits())
        i+=1
    #sys.stdout = sys.__stdout__
    #parse_log.close()
    return mol_fps

def calc_dist_matrix(mol_fps):
    #we must return the mol_ids as well
    mol_ids=list(mol_fps.keys())
    dst_matrix=[]
    nmols=len(mol_ids)
    for i in range(nmols):
        dst_matrix.append([0]*nmols)
    
    for i in range(nmols-1):
        A = len(mol_fps[mol_ids[i]])
        for j in range(i+1,nmols):
            B = len(mol_fps[mol_ids[j]])
            C = len(mol_fps[mol_ids[i]] & mol_fps[mol_ids[j]])
            jacc = C / (A + B - C)
            dst_matrix[i][j] = 1 - jacc
            dst_matrix[j][i] = 1 - jacc
            
    dst_matrix=np.array(dst_matrix)
    return mol_ids, dst_matrix

def calc_test_dist_matrix(train_fps, mol_ids_train, test_fps):
    mol_ids_test=list(test_fps.keys())
    dst_matrix=[]
    nmols_train=len(mol_ids_train)
    nmols_test=len(mol_ids_test)
    
    row = 0
    for mol_test in mol_ids_test:
        dst_matrix.append([0]*nmols_train)

        A = len(test_fps[mol_test])
        col=0
        for mol_train in mol_ids_train:
            B = len(train_fps[mol_train])
            C = len(test_fps[mol_test] & train_fps[mol_train])
            jacc = C / (A + B - C)
            dst_matrix[row][col] = 1 - jacc
            col+=1
        row+=1

    dst_matrix=np.array(dst_matrix)
    return mol_ids_test, dst_matrix

def get_neighbours(mol_ids_test, mol_ids_train, dist_matrix, max_dst, max_siz, min_siz):
    """
    for each molecule in mol_ids_test select the closest neighbours in mol_ids_train
    the outcasts are molecules for which we do not find enough close elements and are difficult to predict
    """
    nmols = np.shape(dist_matrix)[0]
    ncols = np.shape(dist_matrix)[1]
    mol_neighs = {}
    outcasts=[]
    
    for i in range(nmols):
        mol_id=mol_ids_test[i]
        neighs=[]
        dsts=dist_matrix[i, :]
        for j in range(ncols):
            neighs.append((dist_matrix[i, j], j))  #save distance and mol_id
        neighs.sort()
        #this check will verify if in the minimum required elements for modeling
        #the distance is smaller than the maximum allowed distance
        #if that is not the case, then this element is an outcast 
        #print(i, "--->", neighs[:10])
        if neighs[min_siz][0] <= max_dst:
            #print("SIZE:", len(neighs))
            for k in range(min(len(neighs),max_siz+1)):
                n = neighs[k][1] #in this place it is the position of the the mol_id of the neighbouring molecule
                # I will just use the molecules different from itself!
                # so the mol_neighs will have a list of the columns ids that have close molecules
                if mol_ids_test[i]!= mol_ids_train[n] and neighs[min_siz][0]<=max_dst: 
                    mol_neighs.setdefault(mol_id, []).append(n)
        else:
            outcasts.append(mol_id)
            mol_neighs[mol_id]=[]
    
    return mol_neighs, outcasts


def make_X_train(neighs, dist_matrix):
    rc = np.array(neighs, dtype=np.intp)
    X_train = dist_matrix[np.ix_(rc, rc)]        
    return X_train

def make_X_test(i, neighs, dist_matrix):
    cols = np.array(neighs, dtype=np.intp)
    row=np.array([i], dtype=np.intp)
    X_test = dist_matrix[np.ix_(row, cols)]
    return X_test

def make_Y_train(neighs, mol_ids, mols):
    acts=[]
    for n in neighs:
        mol_id=mol_ids[n]
        acts.append(mols[mol_id][1])       
    return np.array(acts)

def get_actives(mols, mol_fps, model_type, thr_actives):
    actives={}
    for mol in mol_fps:
        if model_type=="Regression":
            # store the Fingerprint and the precomputed size (important!)
            if mols[mol][1]> thr_actives: actives[mol]=(mol_fps[mol], len(mol_fps[mol]))
        else:
            if mols[mol][1]=="A": actives[mol]=(mol_fps[mol], len(mol_fps[mol]))
    return actives

def divide_file(scr_fname, nprocs):
    zfile=open(scr_fname, "rt")
    fils=[]
    for i in range(nprocs): fils.append(open("%s.%d" %(scr_fname,i), "wt"))

    lin=zfile.readline()
    i=0
    while lin!="":
        fils[i % nprocs].write(lin)
        lin=zfile.readline()
        i+=1
    zfile.close()
    map(lambda f: f.close(), fils)

def proc_prescreen(scr_fname, radius, nbits, actives, max_act_dist, min_actives, silent, nproc):
    zfile=open(scr_fname, "rt")
    lin=zfile.readline()
    i=1
    cands={}
    cands_fps={}
    cands_smiles={}
    while lin!="": #######################################!!!!!!
        smi, zid = lin.strip().split()
        #print(smi)
        zid=zid.strip()
        m = Chem.MolFromSmiles(smi)
        FP=AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits = nbits)
        zfp=frozenset(FP.GetOnBits())
        A=len(zfp)
        for mid in actives:
            B = actives[mid][1]
            C = len(actives[mid][0] &  zfp)
            jacc= C / ( A + B - C)
            dst=1-jacc
            if dst<max_act_dist:
                #get the number of actives in the vicinity of the candidate
                cands.setdefault(zid, []).append(mid)
        if zid in cands:
            if len(cands[zid])>= min_actives: 
                cands_fps[zid]=zfp
                cands_smiles[zid]=smi
                #print(zid, cands[zid])
                
            
        if i % 10000 ==0 and not silent: print("\t(%d) Mols read: %9d - Candidates found: %5d (%7.4f)"  %(nproc, i, len(cands_fps), len(cands_fps)/i)  )
        i+=1
        lin=zfile.readline()

    zfile.close()
    return cands_fps, cands_smiles
    


def pprescreen(scr_fname, mols, mol_fps, radius, nbits, model_type, max_act_dist, thr_actives, min_actives=2, silent=False, nprocs=4):
    """
This is the parallel prescreener and it is going to be hard. The idea is:
1. split the screen file into several processes (4)
2. Run each specific 
    """

    #create the set of actives
    actives = get_actives(mols, mol_fps, model_type, thr_actives)
    #if silent==False: print("N. of actives considered:", len(actives))

    #divide the file in nproc bits
    divide_file(scr_fname, nprocs)

    #start the pre-screening process
    sfnames=["%s.%d" % (scr_fname,i) for i in range(nprocs)]
    pool = mp.Pool(processes = nprocs)
    res0 = [ pool.apply_async(proc_prescreen, args=(sfnames[i], radius, nbits, actives, max_act_dist, min_actives, silent, i+1)) for i in range(nprocs) ]
    res  = [ p.get() for p in res0]
    #concatenate all results
    cands_fps={}
    cands_smiles={}
    for cand_fps, cand_smiles in res:
        cands_fps={**cands_fps, **cand_fps}
        cands_smiles={**cands_smiles, **cand_smiles}

    return cands_fps, actives, cands_smiles


def prescreen(scr_fname, mols, mol_fps, radius, nbits, model_type, max_act_dist, thr_actives, min_actives=2, silent=False):

    #create the set of actives
    actives = get_actives(mols, mol_fps, model_type, thr_actives)
    #if silent==False: print("N. of actives considered:", len(actives))
    
    #start the pre-screening process
    cands_fps, cands_smiles = proc_prescreen(scr_fname, radius, nbits, actives, max_act_dist, min_actives, silent, 1)
    return cands_fps, actives, cands_smiles


def prescreen_one(smiles, mols, mol_fps, radius, nbits, model_type, max_act_dist, thr_actives, min_actives=2,
                  silent=False):
    """This utility function will check if it is possible to find a positive in the 'hood
       This will only be ran when checking only ONE molecule against a target   
    """
 
    actives = get_actives(mols, mol_fps, model_type, thr_actives)
    
    
    #start the pre-screening process
    zid=0
    m = Chem.MolFromSmiles(smiles)
    FP=AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits = nbits)
    zfp=frozenset(FP.GetOnBits())
    A=len(zfp)
    cands={}
    cands_fps={}
    cands_smiles={}
    
    #check all actives
    for mid in actives:
        B = actives[mid][1]
        C = len(actives[mid][0] &  zfp)
        jacc= C / ( A + B - C)
        dst=1-jacc
        if dst<max_act_dist:
            #get the number of actives in the vicinity of the candidate
            cands.setdefault(zid, []).append(mid)
    #check if, for that molecule, there is a sufficient number of actives for modeling
    if zid in cands:
        if len(cands[zid])>= min_actives: 
            cands_fps[zid]=zfp
            cands_smiles[zid]=smiles
    return cands_fps, actives, cands_smiles
        
def assign_activity(model_type, test_train_neighs, mols_train,
                    mol_ids_train,  mol_ids_test, dist_matrix_train, dist_matrix_test,
                    algo="RF", ntrees=10):

    if model_type=="Regression":
        if algo=="RF": 
            model= RandomForestRegressor(n_estimators=ntrees)
        else: 
            model= SVR(gamma='auto')
    else: 
        if algo=="RF": 
            model = RandomForestClassifier(n_estimators=ntrees)
        else:
            model = SVC(gamma='auto')
    preds={}
    nmols=len(mol_ids_test)
    for i in range(nmols):
        neighs=test_train_neighs[mol_ids_test[i]]
        if len(neighs)>0:
            Y_train=make_Y_train(neighs, mol_ids_train, mols_train)

            if len(set(Y_train))>1:
                X_train=make_X_train(neighs, dist_matrix_train)
                X_test=make_X_test(i, neighs, dist_matrix_test)
                model.fit(X_train, Y_train)
                pred = model.predict(X_test)
            else:
                #if all values are the same, there is no learning possible and the prediction is the 
                #one and only value
                if model_type=="Regression": 
                    pred=[Y_train[0,]]
                else:
                    pred = Y_train[0,]
            preds[mol_ids_test[i]]=pred[0]
        else:
            preds[mol_ids_test[i]]  = "NA"
    return preds


def check_molecules(model_type, mols, mol_ids, mol_neighs, outcasts, dist_matrix, algo="RF", ntrees=10):
    #this is the heart of the model
    #1. for each model not in the outcasts, make a X_train and Y_train
    #2. fit a model
    #3. predict the activity
    
    if model_type=="Regression":
        if algo=="RF": 
            model= RandomForestRegressor(n_estimators=ntrees)
        else: 
            model= SVR(gamma='auto')
    else: 
        if algo=="RF": 
            model = RandomForestClassifier(n_estimators=ntrees)
        else:
            model = SVC(gamma='auto')
    truths={}
    preds={}
    nmols=len(mol_ids)
    for i in range(nmols):
        mol_id = mol_ids[i]
        truth=mols[mol_id][1]
        if mol_ids[i] not in outcasts:
            neighs=mol_neighs[mol_id]
            Y_train=make_Y_train(neighs, mol_ids, mols)
            if len(set(Y_train))>1:
                X_train=make_X_train(neighs, dist_matrix)
                X_test=make_X_test(i, neighs, dist_matrix)
                model.fit(X_train, Y_train)
                pred = model.predict(X_test)
            else:
                #if all values are the same, there is no learning possible and the prediction is the 
                #one and only value
                if model_type=="Regression": 
                    pred=[Y_train[0,]]
                else:
                    pred = Y_train[0,]
            truths[mol_id] = truth
            preds[mol_id] = pred[0]
            #print(mol_ids[i], truth, pred)
        else:
            truths[mol_id] = truth
            preds[mol_id]  = "NA"    
    # print(classification_report(truths, preds))
    return truths, preds


def dataset_preparation(fname, fp_R, fp_NB, max_dst, max_siz, min_siz=5, algo="RF", ntrees=10):
    mols, model_type =     read_molecules(fname)
    mol_fps =              calc_fingerprints(mols, fp_R, fp_NB)
    mol_ids, dist_matrix = calc_dist_matrix(mol_fps)
    mol_neighs, outcasts = get_neighbours(mol_ids, mol_ids, dist_matrix, max_dst, max_siz, min_siz)
    return model_type, mols, mol_ids, mol_neighs, outcasts, dist_matrix


def validate_model(fname_train, fname_test, fp_R, fp_NB, max_dst, max_siz, min_siz=5, algo="RF", ntrees=10, silent=False):
    mols_train, model_type      =     read_molecules(fname_train)
    mols_test,  model_type_test =     read_molecules(fname_test)
    if silent==False:
        print("Model Type:", model_type)
        if model_type!=model_type_test:
            print("Warning: %s will be compared to %s test data." % (model_type, model_type_test))
            
    fps_train =              calc_fingerprints(mols_train, fp_R, fp_NB)
    fps_test  =              calc_fingerprints(mols_test, fp_R, fp_NB)
    mol_ids_train, dist_matrix_train = calc_dist_matrix(fps_train)
    mol_ids_test,  dist_matrix_test  = calc_test_dist_matrix(fps_train, mol_ids_train, fps_test)

    test_train_neighs, test_outcasts = get_neighbours(mol_ids_test, mol_ids_train, dist_matrix_test, max_dst, max_siz, min_siz)
    preds=assign_activity(model_type, test_train_neighs, mols_train,
                          mol_ids_train,  mol_ids_test, dist_matrix_train, dist_matrix_test, algo=algo, ntrees=ntrees)

    truths={}
    for mid in preds: truths[mid]=mols_test[mid][1]
        
    return {"preds":preds, "truth":truths, "modeltype": model_type}




def evaluate_model(fname, fp_R, fp_NB, max_dst, max_siz, min_siz=5, algo="RF", ntrees=10, silent=False):
    if silent==False: print("1. Reading molecules...", end=" ")
    mols, model_type =     read_molecules(fname)
    if silent==False: print("Model Type:", model_type)
    if silent==False: print("2. Determining structure...")
    mol_fps =              calc_fingerprints(mols, fp_R, fp_NB)
    if silent==False: print("3. Computing chemical space...")
    mol_ids, dist_matrix = calc_dist_matrix(mol_fps)
    #mol_ids will appaer repeated as this case is a self similarity matrix
    #it might not be so. In this case its the same ids on rows and columns 
    if silent==False: print("4. Finding modeling matches...")
    mol_neighs, outcasts = get_neighbours(mol_ids, mol_ids, dist_matrix, max_dst, max_siz, min_siz)
    if silent==False: print("5. Fit local models...", end=" ")    
    tr, pr = check_molecules(model_type, mols, mol_ids, mol_neighs, outcasts, dist_matrix, algo=algo, ntrees=ntrees)
    if silent==False: print("Done!")    
    return {"preds":pr, "truth":tr, "modeltype": model_type}


    
def screener(dat_fname, scr_fname, fp_R, fp_NB, max_dst, max_siz, max_act_dist, thr_actives=4.0, min_siz=4, 
             min_actives=2, algo="RF", ntrees=10, silent=False, parallel=False, nprocs=4):
    """this is the most important function where we screen a full database against a possible target
    """
    mols_train, model_type =     read_molecules(dat_fname)
    if silent==False: print("1. Model Type:", model_type)

    if model_type != "Classification":
        print("Sorry. Currently only classification is supported for screening")
        exit()
    
    if silent==False: print("2. Determining structure", len(mols_train))
    fps_train =              calc_fingerprints(mols_train, fp_R, fp_NB)
    if silent==False: print("3. Computing chemical space")
    mol_ids_train, dist_matrix_train = calc_dist_matrix(fps_train)
    # now do a pre_screen and select only the molecules that are close to actives. 
    # This means that even if the data is regression based, then we must select only the 
    # molecules that are close to the acives within a minimum distane
    if silent==False: print("4. Starting pre-screen")
    if parallel==True:
        if silent==False: print("\tThis time it is parallel!")
        fps_candidates, actives, smi_cands = pprescreen(scr_fname, mols_train, fps_train, fp_R, fp_NB, model_type,
                                                        max_act_dist, thr_actives, min_actives=min_actives, silent=silent,
                                                        nprocs=nprocs)
    else:
        fps_candidates, actives, smi_cands = prescreen(scr_fname, mols_train, fps_train, fp_R, fp_NB, model_type,
                                                       max_act_dist, thr_actives, min_actives=min_actives, silent=silent)

    if silent==False: print("5. Pre-screen finished. Candidates found:", len(fps_candidates))
    
    #the process must be: 
    # 1. for each candidate compute the distance to the test set
    # 2. check the closest neighbours
    # 3. fit model and assign activity
    # repeat 1-3 for each little screened molecule
    if silent==False: print("6. Start the candidate screening...")
    pr={}
    smis={}
    for mid_test in fps_candidates:
        #let's use the EXACT same structures, but with only one molecule!
        fps_candidate={mid_test: fps_candidates[mid_test]}
        mol_ids_test, dist_matrix_test = calc_test_dist_matrix(fps_train, mol_ids_train, fps_candidate)
        test_train_neighs, test_outcasts = get_neighbours([mid_test], mol_ids_train, dist_matrix_test, max_dst, max_siz, min_siz)
        pred=assign_activity(model_type, test_train_neighs, mols_train,mol_ids_train,  [mid_test],
                             dist_matrix_train, dist_matrix_test, algo=algo, ntrees=ntrees )

        if len(pred)>0:
            for mid in pred: #there should be only one value in pred!!!
                pr[mid]=pred[mid]
                smis[mid]=smi_cands[mid]
    return {"preds":pr, "modeltype": model_type, "smiles": smis}

def filter_closest(smiles, fp_R, fp_NB, max_siz, mols_train, fps_train):
    # this will get ONLY the closet molecules to the search molecule. 
    # This should be loads faster for screening only one molecule
    m = Chem.MolFromSmiles(smiles)
    FP=AllChem.GetMorganFingerprintAsBitVect(m, radius=fp_R, nBits = fp_NB)
    zfp=frozenset(FP.GetOnBits())
    A=len(zfp)
    cands={}
    cands_fps={}
    cands_smiles={}
    #check all actives
    dsts=[]
    for mid in mols_train:
        B = len(fps_train[mid])
        C = len(fps_train[mid] & zfp)
        jacc= C / ( A + B - C)
        dsts.append( ((1-jacc), mid) )
    #get the smallest distances to the molecule
    dsts.sort()
    filtered_mols={}
    filtered_fps={}
    for d,mid in dsts[:max_siz]:
        filtered_mols[mid] = mols_train[mid]
        filtered_fps[mid]  = fps_train[mid]
    return filtered_mols, filtered_fps

def read_fps(dat_fname, fp_r, fp_NB):
    import pickle
    #first get the model
    rev_mdl= {(3, 1024):1, (3, 2048):2, (4, 1024): 3, (4, 2048):4}
    mdl = rev_mdl[ (fp_r, fp_NB) ]
    #now disassemble the dat_fname to get the path, chembl_id and gene
    stuff = dat_fname.split("_")
    gene=stuff[-1].replace(".sar","")
    path_chid="_".join(stuff[:-1])
    pelems=path_chid.split("/")
    ch_id=pelems[-1]
    path="/".join(pelems[:-1]) +"/fps/"    
    fname="%s%s_%s.fp%d" % (path, ch_id, gene, mdl)
    #print("---> Reading:", fname)
    mols_fp=pickle.load(open(fname, "rb"))
    return mols_fp
    
def screen_one_mol(dat_fname, smiles, fp_R, fp_NB, max_dst, max_siz, max_act_dist, thr_actives=4.0, min_siz=4, 
             min_actives=1, algo="RF", ntrees=10, silent=False, parallel=False, nprocs=4):
    """this is the most important function where we screen only one molecule against a possible target
       this is the entry point for csact
    """ 
    
    #import time
    
    #t0 = time.time()
    mols_train, model_type = read_molecules(dat_fname)    
    #fps_train =              calc_fingerprints(mols_train, fp_R, fp_NB)
    fps_train =               read_fps(dat_fname, fp_R, fp_NB)

    # this. first we need to get just the closest molecules
    mols_filter, fps_filter=filter_closest(smiles, fp_R, fp_NB, max_siz, mols_train, fps_train)
    mol_ids_train, dist_matrix_train = calc_dist_matrix(fps_filter)
    # now do a pre_screen and select only the molecules that are close to actives. 
    # This means that even if the data is regression based, then we must select only the 
    # molecules that are close to the acives within a minimum distane

    #return cands_fps, actives, cands_smiles

    fps_candidates, actives, smi_cands = prescreen_one(smiles, mols_filter, fps_filter, fp_R, fp_NB, model_type,
                                                       max_act_dist, thr_actives, min_actives=min_actives,
                                                       silent=silent)
    #print("TimeIt: 1: %7.4f 2: %7.4f 3: %7.4f 4: %7.4f 5: %7.4f" % (t1-t0, t2-t1, t3-t2, t4-t3, t5-t4))
    #print("#####>", len(fps_candidates), len(actives), len(smi_cands))
    #the process must be: 
    # 1. for each candidate compute the distance to the test set
    # 2. check the closest neighbours
    # 3. fit model and assign activity
    # repeat 1-3 for each little screened molecule

    pr={}
    smis={}
    if len(fps_candidates)==0:
        #this is a situation when there were not enough data for modeling, thus we will output an NA
        #and get out 
        return {"preds":{0: 'NA'}, "modeltype": model_type, "smiles": {0: smiles}}
    
    for mid_test in fps_candidates:
        #let's use the EXACT same structures, but with only one molecule!
        fps_candidate={mid_test: fps_candidates[mid_test]}
        mol_ids_test, dist_matrix_test = calc_test_dist_matrix(fps_filter, mol_ids_train, fps_candidate)
        test_train_neighs, test_outcasts = get_neighbours([mid_test], mol_ids_train, dist_matrix_test, max_dst,
                                                          max_siz, min_siz)
        pred=assign_activity(model_type, test_train_neighs, mols_filter,mol_ids_train,  [mid_test],
                             dist_matrix_train, dist_matrix_test, algo=algo, ntrees=ntrees )

        if len(pred)>0:
            for mid in pred: #there should be only one value in pred!!!
                pr[mid]=pred[mid]
                smis[mid]=smi_cands[mid]
        else:
            pr[mid]="NA" #if no prediction, then we cannot say anything
            smis[mid] = smi_cands[mid]
            
    return {"preds":pr, "modeltype": model_type, "smiles": smis}


def get_binary_stats(tr_d, pr_d):
    mol_ids = list(tr_d.keys())
    tr = []
    pr = []
    N=len(mol_ids)
    for mid in mol_ids:
        if pr_d[mid]!="NA":
            pr.append(pr_d[mid])
            tr.append(tr_d[mid])
    N0=len(pr)
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(tr)):
        if tr[i] == "A" and pr[i] == "A": TP+=1.0
        if tr[i] == "A" and pr[i] != "A": FN+=1.0
        if tr[i] != "A" and pr[i] == "A": FP+=1.0
        if tr[i] != "A" and pr[i] != "A": TN+=1.0
    if TP + FP >0:
        precision = TP / (TP + FP)
    else:
        precision=0
    if TP + FN>0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    if precision*recall >0:
        F1_score = 2/(1/precision+1/recall)
    else:
        F1_score = 0
        
    if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)>0:
        MCC=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    else:
        MCC=0
    #mcc=matthews_corrcoef(tr,pr)
    return {"TP":TP, "TN": TN, "FP": FP, "FN": FN, "precision": precision,
            "recall": recall, "F1": F1_score, "mcc": MCC, "N": N, "N0": N0}


def get_regression_stats(tr_d, pr_d):
    mol_ids = list(tr_d.keys())
    tr = []
    pr = []
    N=len(mol_ids)
    for mid in mol_ids:
        if pr_d[mid]!="NA":
            pr.append(pr_d[mid])
            tr.append(tr_d[mid])
    N0=len(pr)
    corr=np.corrcoef(tr,pr)[0,1] #pearson CC
    pve= explained_variance_score(tr, pr)  
    mse= mean_squared_error(tr, pr)
    rmse=sqrt(mse)
    return {"pve": pve, "rmse": rmse, "pearson": corr, "N": N, "N0": N0}

def get_stats(results):
    """
    An High level function for providing both binary classification or Regression results
    """
    preds=results["preds"]
    truths=results["truth"]
    model_type=results["modeltype"]
    
    if model_type=="Regression":
        return get_regression_stats(truths, preds)
    else: 
        return get_binary_stats(truths, preds)


def write_results(results, out_file, detail, stats):
    import sys
    pr=results["preds"]
    tr=results["truth"]
    model_type=results["modeltype"]
    if out_file is None:
        fout=sys.stdout
    else:
        #perhaps check if exists??
        fout=open(out_file, "wt")

    if stats==True:
        if model_type=="Regression":
            stats=get_regression_stats(tr,pr)
            fout.write("RMSE: %7.4f\nPearson: %7.4f\nPVE: %7.4f\n%%Predicted: %6.3f\n" %(stats["rmse"], stats["pearson"],
                                                                               stats["pve"], stats["N0"]/stats["N"]))    
        elif model_type=="Classification":
            stats=get_binary_stats(tr,pr)
            fout.write("MCC: %7.4f\nF1: %7.4f\nPrecision: %7.4f\nRecall: %7.4f\n%%Predicted: %6.3f\n" %(stats["mcc"], stats["F1"],
                                                                                             stats["precision"],stats["recall"],
                                                                                             stats["N0"]/stats["N"]))
    if detail==True:
        if stats==True: fout.write("Detailed results:\n")
        for mid in tr:
            truth= tr[mid]
            pred = "NA"
            if pr[mid] is not None: pred = pr[mid]
            if not isinstance(pred, str): pred = "%7.4f" % (pred)
            if not isinstance(truth, str): truth = "%7.4f" % (truth)
                
            fout.write("%15s\t%s\t%s\n" %(mid, truth, pred))
            
    if out_file is not None: fout.close()

def write_screen_results(results, out_file):
    pr=results["preds"]
    smi=results["smiles"]
    model_type=results["modeltype"]
    import sys
    if out_file is None:
        fout=sys.stdout
    else:
        #perhaps check if exists??
        fout=open(out_file, "wt")
    for mid in pr:
        if model_type=="Classification":
            fout.write("%-15s\t%s\t%s\n" %(mid, pr[mid], smi[mid]))
        else:
            fout.write("%-15s\t%7.4f\t%s\n" %(mid, pr[mid], smi[mid]))

    if out_file is not None: fout.close()


def print_help():
    s= """
MS-QSAR - (C) Andre Falcao DI/FCUL version 0.2.20230605
Usage: This is a python tool for building Metric space QSAR models.
       This tool requires an enviroment where RDkit and scikit-learn is installed
       To run type python msqsar.py [method] -in [input .sar file] [options]

method can be:
    eval - evaluates the quality of a given data set for making inference
    test - checks predictions made against a data set
    infer - makes predictions against a data set - no stats are provided
    screen - screens a large database, identifying the most likely candidates (classification only)

Control parameters:
    -in file_name - the data set used for model building (required) (.sar format)
    -test file_name - data set required for method=test (.sar format)
    -scr filename - file with data for screening. (.smi format)
    -out filename - file where the output is stored. if ommited, redirects to stdout
    -fpR N - fingerprint radius (default: 2)
    -fpNB N - fingerprint number of bits (default: 1024)
    -max_dist f - maximum distance for prediction (default: 0.9)
    -min_sims N - minimum number of instances for modeling (default: 5)
    -max_sims - maximum number of instances for modeling (default: 20)
    -algo (SVM | RF)- machine learning algorithm  (default: RF)
    -ntrees - Number of trees in random forest (RF (default: 20)
    -nprocs - Number of processes if -parallel option enabled

Output Control Options:
    -silent - no intermediate output at all
    -detail - in the final output show the individual predictions
    -nostats - do not show end model statistics for methods eval and test
    -parallel - run the process in parallel (currently only for screening)
"""
    print(s.strip())
    

if __name__=="__main__":
    import sys

    #defaults
    fpR = 2
    fpNB = 2048
    max_dist = 0.5
    min_sims = 3
    max_sims = 15
    output_file = None
    input_file = None
    screen_file = None
    test_file = None
    silent = False
    detail  = False
    stats = True
    algo="SVM"
    ntrees=20
    parallel=False
    nprocs=4

    bin_args=["-fpR","-fpNB", "-max_dist","-min_sims", "-max_sims","-in", "-out", "-test", "-scr", "-algo", "-ntrees", "-nprocs"]
    all_args=bin_args[:]+["-silent", "-detail","-nostats", "-help", "-parallel"]

    #inp="sartool eval -fpR 3 -fpNB 1024 -max_dist 0.9   -min_sims 4   -max_sims  10  -in cftr.sar"
     
    #n_inp = " ".join(inp.split())
    #inp_fs = n_inp.split(" ")
    inp_fs=sys.argv
    method = inp_fs[1]

    if '-help' in inp_fs:
        print_help()
        exit()
	
    if method not in ["eval", "test", "screen", "infer"]:
        print(" '%s' Unknown Method. Exiting" % method)
        exit()

    #error checker
    for arg in inp_fs:
        if arg[0]=="-" and arg not in all_args:
            print("'%s' Unknown Option. Exiting" % arg)
            exit()
	    
    try:
        #arg_screener
        input_args={}
        for arg in bin_args:
            if arg in inp_fs:
                pos_val = inp_fs.index(arg)+1
                input_args[arg]=inp_fs[pos_val]


        if '-fpR'  in input_args: fpR  = int(input_args["-fpR"])
        if '-fpNB' in input_args: fpNB = int(input_args["-fpNB"])
        if '-max_dist' in input_args: max_dist  = float(input_args["-max_dist"])
        if '-min_sims' in input_args: min_sims  = int(input_args["-min_sims"])
        if '-max_sims' in input_args: max_sims  = int(input_args["-max_sims"])
        if '-in'  in input_args:  input_file  = input_args["-in"]
        if '-out' in input_args:  output_file  = input_args["-out"]
        if '-test' in input_args: test_file  = input_args["-test"]
        if '-scr' in input_args:  screen_file  = input_args["-scr"]
        if '-algo' in input_args: algo  = input_args["-algo"]
        if '-ntrees' in input_args: ntrees  = int(input_args["-ntrees"])
        if '-nprocs' in input_args: nprocs  = int(input_args["-nprocs"])
        if '-silent' in inp_fs: silent = True
        if '-detail' in inp_fs: detail = True
        if '-nostats' in inp_fs: stats = False
        if '-parallel' in inp_fs: parallel = True
        
    except:
        print("Illegal Option or Value. Exiting")
        exit()
        
    #now verify the impossibilities

    if algo not in ["SVM", "RF"]:
        print("Illegal algorithm option. Exiting")
        exit()
        
    if input_file is None: 
        print("No input file! Nothing to do. Exiting")
        exit()
    if method=="screen" and screen_file is None:
        print("No output file! A screen file (option '-scr') is required for screening. Exiting")
        exit()
    if method=="test" and test_file is None:
        print("No test file! A test file (option '-test') is required for testing models. Exiting")
        exit()
    if method=="infer" and test_file is None:
        print("No file for inference! A test file (option '-test') is required for testing or inferring data. Exiting")
        exit()
        
    #print(method, input_args)


    #method handler
    if method == "screen":
        if not silent: print("Screening!")
        res = screener(input_file, screen_file, fpR, fpNB, max_dist, max_sims, max_dist,
                       silent=silent, algo=algo, ntrees=ntrees, parallel=parallel, nprocs=nprocs)
        write_screen_results(res, output_file)
    elif method == "test":
        if not silent: print("Testing!")
        train_fname = input_file
        test_fname = test_file
        
        res = validate_model(input_file, test_file, fpR, fpNB, max_dist, max_sims,
                             algo=algo, ntrees=ntrees, silent=silent)

        write_results(res, output_file, detail, stats)

    elif method == "eval":
        if not silent: print("Evaluating!")
        res = evaluate_model(input_file, fpR, fpNB, max_dist, max_sims, algo=algo,
                             ntrees=ntrees, silent=silent)

        write_results(res, output_file, detail, stats)

    elif method == "infer":
        #this is basically the same thing as testing but without statistics, 
        #so the detail must ALWAYS be given 
        # as well as stats must NEVER be computed
        detail = True
        stats  = False
        if not silent: print("Inferring!")
        train_fname = input_file
        test_fname = test_file
        res = validate_model(input_file, test_file, fpR, fpNB, max_dist, max_sims,
                             algo=algo, ntrees=ntrees, silent=silent)

        write_results(res, output_file, detail, stats)
