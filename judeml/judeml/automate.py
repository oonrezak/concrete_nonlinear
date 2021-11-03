import time
# from datetime import timedelta
from collections import Counter
import pandas as pd
import numpy as np
import warnings
import math

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from .regression.automate import Judas as JudasRegressor
from .classification.automate import Judas as JudasClassifier

class General():

    def __init__(self, DEBUG=False):
        self.DEBUG=DEBUG
        return

    def sensi(self, X, y, tp='cls'):
        Xn = np.array(X)
        if tp == 'cls':
            clf = KNeighborsClassifier()
        elif tp == 'reg':
            clf = KNeighborsRegressor()
        number_features = Xn.shape[1]
        
        lst=[]
        for i in range(number_features):
            dict_1 = {}
            Xp = Xn[:, i].reshape(-1, 1)
            scores = cross_val_score(clf, Xp, y, cv = 3)
            dict_1={'Feature':X.columns[i],'Accuracy':scores.mean()}
            lst.append(dict_1)
        return pd.DataFrame(lst)
    
    def sensi_feat(self, X, y, n=3, tp='cls'):
        Xn = np.array(X)
        if tp == 'cls':
            clf = KNeighborsClassifier()
        elif tp == 'reg':
            clf = KNeighborsRegressor()
        number_features = Xn.shape[1]
        print('Num Feature  Accuracy')
        lst=[]
        idx=[]
        for i in range(number_features):
            X_head = np.atleast_2d(Xn[:, 0:i])
            X_tail = np.atleast_2d(Xn[:, i+1:])
            Xp = np.hstack((X_head, X_tail))
            scores = cross_val_score(clf, Xp, y, cv = 3)
            lst.append(scores.mean())
            idx.append(i+1)
            print('%d        %g' % (i+1, scores.mean()))
        plt.plot(idx,lst)
        plt.ylabel("accuracy")
        plt.xlabel("number of features")
        plt.show();
        
    def pcc(self,target):
        """Displays PCC Details
        """
        state_counts = Counter(target)
        print(state_counts)
        df_state = pd.DataFrame.from_dict(state_counts, orient='index')
        df_state.plot(kind='bar')

        num = (df_state[0]/df_state[0].sum())**2
        print("State Count: {}\n".format(df_state))
        print("1.25 * Proportion Chance Criterion: {}%".format(1.25*100*num.sum()))
        
    def heatmap(self,X):
        corr = X.corr()
        plt.figure(figsize=(13,7))
        ax = sns.heatmap(
            corr, 
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        );
        
    def LassoFS(self,X,y):
        trials = 10
        lassoFS = JudasRegressor()
        params = [
            {'model' : 'lasso', 'trials' : trials},
        ]
        lassoFS.automate(X,y,params)
        f = pd.DataFrame(lassoFS.models[0].coefl)
        i=0
        lstlasso=[]
        for fc in f.columns:
            fet = [X.columns[key] for key in np.argsort(abs(f[fc].mean()))[::-1]]
            dict_1 = { "alpha" : fc, "top": fet, "acc":lassoFS.models[0].score[i]}
            lstlasso.append(dict_1)
            i+=1
        summary_lasso=pd.DataFrame(lstlasso)
        return summary_lasso.sort_values('acc',ascending=False)

    
    def OoOKNN(self,X,y,tp='cls'):
        trials = 1
        if tp == 'cls':
            oKNN = JudasClassifier()
        elif tp == 'reg':
            oKNN = JudasRegressor()
        params = [
            {'model' : 'knn', 'trials' : trials, 'k' : range(1,30)},
        ]
        oKNN.automate(X,y,params)
        lsttabs = []
        for Xx in X.columns:
            Xcept = X.loc[:, X.columns != Xx ]
            dict_1 = {}
            print(Xx)
            oKNN.automate(Xcept,y,params)
            dict_1 = {'column':Xx,'accuracy':oKNN.score()['Test Accuracy'][0]}
            lsttabs.append(dict_1)
    
        topval =  pd.DataFrame(lsttabs).sort_values('accuracy',ascending=False).reset_index(drop=True)
        return topval