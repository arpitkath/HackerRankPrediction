import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
def mutual_inf(a,b):
    from sklearn.metrics import mutual_info_score
    return mutual_info_score(a,b)
def create_binary_features(df, target):
    #from scipy.stats import chisquare
    binary_features = []
    for i in list(df.columns):
        if len(np.unique(df[i])) == 2 and i!="is_test":
            binary_features.append(i)
    print(binary_features)
    tot_mut_inf = 0
    count = 1
    for i in range(len(binary_features)):
        df[binary_features[i]] = df[binary_features[i]].astype(int)
    for i in range(len(binary_features)):
        for j in range(i+1,len(binary_features)):
            #print(df[binary_features[i]].dtypes,df[binary_features[j]].dtypes)
            if(mutual_inf(df[binary_features[i]][df.is_test==0]&df[binary_features[j]][df.is_test==0], target) > tot_mut_inf/count):
                df[str(binary_features[i])+"&"+str(binary_features[j])] = df[binary_features[i]]&df[binary_features[j]]
                tot_mut_inf += mutual_inf(df[binary_features[i]][df.is_test==0]&df[binary_features[j]][df.is_test==0], target)
                count += 1
            if(mutual_inf(df[binary_features[i]][df.is_test==0]|df[binary_features[j]][df.is_test==0], target) > tot_mut_inf/count):
                df[str(binary_features[i])+"|"+str(binary_features[j])] = df[binary_features[i]]|df[binary_features[j]]
                tot_mut_inf += mutual_inf(df[binary_features[i]][df.is_test==0]|df[binary_features[j]][df.is_test==0], target)
                count += 1
            if(mutual_inf(df[binary_features[i]][df.is_test==0]^df[binary_features[j]][df.is_test==0], target) > tot_mut_inf/count):
                df[str(binary_features[i])+"^"+str(binary_features[j])] = df[binary_features[i]]^df[binary_features[j]]
                tot_mut_inf += mutual_inf(df[binary_features[i]][df.is_test==0]^df[binary_features[j]][df.is_test==0], target)
                count += 1
        print(count, tot_mut_inf)
    return df

def build_cluster(df):
    #df = pd.DataFrame(df)
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    km = KMeans(n_clusters=7).fit(df)
    clusters = pd.DataFrame(km.labels_)
    col = list(df.columns)+["cluster_id"]
    print(type(df), type(clusters))

    df = pd.DataFrame(pd.concat([pd.DataFrame(df), pd.DataFrame(clusters)], axis=1))
    df.columns = col
    le = LabelEncoder().fit(df["cluster_id"])
    df["cluster_id"] = pd.Series(le.transform(df["cluster_id"]))
    temp = pd.get_dummies(df["cluster_id"], prefix="cluster", prefix_sep="_")
    df = pd.concat([df, temp], axis=1)
    return df

df = pd.read_csv("training_dataset.csv")
test = pd.read_csv("test_dataset.csv")
target = df["opened"]
le = LabelEncoder().fit(target)
target = le.transform(target)
drop_list = ["user_id", "mail_id", "open_time", "click_time", "unsubscribe_time", "clicked", "opened", "unsubscribed"]
df = df.drop(drop_list, axis=1)
drop_list = [i for i in drop_list if i in list(test.columns)]
test = test.drop(drop_list, axis=1)
categorical = ["mail_category", "mail_type", "hacker_confirmation"]
print(df.shape,test.shape)
df["is_test"] = 0
test["is_test"] = 1
df = df.append(test)

tim = ["sent_time", "hacker_created_at"]
from datetime import datetime
for i in tim:
    #df[i] = df[i] // 1000
    temp = df[i].apply(datetime.fromtimestamp)
   # print(temp)
    df[str(i)+"_month"] = temp.apply(lambda x:x.month)
    df[str(i)+"_day"] = temp.apply(lambda x:x.day)
    df[str(i)+"_hour"] = temp.apply(lambda x:x.hour)
    df[str(i)+"_minute"] = temp.apply(lambda x:x.minute)
#print(df["sent_time_day"])
for i in list(df.columns):
    df[i] = df[i].fillna(df[i].value_counts().index[0])
df["hacker_confirmation"] = df["hacker_confirmation"].astype(bool)
df["mail_type"] = df["mail_type"].astype(str)
df["mail_category"] = df["mail_category"].astype(str)

for cat in categorical:
    le = LabelEncoder().fit(df[cat])
    df[cat] = pd.Series(le.transform(df[cat]))

df["new_mail_category"] = 0
for i in [11,4,12,15,1,14]:
    df["new_mail_category"].ix[df["mail_category"] ==i ] = 1
df["new_mail_category"].ix[df["mail_category"]==6 ]= 2
df["new_mail_category"].ix[df["mail_category"]==0 ]= 2
df["new_mail_category"].ix[df["mail_category"]==9] = 3

df["new_mail_type"] = 0
df["new_mail_type"].ix[df["mail_type"]==0] = 1
df["new_mail_type"].ix[df["mail_type"]==1] = 1

df["new_contest_login_count"] = 0
for i in range(7):
    df["new_contest_login_count"].ix[df["contest_login_count"] == i] = 1
for i in range(7,27):
    df["new_contest_login_count"].ix[df["contest_login_count"] == i] = 2

df["new_contest_participation_count"] = 0
for i in range(9):
    df["new_contest_participation_count"].ix[df["contest_participation_count"] == i] = 1

df["new_submissions_count"] = 0
for i in range(5):
    df["new_submissions_count"].ix[df["submissions_count"] == i] = 1

df["new_forum_comment_count"] = 0
for i in range(8):
    df["new_forum_comment_count"].ix[df["forum_comments_count"] == i] = 1

binarize = ["new_mail_category", "new_mail_type", "new_contest_login_count", "new_contest_participation_count", "new_submissions_count", "new_forum_comment_count"]
for i in binarize:
    temp = df[i]
    temp = pd.get_dummies(pd.Series(temp),prefix_sep="_", prefix=i)
    df = pd.concat([df, temp], axis=1)
    #df.drop(i, inplace=True, axis=1)


#df = create_binary_features(df, target)
df["submissions_count_master/forum_count"] = df.submissions_count_master / (df.forum_count+1)
df["submissions_count/contest_login_count"] = df.submissions_count / (df.contest_login_count+1)
df["1/ipn_read"] = 1/(df.ipn_read+1)
df["last_online-sent_time"] = df.last_online - df.sent_time
df["ipn_read/submissions_count"] = df.ipn_read/(df.submissions_count_master+1)
df["forum_count/sumbssions_count"] = df.forum_count/(df.submissions_count+1)
df["is_sent_time_day"] = 0
df["is_sent_time_day"].ix[df["sent_time_hour"] >10] = 1
'''
for i in list(df.columns):
    print(i)
    print(df[i].value_counts())
    plt.hist(df[i])
    plt.xlabel(i)
    plt.show()
'''
test = df[df["is_test"] == 1]
df = df[df["is_test"] == 0]

df = df.drop(["is_test"], axis=1)
test = test.drop(["is_test"], axis=1)

print(df.shape,test.shape)
'''
def apply_pca(train, n):
    pca = PCA(n_components=n)
    temp = pd.DataFrame()
    for i in list(train.columns):
        if len(np.unique(train[i])) == 2:
            temp = pd.concat([temp, df[i]], axis=1)
            train.drop()

'''
'''
l=sorted(zip(feature_importances), key=lambda x:x[1], reverse=True)
thres = np.mean(list(dict(l).values()))
features = list(dict(filter(lambda x:x[1]>thres, l)).keys())
print(thres,len(features),features)
sol = pd.Series(rf.predict(test))
sol.to_csv("hackerrank.csv", index=False)
'''
'''
import xgboost as xgb
gbm  = xgb.XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=10,
 min_child_weight=6,
).fit(df, target)
sol = pd.Series(gbm.predict(test))
print col
print xgb.feature_importances_
sol.to_csv("hackerrank.csv", index=False)
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cross_validation import cross_val_score
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
pca = PCA(n_components=11)#11
df = pca.fit_transform(df)
from sklearn.svm import SVC
#sv = LogisticRegression()
#cross = cross_val_score(sv, df, target, cv=5)
#print(cross.mean())
test = pca.transform(test)
def voting_ensemble(model_list):
    pass


def stacking(train, test, target):
    from math import sqrt
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cross_validation import train_test_split
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from sklearn.svm import LinearSVC
    train1, train2, target1, target2= train_test_split(train, target, test_size=0.5, random_state=999)
    target = pd.Series(target1).append(pd.Series(target2))

    l0_models = [#BaggingClassifier(LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.1), n_estimators=1000, verbose=1, bootstrap_features=True),
              #LinearDiscriminantAnalysis(solver='eigen', n_components=int(sqrt(train.shape[1])),shrinkage=0.01),
              #LogisticRegression(C=0.1),
              RandomForestClassifier(n_estimators=300,n_jobs=-1),#617744,61928
              RandomForestClassifier(n_estimators=300, criterion='entropy', n_jobs=-1),
              ExtraTreesClassifier(n_estimators=300, verbose=1),
              ExtraTreesClassifier(n_estimators=300, verbose=1, criterion='entropy'),
              #GradientBoostingClassifier(learning_rate=0.1),
              #GradientBoostingClassifier(loss='exponential', learning_rate=0.75, n_estimators=300),
              #QuadraticDiscriminantAnalysis(reg_param=0.5),
              #QuadraticDiscriminantAnalysis(reg_param=0.1),
              #QuadraticDiscriminantAnalysis(reg_param=0.75),
              #KNeighborsClassifier(n_neighbors=2, weights='distance'),
              #KNeighborsClassifier(n_neighbors=4, weights='distance'),
              #KNeighborsClassifier(n_neighbors=5, weights='distance'),
              #LinearSVC(C=0.2),
              #LinearSVC(C=0.5)
        ]


    def layer_formation(models):
        trainPredictDf = pd.DataFrame()
        testPredictDf = pd.DataFrame()
        sm = 0
        for model in models:
            print(model)
            model1, model2 = clone(model), clone(model)
            model1.fit(train1, target1)
            model2.fit(train2, target2)
            predict1 = pd.Series(model1.predict_proba(train2)[:,1])
            predict2 = pd.Series(model2.predict_proba(train1)[:,1])
            predict = predict2.append(predict1)
            predict[predict < 0.365] = 0#41-56.5,40-58.6,35-58.7,
            predict[predict >= 0.365] = 1
            predict = predict.astype(int)
            f1 = f1_score(target, predict)
            sm += f1
            print(f1)
            trainPredictDf = pd.concat([trainPredictDf, predict], axis=1)

            #model.fit(train1, target1)
            model1.fit(train2, target2)
            testPred = pd.Series(model1.predict_proba(test)[:,1])
            testPred[testPred < 0.365] = 0
            testPred[testPred >= 0.365] = 1
            testPred = testPred.astype(int)
            testPredictDf = pd.concat([testPredictDf, testPred], axis=1)
        print("Avg: ",sm/len(l0_models))
        return trainPredictDf, testPredictDf

    trainPredictDf, testPredictDf = layer_formation(l0_models)
    print(trainPredictDf.shape, testPredictDf.shape)
    temp = testPredictDf.apply(np.sum, axis=1)
    temp2 = testPredictDf.apply(np.sum, axis=1)
    temp3 = testPredictDf.apply(np.sum, axis=1)
    temp[temp < 1 ] = 0
    temp[temp >= 1 ] = 1
    temp2[temp2 < 2 ] = 0
    temp2[temp2 >= 2 ] = 1
    temp3[temp3 < 3 ] = 0
    temp3[temp3 >= 3 ] = 1
    temp.to_csv("voting_hackerrank_1.csv", index=False)
    temp2.to_csv("voting_hackerrank_2.csv", index=False)
    temp3.to_csv("voting_hackerrank_3.csv", index=False)
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(trainPredictDf, target)
    sol = pd.Series(rf.predict(testPredictDf))
    sol.to_csv("stacking_hackerrank.csv", index=False)

stacking(df, test, target)