import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.svm import SVC
import pickle


# df = pd.read_csv("../code/data/rec.csv")
# dfp1 = df[["plts","pltrs"]]
# fig, ax = plt.subplots(figsize = (9,9))
# sns.heatmap(dfp1,annot=True, vmax=10,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
# #sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
# #            square=True, cmap="YlGnBu")
# ax.set_title('plts', fontsize = 18)
# ax.set_ylabel('mentality', fontsize = 18)
# ax.set_xlabel('pltrs', fontsize = 18)
# plt.savefig("./confusion_matrix.jpg")
# plt.show()

df = pd.read_csv("./data/rec.csv")
label_encoder = preprocessing.LabelEncoder()

df['maxl']= label_encoder.fit_transform(df['maxl'])
df['avgl']= label_encoder.fit_transform(df['avgl'])
df['region']= label_encoder.fit_transform(df['region'])
df['rrate']= label_encoder.fit_transform(df['rrate'])
df['ind']= label_encoder.fit_transform(df['ind'])
df['catl']= label_encoder.fit_transform(df['catl'])
df['plts']= label_encoder.fit_transform(df['plts'])
df['pltrs']= label_encoder.fit_transform(df['pltrs'])
df['ots']= label_encoder.fit_transform(df['ots'])
df['blcl']= label_encoder.fit_transform(df['blcl'])
df['status']= label_encoder.fit_transform(df['status'])
df['trf']= label_encoder.fit_transform(df['trf'])
df['ur']= label_encoder.fit_transform(df['ur'])
df['cmt']= label_encoder.fit_transform(df['cmt'])

y = df["cmt"]
# df.drop(columns = ["cmt"], axis = 1,inplace=True)
df = df[["maxl","avgl","region","rrate","ind","catl","plts","pltrs","ots","blcl","status","trf","ur"]]
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=0,shuffle=True)

# Method selection
# 1.LogisticRegression
LR2 = LogisticRegression(solver='liblinear', multi_class='auto')
# 2.knn
knn = KNeighborsClassifier()
# 3.Defining the model Bayesian model
nb = naive_bayes.GaussianNB()
# 4.bp definition model
bp = MLPClassifier(max_iter=10000)
# 5.Decision Trees
dt = tree.DecisionTreeClassifier()
# 6.svm
svm = SVC(gamma='scale')


def confusion_matrix(i,y_true, y_pred, labels=None):
    n = len(labels)
    labels_dict = {label: i for i, label in enumerate(labels)}
    res = np.zeros([n, n], dtype=np.int32)
    for gold, predict in zip(y_true, y_pred):
        res[labels_dict[gold]][labels_dict[predict]] += 1
    df = pd.DataFrame(res, index=labels, columns=labels)
    sns.heatmap(df, annot=True, fmt='d')
    # plt.savefig("model_heatmap/"+i+"_confusion_matrix.jpg")
    plt.show()
    plt.cla()

def roccurve(y_label, y_pre,alg):
    fpr, tpr, thersholds = roc_curve(y_label, y_pre)
    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{alg} ROC Curve")
    plt.legend(loc="lower right")
    # plt.savefig(f"model_ROC_curve/{alg}roc curve.jpg")
    plt.show()
    plt.cla()

#Select the model with the best results
if __name__ == '__main__':
        count = 0
        num = 1
        sort_alg = [LR2,knn,nb,bp,dt,svm]
        alg_str = ["LR2","knn","nb","bp","dt","svm"]
        j=0
        for i in sort_alg:
            i.fit(X_train, y_train)
            score = i.score(X_test, y_test)
            y_pre = i.predict(X_test)
            confusion_matrix(alg_str[j], y_test, y_pre,[0,1])
            roccurve(y_test, y_pre,alg_str[j])
            j=j+1
            print('---------------------------------')
            print(f'the score of {i} method is ', score)
            count = count +score
            # pprint(count)
            # pre = [i/100 for i in pre ]
            # y_test = [i/100 for i in y_test]
            print('The predicted results are', y_pre)
            print('The actual results are', y_test)
        print(count/num)

#Save model
# fw = open("code/model/nb.pkl", "wb")
# nb = naive_bayes.GaussianNB()
# nb.fit(X, y)
# pickle.dump(nb, fw)
# fw.close()