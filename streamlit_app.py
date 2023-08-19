from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.title('Incoming Call Classification')

#######################
# 侧边栏，加载数据
#######################
data_path = Path('./ICC/data')
file_name = ['train', 'test']

train_data = pd.read_csv(data_path / 'rec.csv')
test_data = pd.read_csv(data_path / 'test.csv')

selected_file = st.sidebar.selectbox('选择需要加载的文件：', file_name)

# 加载数据
st.header('数据查看')
st.write('选择的数据为：', selected_file)
if selected_file == 'train':
    data = train_data
elif selected_file == 'test':
    data = test_data
else:
    data = None
    st.write('请选择正确的数据')
st.dataframe(data.head())

#######################
# 字段可视化
#######################
st.header('字段可视化')
columns = data.columns
# 离散字段
cate_cols = data.select_dtypes(include=['object', 'bool']).columns
# 连续字段
num_cols = data.select_dtypes(include=['int64', 'float64']).columns

selected_col = st.selectbox('选择需要呈现的字段：', columns)
st.write('字段类型为：', data[selected_col].dtype)

if selected_col in cate_cols:
    fig = plt.figure()
    sns.countplot(data=data, x=selected_col)
    st.pyplot(fig)
elif selected_col in num_cols:
    fig = plt.figure()
    sns.histplot(data=data, x=selected_col, kde=True)
    st.pyplot(fig)

#######################
# 模型预处理
#######################
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

for col in cate_cols.drop('cmt'):
    enc = LabelEncoder()
    train_data[col] = enc.fit_transform(train_data[col])
    test_data[col] = enc.transform(test_data[col])

# 标签编码
train_data['cmt'] = train_data['cmt'].map({'type1': 0, 'type2': 1})

# 数据标准化
scaler = StandardScaler()
scaler.fit(train_data[num_cols])
train_data[num_cols] = scaler.transform(train_data[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])

#######################
# 模型验证
#######################
st.header('模型验证')
models_name = ['LogisticRegression', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'MLPClassifier', 'SVC',
               'GaussianNB']

# 数据划分
X_train = train_data.drop(['cmt'], axis=1)
y_train = train_data['cmt']
X_test = test_data
X_trn, X_val, y_trn, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# 准备模型
lr = LogisticRegression()
dt = DecisionTreeClassifier(random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
mlp = MLPClassifier(random_state=0)
svm = SVC(random_state=0)
nb = GaussianNB()
models = [lr, dt, knn, mlp, svm, nb]


@st.cache()
def plot_cm_auc(y_true, y_pred):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axs[0])
    axs[0].set_title('Confusion Matrix')
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('Actual')

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    axs[1].plot(fpr, tpr)
    axs[1].set_title('ROC Curve')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')

    plt.tight_layout()
    return fig


if st.button('开始验证'):
    scores = []
    for model in models:
        model.fit(X_trn, y_trn)
        y_pred_val = model.predict(X_val)
        score = roc_auc_score(y_val, y_pred_val)
        st.write(model.__class__.__name__, ": ", '验证集AUC：', score)
        scores.append(score)
        fig = plot_cm_auc(y_val, y_pred_val)
        st.pyplot(fig)
    st.success("Finished")
    st.write('最佳模型为：', models_name[scores.index(max(scores))], '，AUC为：', max(scores))

#######################
# 模型预测
#######################
st.header('模型预测')
selected_model = st.selectbox('选择需要使用的模型：', models_name)
if st.button('开始预测'):
    model = models[models_name.index(selected_model)]
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    df_pred = pd.DataFrame({'cmt': y_pred_test})
    st.write('预测结果为：')
    fig = plt.figure()
    df_pred['cmt'].value_counts().plot.pie(autopct='%1.1f%%')
    st.pyplot(fig)
