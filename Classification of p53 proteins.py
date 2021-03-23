#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.pyplot import figure
import seaborn as sns
import statistics as st


# In[2]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# In[3]:


from imblearn.under_sampling import NearMiss
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# In[4]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve


# In[6]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy


# In[7]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV


# In[8]:


import warnings
warnings.filterwarnings("ignore")


# In[9]:


# read the data into a Pandas DataFrame
df = pd.read_csv('~/datasets/K9.data', sep=',', header=None, prefix= 'Feature', low_memory=False)


# In[8]:


df.head(20)


# In[9]:


# see the dimensions
print(df.shape)


# In[13]:


# see the class proportion
print(df['Feature5408'].value_counts())
sns.countplot(df['Feature5408'], label="count")


# # DATA PREPROCESSING & EXPLORATION

# ## Data Pre-processing

# In[10]:


# make the class binary
df['Feature5408'] = LabelEncoder().fit_transform(df['Feature5408'])


# In[11]:


# rename the column of class for classification
df.rename(columns={'Feature5408':'class'}, inplace=True) 


# ### Missing Detection

# In[12]:


#make all missing same type
df.replace('?', np.NaN, inplace=True)
df.replace('', np.NaN, inplace=True)
df.replace('NA', np.NaN, inplace=True)
df.replace('None', np.NaN, inplace=True)
df.replace('-', np.NaN, inplace=True)
df.replace('na', np.NaN, inplace=True)
df.replace('N/A', np.NaN, inplace=True)
df.replace('n/a', np.NaN, inplace=True)


# In[16]:


#check amount of missing by column
df.isnull().sum()


# In[17]:


#percentage of missing by column
df.isnull().sum()/len(df)*100


# In[18]:


#check amount of missing by row
for i in range(len(df.index)) :
    if df.iloc[i].isnull().sum() != 0:
       print("NaN in row ",i , " : " ,  df.iloc[i].isnull().sum()/len(df.columns)*100)


# In[20]:


#the number of rows has missing values : 261 rows have missing values
sum([True for idx,row in df.iterrows() if any(row.isnull())])


# In[13]:


#remove the column with all NaN 
df.dropna(axis='columns', how='all', inplace=True) 


# In[14]:


#drop all rows have missing
df.dropna(axis=0, inplace=True)


# In[10]:


#check the column type
for y in df.columns:
    print(y, df[y].dtype)


# In[15]:


# convert columns types from object to numeric 
df.apply(pd.to_numeric)


# ### Dublications

# In[123]:


# check dublications : 6 dublicated rows are found
duplicateddf = df[df.duplicated()]
print(duplicateddf)


# In[16]:


# delete dublicate rows
df.drop([23118,26780,26994,30424,30635,31285],axis=0, inplace=True )


# In[138]:


# see the dimensions after cleaning
print(df.shape)


# In[150]:


# see the class proportion after cleaing
print(df['class'].value_counts())
sns.countplot(df['class'], label="count")


# ### Outlier Detection

# In[24]:


#Split features and class for analysis
features = df.drop('class', axis = 1)
target = df['class']


# In[25]:


#Convert to numeric
features = features.apply(pd.to_numeric)


# In[26]:


# feature visualization (columns 0:100)
features.iloc[:,0:100].plot(kind='box', rot=90, figsize=(14, 8))
plt.tight_layout()
plt.show()


# In[27]:


# feature visualization (columns 100:200)
features.iloc[:,4000:5000].plot(kind='box', rot=90, figsize=(20, 12))
plt.tight_layout()
plt.show()


# In[132]:


# outlier analysis 

Q1 = features.iloc[:,0:5408].quantile(0.25)
Q3 = features.iloc[:,0:5408].quantile(0.75)
IQR = Q3 - Q1

out = ((features.iloc[:,0:5408] < (Q1 - 1.5 * IQR)) | (features.iloc[:,0:5408] > (Q3 + 1.5 * IQR))).sum()

print(st.mean(out/31159))

figure(figsize=(20,10))
plt.style.use('ggplot')
plt.xticks(rotation=90)
plt.plot(out/31159, color = 'blue')


# ## Data Expolaration

# ### k-Means

# In[94]:


x_kmns = df.drop('class', axis = 1)
y_kmns = df['class']


# In[95]:


#Scaling 
scaler = StandardScaler()
scaler.fit(x_kmns)
x_kmns_scd = scaler.transform(x_kmns)


# In[96]:


#Convert to dataframe for usability in the algorithms
x_kmns_scd= pd.DataFrame(x_kmns_scd)


# In[101]:


#Define the method and transform the dataset
kmeans2 = KMeans(n_clusters=2)
kmeans2.fit(x_kmns_scd)
y_kmeans2 = kmeans2.predict(x_kmns_scd)


# In[110]:


#K-Means Feature 0-1
plt.scatter(x_kmns_scd.iloc[:, 0], x_kmns_scd.iloc[:, 1], c=y_kmeans2, s=50, cmap='viridis')

centers2 = kmeans2.cluster_centers_
plt.scatter(centers2[:, 0], centers2[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# In[103]:


#K-Means Feature 1-2
plt.scatter(x_kmns_scd.iloc[:, 1], x_kmns_scd.iloc[:, 2], c=y_kmeans2, s=50, cmap='viridis')

centers2 = kmeans2.cluster_centers_
plt.scatter(centers2[:, 0], centers2[:, 1], c='black', s=200, alpha=0.5)


# In[105]:


#K-Means Feature 2-3
plt.scatter(x_kmns_scd.iloc[:, 2], x_kmns_scd.iloc[:, 3], c=y_kmeans2, s=50, cmap='viridis')
centers2 = kmeans2.cluster_centers_
plt.scatter(centers2[:, 0], centers2[:, 1], c='black', s=200, alpha=0.5)


# ### PCA

# In[15]:


x_pcax= df.drop('class', axis = 1)
y_pcax = df['class']

#Scaling 
scaler = StandardScaler()
scaler.fit(x_pcax)
x_scaled = scaler.transform(x_pcax)


# In[182]:


#PCA : components = 300, %80 of the variance explained with 300 dimensions.
# define the method and transform the dataset

pca1 = PCA(n_components = 300)
xpcax =  pca1.fit_transform(x_scaled)

pca1.explained_variance_ratio_

print(pca1.explained_variance_ratio_.cumsum())

plt.plot(pca1.explained_variance_ratio_)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.ylim(0,1)


# In[20]:


#PCA : components = 10
# define the method and transform the dataset

pca = PCA(n_components = 10)
x_pca =  pca.fit_transform(x_scaled)

pca.explained_variance_ratio_

print(pca.explained_variance_ratio_.cumsum())

plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')


# In[21]:


#Convert to dataframe for usability in the algorithms
x_scaled = pd.DataFrame(x_scaled)
y_pcax = pd.DataFrame(y_pcax)


# In[22]:


#Help codes for scatter plot
n_components=10
pca_columns = []
for i in range(1,n_components+1):
    pca_columns.append('principal_component'+str(i))


# In[23]:


#Convert to dataframe for usability in the algorithms
pca_data = pd.DataFrame(x_pca,index=x_scaled.index,columns=pca_columns)


# In[25]:


#Add class variable into pca_data set
pca_data['class'] = y_pcax['class'].values


# In[27]:


#Pairwise scatter plot of the continous features
sns.set(style="ticks")
flatui = ["#EA1715", "#26C0EA"]
#sns.palplot(sns.color_palette(flatui))
sns.pairplot(pca_data, hue="class",palette = flatui,vars = pca_columns[0:4],
             plot_kws=dict(s=40, #edgecolor="white", 
                           linewidth=2.5))


# In[30]:


#PC1 & PC2
plt.scatter(pca_data.iloc[:, 0], pca_data.iloc[:, 1], 
            c = pca_data["class"],edgecolor='red')
plt.xlabel('component 1')
plt.ylabel('component 2')


# In[90]:


#PC2 & PC3
plt.scatter(pca_data.iloc[:, 1], pca_data.iloc[:, 2], 
            c = pca_data["class"],edgecolor='red')
plt.xlabel('component 2')
plt.ylabel('component 3')


# In[92]:


#PC3 & PC4
plt.scatter(pca_data.iloc[:, 0], pca_data.iloc[:, 2], 
            c = pca_data["class"],edgecolor='red')
plt.xlabel('component 1')
plt.ylabel('component 3')


# In[ ]:





# ### Oversampling

# In[17]:


# define the method and transform the dataset
smote=SMOTE()
x_over, y_over = smote.fit_sample(x_scaled,y_pcax)


# In[86]:


# Plot helper function
def draw_plot(X, y, label):
   for l in np.unique(y):
      plt.scatter(
         X[y==l, 0],
         X[y==l, 1],
         label=l
      )
   plt.title(label)
   plt.xlabel("Feature0")
   plt.ylabel("Feature1")  
   plt.legend()
   plt.show()

# plot the examples by class label
draw_plot(x_over, y_over, "Oversampled data")


# In[109]:


# Plot helper function
def draw_plot(X, y, label):
   for l in np.unique(y):
      plt.scatter(
         X[y==l, 1],
         X[y==l, 2],
         label=l
      )
   plt.title(label)
   plt.xlabel("Feature1")
   plt.ylabel("Feature2")  
   plt.legend()
   plt.show()

# plot the examples by class label
draw_plot(x_over, y_over, "Oversampled data")


# ### Undersampling

# In[108]:


# Define the undersampling method
undersample = NearMiss(version=3, n_neighbors_ver3=3)
# transform the dataset
x_under, y_under = undersample.fit_resample(x_scaled, y_pcax)
# summarize the new class distribution
counter = Counter(y_under)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = where(y_under == label)[0] 
    plt.scatter(x_under[row_ix, 1], x_under[row_ix, 2], label=str(label))
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ### t-SNE

# In[17]:


# Sample the data for computation time
seed = 0
df_samp3 = df.sample(frac = 0.5, random_state=seed)


# In[18]:


X_df3 = df_samp3.drop('class', axis = 1)
y_df3 = df_samp3['class']


# In[20]:


#Scaling 
scaler = StandardScaler()
scaler.fit(X_df3)
X_sc3= scaler.transform(X_df3)


# In[21]:


# Define the method and transform the dataset
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results3 = tsne.fit_transform(X_sc3)


# In[22]:


tsne_results3


# In[37]:


#Convert to dataframe for usability in the algorithms
y_df3 = pd.DataFrame(y_df3)
X_sc3 = pd.DataFrame(X_sc3)


# In[38]:


#Add class variable into pca_data set
X_sc3["y"] = y_df3["class"].values


# In[23]:



kmeans_sne = KMeans(n_clusters=2)
kmeans_sne.fit(tsne_results3)
y_kmeans_sne = kmeans_sne.predict(tsne_results3)


# In[25]:


tsne_results3_k = pd.DataFrame(tsne_results3)


# In[26]:



plt.scatter(tsne_results3_k.iloc[:, 0], tsne_results3_k.iloc[:, 1], c=y_kmeans_sne, s=50, cmap='viridis')

centers_sne = kmeans_sne.cluster_centers_
plt.scatter(centers_sne[:, 0], centers_sne[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel("tsne1")
plt.ylabel("tsne2")


# In[39]:


#Scatter plot of examples by class label

X_sc3['tsne-2d-one'] = tsne_results3[:,0]
X_sc3['tsne-2d-two'] = tsne_results3[:,1]
plt.figure(figsize=(16,10))

sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 2),
    hue= "y",
    data=X_sc3,
    legend="full",
    alpha=0.3
)


# ### PCA & t-SNE

# In[27]:


# Define the method and transform the dataset
pca = PCA(n_components = 300)
df_pca =  pca.fit_transform(X_sc3)


# In[29]:


kmeans_pca = KMeans(n_clusters=2)
kmeans_pca.fit(df_pca)
y_kmeans_pca = kmeans_pca.predict(df_pca)


# In[31]:


df_pca2 = pd.DataFrame(df_pca)


# In[44]:


plt.scatter(df_pca2.iloc[:, 0], df_pca2.iloc[:, 200], c=y_kmeans_pca, s=50, cmap='viridis')

centers_pca = kmeans_pca.cluster_centers_
plt.scatter(centers_pca[:, 0], centers_pca[:, 200], c='black', s=200, alpha=0.5)
plt.xlabel("pc1")
plt.ylabel("pc2")


# In[82]:


# Define the method and transform the dataset

tsne_pca = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
pca_tsne_results = tsne_pca.fit_transform(df_pca)


# In[83]:


# Scatter plot of examples by class label

X_sc3['tsne-pca-one'] = pca_tsne_results[:,0]
X_sc3['tsne-pca-two'] = pca_tsne_results[:,1]
plt.figure(figsize=(16,10))

sns.scatterplot(
    x="tsne-pca-one", y="tsne-pca-two",
    palette=sns.color_palette("hls", 2),
    hue= "y",
    data=X_sc3,
    legend="full",
    alpha=0.3
)


# # SELECTION OF METHODS

# In[107]:


# get a random sample from the main data to reduce the computation time
seed = 3
df_s = df.sample(frac = 0.40, random_state=seed)


# In[108]:


# see the dimensions of sample data
print(df_s.shape)


# In[104]:


X = df_s.drop('class', axis = 1)
y = df_s['class']


# In[176]:


#Scaling Methods - LogisticRegression

random_state = 0
skf = KFold(n_splits=5, shuffle=True, random_state=random_state) 
fold_idxs = list(skf.split(X))

clf = LogisticRegression() 

for name, scaler in [
('no scaling', None), ('min_max', MinMaxScaler()), ('robust', RobustScaler()), ('standard', StandardScaler())]:

    scores = []

    for i, (train_idx, test_idx) in enumerate(fold_idxs): 
        X_train = X.values[train_idx]
        y_train = y.values[train_idx]
        X_val = X.values[test_idx]
        y_val = y.values[test_idx]

        if scaler:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train) 
            X_val = scaler.transform(X_val)

        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_val)
        scores.append(roc_auc_score(y_val, y_pred[:, 1]))

    print(name, 'AUROC:', np.mean(scores))


# In[105]:


#Scaling Methods - KNN

random_state = 0
skf = KFold(n_splits=5, shuffle=True, random_state=random_state) 
fold_idxs = list(skf.split(X))

clf = KNeighborsClassifier(n_neighbors=5) 

for name, scaler in [
('no scaling', None), ('min_max', MinMaxScaler()), ('robust', RobustScaler()), ('standard', StandardScaler())]:

    scores = []

    for i, (train_idx, test_idx) in enumerate(fold_idxs): 
        X_train = X.values[train_idx]
        y_train = y.values[train_idx]
        X_val = X.values[test_idx]
        y_val = y.values[test_idx]

        if scaler:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train) 
            X_val = scaler.transform(X_val)

        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_val)
        scores.append(roc_auc_score(y_val, y_pred[:, 1]))

    print(name, 'AUROC:', np.mean(scores))


# In[180]:


#Scaling Methods - RandomForest

random_state = 0
skf = KFold(n_splits=5, shuffle=True, random_state=random_state) 
fold_idxs = list(skf.split(X))

clf = RandomForestClassifier(random_state = random_state)

for name, scaler in [
('no scaling', None), ('min_max', MinMaxScaler()), ('robust', RobustScaler()), ('standard', StandardScaler())]:

    scores = []

    for i, (train_idx, test_idx) in enumerate(fold_idxs): 
        X_train = X.values[train_idx]
        y_train = y.values[train_idx]
        X_val = X.values[test_idx]
        y_val = y.values[test_idx]

        if scaler:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train) 
            X_val = scaler.transform(X_val)

        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_val)
        scores.append(roc_auc_score(y_val, y_pred[:, 1]))

    print(name, 'AUROC:', np.mean(scores))


# In[181]:


#Scaling Method - NaiveBayes

random_state = 0
skf = KFold(n_splits=5, shuffle=True, random_state=random_state) 
fold_idxs = list(skf.split(X))

clf = GaussianNB()

for name, scaler in [
('no scaling', None), ('min_max', MinMaxScaler()), ('robust', RobustScaler()), ('standard', StandardScaler())]:

    scores = []

    for i, (train_idx, test_idx) in enumerate(fold_idxs): 
        X_train = X.values[train_idx]
        y_train = y.values[train_idx]
        X_val = X.values[test_idx]
        y_val = y.values[test_idx]

        if scaler:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train) 
            X_val = scaler.transform(X_val)

        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_val)
        scores.append(roc_auc_score(y_val, y_pred[:, 1]))

    print(name, 'AUROC:', np.mean(scores))


# ###### At the end of this phase, Min-MaxScaler was selected as scaling method.

# # MODEL BUILDING

# ## Prepare data

# In[16]:


# Prepare main data 
X1 = df.drop('class', axis = 1)
y1 = df['class']


# In[17]:


# Normalization
scaler = MinMaxScaler()
scaler.fit(X1)
X1 = scaler.transform(X1)


# In[18]:


#Left a part of the data apart as a test data set
x_m, x_test, y_m, y_test = train_test_split(X1, y1, test_size=0.2, random_state=0)


# In[19]:


#To be able to split in the cross-validation
x_m = pd.DataFrame(x_m)


# ## Models without oversampling and dimension reduction

# Models are trained with original imbalance data.

# In[95]:


#LOGISTIC REGRESSION & NAIVE BAYES 

random_state = 0 
skf = KFold(n_splits=5, shuffle=True, random_state=random_state) 
fold_idxs = list(skf.split(x_m))


for name, model in [ ('LR' ,LogisticRegression(class_weight="balanced")) ,('NB', GaussianNB())]:

        acc_scores = []
        auc_scores = []
        precision_scores = []
        recall_scores = []
        balanced_acc = []
        names = []
        y_probs = []
        y_vals = []
        
        
        f, axes = plt.subplots(1, 2, figsize=(10, 5))

        for i, (train_idx, test_idx) in enumerate(fold_idxs): 
            X_train = x_m.values[train_idx]
            y_train = y_m.values[train_idx]
            X_val = x_m.values[test_idx]
            y_val = y_m.values[test_idx]
            
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)
            y_pred = model.predict(X_val)
            y_probs.append(y_prob[:,1])
            y_vals.append(y_val)

            auc_scores.append(roc_auc_score(y_val, y_prob[:,1]))
            acc_scores.append(accuracy_score(y_val,y_pred))
            precision_scores.append(precision_score(y_val, y_pred))
            recall_scores.append(recall_score(y_val,y_pred))
            balanced_acc.append(balanced_accuracy_score(y_val,y_pred))
            names.append(name)
            
            precision, recall, _ = precision_recall_curve(y_val, y_prob[:,1])
            fpr, tpr, _ = roc_curve(y_val, y_prob[:,1])
            
            lab1 = 'Fold %d AUC=%.4f' % (i+1, auc(fpr, tpr))
            lab2 = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
            axes[0].step(fpr,tpr,label = lab1)
            axes[1].step(recall, precision, label=lab2)
            y_vals.append(y_val)
            y_probs.append(y_prob[:,1])
               
            
        print(name, ':\n', 'AUROC:', round(np.mean(auc_scores),3), 'Balanced ACC:', round(np.mean(balanced_acc),3), 'ACC:', round(np.mean(acc_scores),3), 'Precision:', round(np.mean(precision_scores),3), 
                 'Recall:', round(np.mean(recall_scores),3), '\n')
        
        y_vals = numpy.concatenate(y_vals)
        y_probs = numpy.concatenate(y_probs)
        precision, recall, _ = precision_recall_curve(y_vals, y_probs)
        fpr, tpr, _ = roc_curve(y_vals, y_probs)
        lab1 = 'Overall AUC=%.4f' % (auc(fpr,tpr))
        lab2 = 'Overall AUC=%.4f' % (auc(recall, precision))
        axes[0].step(fpr,tpr, label = lab1, lw=2, color = 'black')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend(loc='lower right', fontsize='small')
        axes[1].step(recall, precision, label=lab2, lw=2, color='black')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].legend(loc='lower left', fontsize='small')
        f.tight_layout()


# In[97]:


#RANDOM FOREST & KNN

random_state = 0 
skf = KFold(n_splits=5, shuffle=True, random_state=random_state) 
fold_idxs = list(skf.split(x_m))


for name, model in [ ('RF', RandomForestClassifier(random_state = 0, class_weight="balanced")), ('KNN', KNeighborsClassifier())]:

        acc_scores = []
        auc_scores = []
        precision_scores = []
        recall_scores = []
        balanced_acc = []
        names = []
        y_probs = []
        y_vals = []
        
        f, axes = plt.subplots(1, 2, figsize=(10, 5))

        for i, (train_idx, test_idx) in enumerate(fold_idxs): 
            X_train = x_m.values[train_idx]
            y_train = y_m.values[train_idx]
            X_val = x_m.values[test_idx]
            y_val = y_m.values[test_idx]
            
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)
            y_pred = model.predict(X_val)
            y_probs.append(y_prob[:,1])
            y_vals.append(y_val)

            auc_scores.append(roc_auc_score(y_val, y_prob[:,1]))
            acc_scores.append(accuracy_score(y_val,y_pred))
            precision_scores.append(precision_score(y_val, y_pred))
            recall_scores.append(recall_score(y_val,y_pred))
            balanced_acc.append(balanced_accuracy_score(y_val,y_pred))
            names.append(name)
            
            precision, recall, _ = precision_recall_curve(y_val, y_prob[:,1])
            fpr, tpr, _ = roc_curve(y_val, y_prob[:,1])
            
            lab1 = 'Fold %d AUC=%.4f' % (i+1, auc(fpr, tpr))
            lab2 = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
            axes[0].step(fpr,tpr,label = lab1)
            axes[1].step(recall, precision, label=lab2)
            y_vals.append(y_val)
            y_probs.append(y_prob[:,1])
               
            
        print(name, ':\n', 'AUROC:', round(np.mean(auc_scores),3),'Balanced ACC:', round(np.mean(balanced_acc),3), 'ACC:', round(np.mean(acc_scores),3), 'Precision:', round(np.mean(precision_scores),3), 
                 'Recall:', round(np.mean(recall_scores),3), '\n')
        
        y_vals = numpy.concatenate(y_vals)
        y_probs = numpy.concatenate(y_probs)
        precision, recall, _ = precision_recall_curve(y_vals, y_probs)
        fpr, tpr, _ = roc_curve(y_vals, y_probs)
        lab1 = 'Overall AUC=%.4f' % (auc(fpr,tpr))
        lab2 = 'Overall AUC=%.4f' % (auc(recall, precision))
        axes[0].step(fpr,tpr, label = lab1, lw=2, color = 'black')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend(loc='lower right', fontsize='small')
        axes[1].step(recall, precision, label=lab2, lw=2, color='black')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].legend(loc='lower left', fontsize='small')
        f.tight_layout()


# ## Models with oversampled data

# Models are trained with oversampled and balanced data.

# In[42]:


#Logistic Regression & Naive Bayes

random_state = 0 
skf = KFold(n_splits=5, shuffle=True, random_state=random_state) 
fold_idxs = list(skf.split(x_m))


for name, model in [ ('LR' ,LogisticRegression()) ,('NB', GaussianNB())]:

        acc_scores = []
        auc_scores = []
        precision_scores = []
        recall_scores = []
        names = []
        y_probs = []
        y_vals = []
        
        f, axes = plt.subplots(1, 2, figsize=(10, 5))

        for i, (train_idx, test_idx) in enumerate(fold_idxs): 
            X_train = x_m.values[train_idx]
            y_train = y_m.values[train_idx]
            X_val = x_m.values[test_idx]
            y_val = y_m.values[test_idx]

            #OverSampling
            smote= SMOTE()
            x_resamp, y_resamp = smote.fit_resample(X_train, y_train)
            
            model.fit(x_resamp, y_resamp)
            y_prob = model.predict_proba(X_val)
            y_pred = model.predict(X_val)
            y_probs.append(y_prob[:,1])
            y_vals.append(y_val)

            auc_scores.append(roc_auc_score(y_val, y_prob[:,1]))
            acc_scores.append(accuracy_score(y_val,y_pred))
            precision_scores.append(precision_score(y_val, y_pred))
            recall_scores.append(recall_score(y_val,y_pred))
            names.append(name)
            
            precision, recall, _ = precision_recall_curve(y_val, y_prob[:,1])
            fpr, tpr, _ = roc_curve(y_val, y_prob[:,1])
            
            lab1 = 'Fold %d AUC=%.4f' % (i+1, auc(fpr, tpr))
            lab2 = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
            axes[0].step(fpr,tpr,label = lab1)
            axes[1].step(recall, precision, label=lab2)
            y_vals.append(y_val)
            y_probs.append(y_prob[:,1])
               
            
        print(name, ':\n', 'AUROC:', round(np.mean(auc_scores),3), 'ACC:', round(np.mean(acc_scores),3), 'Precision:', round(np.mean(precision_scores),3), 
                 'Recall:', round(np.mean(recall_scores),3), '\n')
        
        y_vals = numpy.concatenate(y_vals)
        y_probs = numpy.concatenate(y_probs)
        precision, recall, _ = precision_recall_curve(y_vals, y_probs)
        fpr, tpr, _ = roc_curve(y_vals, y_probs)
        lab1 = 'Overall AUC=%.4f' % (auc(fpr,tpr))
        lab2 = 'Overall AUC=%.4f' % (auc(recall, precision))
        axes[0].step(fpr,tpr, label = lab1, lw=2, color = 'black')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend(loc='lower right', fontsize='small')
        axes[1].step(recall, precision, label=lab2, lw=2, color='black')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].legend(loc='lower left', fontsize='small')
        f.tight_layout()
        


# In[37]:


#Random Forest & KNN

random_state = 0 
skf = KFold(n_splits=5, shuffle=True, random_state=random_state) 
fold_idxs = list(skf.split(x_m))


for name, model in [('RF', RandomForestClassifier(random_state = 0)), ('KNN', KNeighborsClassifier())]:

        acc_scores = []
        auc_scores = []
        precision_scores = []
        recall_scores = []
        names = []
        y_probs = []
        y_vals = []
        
        f, axes = plt.subplots(1, 2, figsize=(10, 5))

        for i, (train_idx, test_idx) in enumerate(fold_idxs): 
            X_train = x_m.values[train_idx]
            y_train = y_m.values[train_idx]
            X_val = x_m.values[test_idx]
            y_val = y_m.values[test_idx]

            #OverSampling
            smote= SMOTE()
            x_resamp, y_resamp = smote.fit_resample(X_train, y_train)
            
            model.fit(x_resamp, y_resamp)
            y_prob = model.predict_proba(X_val)
            y_pred = model.predict(X_val)
            y_probs.append(y_prob[:,1])
            y_vals.append(y_val)

            auc_scores.append(roc_auc_score(y_val, y_prob[:,1]))
            acc_scores.append(accuracy_score(y_val,y_pred))
            precision_scores.append(precision_score(y_val, y_pred))
            recall_scores.append(recall_score(y_val,y_pred))
            names.append(name)
            
            precision, recall, _ = precision_recall_curve(y_val, y_prob[:,1])
            fpr, tpr, _ = roc_curve(y_val, y_prob[:,1])
            
            lab1 = 'Fold %d AUC=%.4f' % (i+1, auc(fpr, tpr))
            lab2 = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
            axes[0].step(fpr,tpr,label = lab1)
            axes[1].step(recall, precision, label=lab2)
            y_vals.append(y_val)
            y_probs.append(y_prob[:,1])
               
            
        print(name, ':\n', 'AUROC:', round(np.mean(auc_scores),3), 'ACC:', round(np.mean(acc_scores),3), 'Precision:', round(np.mean(precision_scores),3), 
                 'Recall:', round(np.mean(recall_scores),3), '\n')
        
        y_vals = numpy.concatenate(y_vals)
        y_probs = numpy.concatenate(y_probs)
        precision, recall, _ = precision_recall_curve(y_vals, y_probs)
        fpr, tpr, _ = roc_curve(y_vals, y_probs)
        lab1 = 'Overall AUC=%.4f' % (auc(fpr,tpr))
        lab2 = 'Overall AUC=%.4f' % (auc(recall, precision))
        axes[0].step(fpr,tpr, label = lab1, lw=2, color = 'black')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend(loc='lower right', fontsize='small')
        axes[1].step(recall, precision, label=lab2, lw=2, color='black')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].legend(loc='lower left', fontsize='small')
        f.tight_layout()


# ##### Learning Curves

# In[ ]:


#Oversampling for Learning Curves
x_m_samp,y_m_samp = SMOTE().fit_resample(x_m,y_m)


# In[156]:


print(x_m_samp.shape)


# In[30]:


#Learning Curves

def data_size_response(model,trX,teX,trY,teY,score_func,prob=True,n_subsets=6):

    train_accs,test_accs = [],[]
    subset_sizes = np.exp(np.linspace(3,np.log(trX.shape[0]),n_subsets)).astype(int)

    for m in subset_sizes:
        model.fit(trX[:m],trY[:m])
        if prob:
            train_acc = score_func(trY[:m],model.predict_proba(trX[:m]))
            test_acc = score_func(teY,model.predict_proba(teX))
        else:
            train_acc = score_func(trY[:m],model.predict(trX[:m]))
            test_acc = score_func(teY,model.predict(teX))
        print('training accuracy: %.3f test accuracy: %.3f subset size: %.3f' % (train_acc,test_acc,m))
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    return subset_sizes,train_accs,test_accs

def plot_response(subset_sizes,train_accs,test_accs):

    plt.plot(subset_sizes,train_accs,lw=2)
    plt.plot(subset_sizes,test_accs,lw=2)
    plt.legend(['Training Accuracy','Test Accuracy'])
    plt.xscale('log')
    plt.xlabel('Dataset size')
    plt.ylabel('Accuracy')
    plt.title('Model response to dataset size')
    plt.show()
    
#Train-val split
x_tr, x_val, y_tr, y_val = train_test_split(x_m_samp, y_m_samp, test_size=0.2, random_state=0)


# In[40]:


#LogisticRegression - Learning Curve

model = LogisticRegression()
score_func = accuracy_score
response = data_size_response(model,x_tr,x_val,y_tr,y_val,score_func,prob=False)
plot_response(*response)


# In[39]:


#Random Forest - Learning Curve

model = RandomForestClassifier(random_state = 0)
score_func = accuracy_score
response = data_size_response(model,x_tr,x_val,y_tr,y_val,score_func,prob=False)
plot_response(*response)


# In[41]:


#KNN -  Learning Curve

model = KNeighborsClassifier()
score_func = accuracy_score
response = data_size_response(model,x_tr,x_val,y_tr,y_val,score_func,prob=False)
plot_response(*response)


# In[42]:


#NaiveBayes - Learning Curve

model = GaussianNB()
score_func = accuracy_score
response = data_size_response(model,x_tr,x_val,y_tr,y_val,score_func,prob=False)
plot_response(*response)


# ##### At the end of model building phase, LogisticRegression was selected as the best model.

# ## Selection of Dimension Reduction Method for LogisticRegression

# In[45]:


#Selection Dim Reduction Method for Logistic Regression : PCA & Feature Agglomeration

random_state=0
skf = KFold(n_splits=5, shuffle=True, random_state=random_state) 
fold_idxs = list(skf.split(x_m))

clf = LogisticRegression()

for name, feature_reducer in [
    ('pca', PCA(n_components=10)),
    ('clustering single', FeatureAgglomeration(n_clusters=10, linkage='single')), 
    ('clustering complete', FeatureAgglomeration(n_clusters=10, linkage='complete')), 
    ('clustering averag', FeatureAgglomeration(n_clusters=10, linkage='average')), 
    ('clustering ward', FeatureAgglomeration(n_clusters=10, linkage='ward')), ]:


    acc_scores = []
    auc_scores = []
    precision_scores = []
    recall_scores = []
    names = []

    for i, (train_idx, test_idx) in enumerate(fold_idxs): 
        X_train = x_m.values[train_idx]
        y_train = y_m.values[train_idx]
        X_val = x_m.values[test_idx]
        y_val = y_m.values[test_idx]
        
        #OverSampling
        smote= SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        feature_reducer.fit(X_resampled)
        X_resampled = feature_reducer.transform(X_resampled)
        X_val = feature_reducer.transform(X_val)
        
        clf.fit(X_resampled, y_resampled)
        y_prob = clf.predict_proba(X_val)
        y_pred = clf.predict(X_val)
        auc_scores.append(roc_auc_score(y_val, y_prob[:,1]))
        acc_scores.append(accuracy_score(y_val,y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val,y_pred))
        names.append(name)
               
    print(name, ':\n', 'AUROC:', round(np.mean(auc_scores),3), 'ACC:', round(np.mean(acc_scores),3), 'Precision:', round(np.mean(precision_scores),3), 
                 'Recall:', round(np.mean(recall_scores),3), '\n')
        


# In[47]:


#Selection Dim Reduction Method for Logistic Regression2 : Random Forest based Importance Features

random_state = 0 
skf = KFold(n_splits=5, shuffle=True, random_state=random_state) 
fold_idxs = list(skf.split(x_m))

acc_scores = []
auc_scores = []
precision_scores = []
recall_scores = []
y_probs = []
y_vals = []
        
f, axes = plt.subplots(1, 2, figsize=(10, 5))

for i, (train_idx, test_idx) in enumerate(fold_idxs): 
        X_train = x_m.values[train_idx]
        y_train = y_m.values[train_idx]
        X_val = x_m.values[test_idx]
        y_val = y_m.values[test_idx]

        #OverSampling
        smote= SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
        sel = SelectFromModel(RandomForestClassifier(random_state = 0, n_jobs = -1))
        sel.fit(X_resampled,y_resampled)
        X_train_rfc = sel.transform(X_resampled)
        X_val_rfc = sel.transform(X_val)
            
        model = LogisticRegression()
        model.fit(X_train_rfc, y_resampled)
        y_prob = model.predict_proba(X_val_rfc)
        y_pred = model.predict(X_val_rfc)
        y_probs.append(y_prob[:,1])
        y_vals.append(y_val)

        auc_scores.append(roc_auc_score(y_val, y_prob[:,1]))
        acc_scores.append(accuracy_score(y_val,y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        recall_scores.append(recall_score(y_val,y_pred))
            
        precision, recall, _ = precision_recall_curve(y_val, y_prob[:,1])
        fpr, tpr, _ = roc_curve(y_val, y_prob[:,1])
            
        lab1 = 'Fold %d AUC=%.4f' % (i+1, auc(fpr, tpr))
        lab2 = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
        axes[0].step(fpr,tpr,label = lab1)
        axes[1].step(recall, precision, label=lab2)
        y_vals.append(y_val)
        y_probs.append(y_prob[:,1])
               
            
print('AUROC:', round(np.mean(auc_scores),3), 'ACC:', round(np.mean(acc_scores),3), 'Precision:', round(np.mean(precision_scores),3), 
                 'Recall:', round(np.mean(recall_scores),3), '\n')

y_vals = numpy.concatenate(y_vals)
y_probs = numpy.concatenate(y_probs)
precision, recall, _ = precision_recall_curve(y_vals, y_probs)
fpr, tpr, _ = roc_curve(y_vals, y_probs)
lab1 = 'Overall AUC=%.4f' % (auc(fpr,tpr))
lab2 = 'Overall AUC=%.4f' % (auc(recall, precision))
axes[0].step(fpr,tpr, label = lab1, lw=2, color = 'black')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right', fontsize='small')
axes[1].step(recall, precision, label=lab2, lw=2, color='black')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='lower left', fontsize='small')
f.tight_layout()


# RandomForest based Importance Features method was the best for dimension reduction. 

# In[49]:


#Learning Curve with reduced data,LogisticRegression

x_tr, x_val, y_tr, y_val = train_test_split(x_m_samp, y_m_samp, test_size=0.2, random_state=0)

sel = SelectFromModel(RandomForestClassifier(random_state = 0, n_jobs = -1))
sel.fit(x_tr,y_tr)
x_train_rfc = sel.transform(x_tr)
x_val_rfc = sel.transform(x_val)

model = LogisticRegression()
score_func = accuracy_score
response = data_size_response(model,x_train_rfc,x_val_rfc,y_tr,y_val,score_func,prob=False)
plot_response(*response)


# In[99]:


#See best hyperparameters for logistic regression on reduced data (to have convergence on Learning Curve)

f, axes = plt.subplots(1, 2, figsize=(10, 5))

#Splitting
x_m_tr, x_val, y_m_tr, y_val =train_test_split(x_m, y_m,test_size = 0.2, random_state=0)

#OverSampling
smote= SMOTE()
x_resampled, y_resampled = smote.fit_resample(x_m_tr, y_m_tr)
            
sel = SelectFromModel(RandomForestClassifier(random_state = 0, n_jobs = -1))
sel.fit(x_resampled,y_resampled)
X_train_rfc = sel.transform(x_resampled)
X_val_rfc = sel.transform(x_val)
            
model = LogisticRegression()
grid = {"solver" :["sag", "saga", "lbfgs", "newton-cg"], "class_weight": ["None", "balanced"]}
gscv = GridSearchCV(estimator=model,param_grid=grid,scoring="accuracy",cv=5,verbose=0,n_jobs=-1,return_train_score=True)
gscv.fit(X_train_rfc,y_resampled)
lropt = gscv.best_estimator_
print(lropt)

y_pred = lropt.predict(X_val_rfc)
y_prob = lropt.predict_proba(X_val_rfc)

auc_score = roc_auc_score(y_val, y_prob[:,1])
acc_score = accuracy_score(y_val,y_pred)
precision_score = precision_score(y_val, y_pred)
recall_score = recall_score(y_val,y_pred)
            
precision, recall, _ = precision_recall_curve(y_val, y_prob[:,1])
fpr, tpr, _ = roc_curve(y_val, y_prob[:,1])

print('AUROC:', round(auc_score,3), 'ACC:', round(acc_score,3), 'Precision:', round(precision_score,3), 'Recall:', round(recall_score,3))          
lab1 = 'AUC=%.4f' % (auc(fpr, tpr))
lab2 = 'AUC=%.4f' % (auc(recall, precision))
axes[0].step(fpr,tpr,label = lab1, lw=2 , color = "black")
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right', fontsize='small')
axes[1].step(recall, precision, label=lab2, lw=2, color="black")
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='lower left', fontsize='small')
f.tight_layout()
            


# In[104]:


#Selection Dim Reduction Method for Logistic Regression with selected hyperparameters

random_state = 0 
skf = KFold(n_splits=5, shuffle=True, random_state=random_state) 
fold_idxs = list(skf.split(x_m))

acc_scores = []
auc_scores = []
y_probs = []
y_vals = []
        
f, axes = plt.subplots(1, 2, figsize=(10, 5))

for i, (train_idx, test_idx) in enumerate(fold_idxs): 
        X_train = x_m.values[train_idx]
        y_train = y_m.values[train_idx]
        X_val = x_m.values[test_idx]
        y_val = y_m.values[test_idx]

        #OverSampling
        smote= SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
        sel = SelectFromModel(RandomForestClassifier(random_state = 0, n_jobs = -1))
        sel.fit(X_resampled,y_resampled)
        X_train_rfc = sel.transform(X_resampled)
        X_val_rfc = sel.transform(X_val)
            
        model = LogisticRegression(C=1.0, class_weight='None', dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
        model.fit(X_train_rfc, y_resampled)
        y_prob = model.predict_proba(X_val_rfc)
        y_pred = model.predict(X_val_rfc)
        y_probs.append(y_prob[:,1])
        y_vals.append(y_val)

        auc_scores.append(roc_auc_score(y_val, y_prob[:,1]))
        acc_scores.append(accuracy_score(y_val,y_pred))
            
        precision, recall, _ = precision_recall_curve(y_val, y_prob[:,1])
        fpr, tpr, _ = roc_curve(y_val, y_prob[:,1])
            
        lab1 = 'Fold %d AUC=%.4f' % (i+1, auc(fpr, tpr))
        lab2 = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
        axes[0].step(fpr,tpr,label = lab1)
        axes[1].step(recall, precision, label=lab2)
        y_vals.append(y_val)
        y_probs.append(y_prob[:,1])
               
            
print('AUROC:', round(np.mean(auc_scores),3), 'ACC:', round(np.mean(acc_scores),3), 
                 'Recall:', round(np.mean(recall_scores),3), '\n')

y_vals = numpy.concatenate(y_vals)
y_probs = numpy.concatenate(y_probs)
precision, recall, _ = precision_recall_curve(y_vals, y_probs)
fpr, tpr, _ = roc_curve(y_vals, y_probs)
lab1 = 'Overall AUC=%.4f' % (auc(fpr,tpr))
lab2 = 'Overall AUC=%.4f' % (auc(recall, precision))
axes[0].step(fpr,tpr, label = lab1, lw=2, color = 'black')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right', fontsize='small')
axes[1].step(recall, precision, label=lab2, lw=2, color='black')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='lower left', fontsize='small')
f.tight_layout()


# In[105]:


#Learning Curve with reduced data and selected parameters,LogisticRegression

x_tr, x_val, y_tr, y_val = train_test_split(x_m_samp, y_m_samp, test_size=0.2, random_state=0)

sel = SelectFromModel(RandomForestClassifier(random_state = 0, n_jobs = -1))
sel.fit(x_tr,y_tr)
x_train_rfc = sel.transform(x_tr)
x_val_rfc = sel.transform(x_val)

model = LogisticRegression(C=1.0, class_weight='None', dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
score_func = accuracy_score
response = data_size_response(model,x_train_rfc,x_val_rfc,y_tr,y_val,score_func,prob=False)
plot_response(*response)


# Since convergence was not obtained with hyperparameter tuning, we decided to not applying dimension reduction.

# ## FİNAL EVALUATION

# Based on the results above, Logistic Regression is the best model. We will get results for the test data.

# In[51]:


#Check size of the test data

print(x_test.shape)
print(y_test.shape)


# In[52]:


#Check the proportion of classes in test data

y_test_df = y_test.to_frame()

print(y_test_df['class'].value_counts())
sns.countplot(y_test_df['class'], label="count")


# In[70]:


# Make predictions on test dataset

model = LogisticRegression()
model.fit(x_m_samp, y_m_samp)
pred = model.predict(x_test)
prob = model.predict_proba(x_test)

#Evaluate predictions - Logistic Regression
print('AUROC:', round(roc_auc_score(y_test, prob[:,1]), 3),'\n',
      'Balanced Accuracy:', round(balanced_accuracy_score(y_test, pred),3),'\n',
      'Accuracy:', round(accuracy_score(y_test, pred),3),'\n',
      'Precision:', round(precision_score(y_test,pred), 3),'\n',
      'recall:' , round(recall_score(y_test, pred),3),'\n')


print(classification_report(y_test, pred))


# In[66]:


#Confusion Matrix

labels = [0, 1]
cm = confusion_matrix(y_test,pred, labels)
print(cm)


fig, ax = plot_confusion_matrix(cm)
plt.show()


# In[57]:


#Roc Curve

# calculate roc curve for model
fpr, tpr, _ = roc_curve(y_test, prob[:,1])
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='Logistic Regression')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('ROC Curve')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# In[56]:


#Precision-Recall Curve

precision, recall, thresholds = precision_recall_curve(y_test, prob[:,1])
# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='Logistic Regression')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.title('Precision-Recall Curve')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

#Check different thresholds
plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])

