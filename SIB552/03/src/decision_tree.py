import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score 

from sklearn.tree import export_graphviz
import graphviz 

import seaborn as sns

# veri kumesini oku
kolon_adlari = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
'dst_host_srv_rerror_rate','label']

verikumesi = pd.read_csv("../../02-multi-variate-reg/src/kddcup99.tar.gz",compression="gzip", names=kolon_adlari, 
low_memory=False, skiprows=1)

# ilgili kolonlari sec
secilecek_kolonlar = ['protocol_type','service','flag','land','logged_in','is_host_login','is_guest_login']

verikumesi['label'] = verikumesi['label'].apply(lambda d:0 if d == 'normal.' else 1)
cross_tab = pd.crosstab(verikumesi['protocol_type'], verikumesi['label'])
print(cross_tab.to_latex())

lb = LabelEncoder() 
verikumesi['protocol_type'] = lb.fit_transform(verikumesi['protocol_type']) 
verikumesi['service'] = lb.fit_transform(verikumesi['service']) 
verikumesi['flag'] = lb.fit_transform(verikumesi['flag']) 
verikumesi['land'] = lb.fit_transform(verikumesi['land']) 
verikumesi['logged_in'] = lb.fit_transform(verikumesi['logged_in']) 
verikumesi['is_host_login'] = lb.fit_transform(verikumesi['is_host_login']) 
verikumesi['is_guest_login'] = lb.fit_transform(verikumesi['is_guest_login']) 

X = verikumesi[secilecek_kolonlar].as_matrix()
y = verikumesi['label'].apply(lambda d:0 if d == 'normal.' else 1).as_matrix()


clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)

y_pred = clf.predict(X)
print("Accuracy is :{0}".format(accuracy_score(y,y_pred) * 100))

dot_data = export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 

graph
