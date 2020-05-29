import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# veri kumesini oku
kolon_adlari = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
'dst_host_srv_rerror_rate','label']

verikumesi = pd.read_csv("kddcup99.tar.gz",compression="gzip", names=kolon_adlari, 
low_memory=False, skiprows=1)


secilecek_kolonlar = ['serror_rate','srv_serror_rate',
'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
'dst_host_srv_rerror_rate']


X = verikumesi[secilecek_kolonlar].as_matrix()
y = verikumesi['label'].apply(lambda d:0 if d == 'normal.' else 1).as_matrix()

# lambda degerleri
lambdas=[x  for x in range(1, 1000, 50)]

# agirliklar icin zero matris tanimla
coefs_ridge = np.zeros((len(lambdas),X.shape[1]))
print(coefs_ridge.shape)

i = 0
for l in lambdas:
   print(datetime.now(),i)
   clf = LogisticRegression(verbose=0, C=1/l)
   clf.fit(X,y)
   coefs_ridge[i,:] = clf.coef_
   i += 1
plt.figure(1)
ax = plt.gca()

for i in range(X.shape[1]):
    coef_l = coefs_ridge[:,i]
    print('*'*100)
    print(coef_l.shape)
    l1 = plt.plot(lambdas, coef_l, label= secilecek_kolonlar[i])
plt.legend()
plt.show()

'''
plt.figure(1)
ax = plt.gca()

colors = cycle(['b', 'r', 'g', 'c', 'k'])
neg_log_alphas_lasso = -np.log10(alphas_lasso)
for coef_l, c in zip(coefs_lasso, colors):
    l1 = plt.plot(alphas_lasso, coef_l, c=c)
	
plt.axis('tight')
plt.show()
'''