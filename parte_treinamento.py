import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, classification_report, RocCurveDisplay, ConfusionMatrixDisplay, make_scorer, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# pre processamento
df = pd.read_csv('cardio_train.csv', sep=';')
df.rename(columns={'cardio': 'disease'}, inplace=True)
df.drop('id', axis=1, inplace=True)
df['age'] = (df['age'] / 365).astype(int)
df_original = df.copy() 

# Mantıksız kan basıncı değerlerini filtreleme işlemi:
# ap_hi'nin ap_lo'dan büyük olması, ap_hi'nin 70-250 arasında olması ve ap_lo'nun 40-200 arasında olması gereklidir.
print(df.shape)
df = df[
    (df['ap_hi'] > df['ap_lo']) &
    (df['ap_hi'] >= 70) & (df['ap_hi'] <= 250) &
    (df['ap_lo'] >= 40) & (df['ap_lo'] <= 200)
]

# regrinha de fisiologia afs
df = df[df['ap_hi'] > df['ap_lo']]
print(f"\nTamanho após aplicar regra 'ap_hi > ap_lo': {df.shape}")

print(df.shape)

# comeca aqui
df_2 = df.copy()

X = df_2.drop("disease", axis= 1)
y = df_2["disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Para TREINO, temos {X_train.shape[0]} amostras COM rótulo.")
print(f"Para TESTE, temos {X_test.shape[0]} amostras SEM rótulo.")


#pca
pca = PCA(n_components=0.95, whiten=True, random_state=42)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Número de componentes após PCA: {pca.n_components_}")

# PIPELINE FINAL -> mudar aqui
#{'mlp__activation': 'logistic', 'mlp__batch_size': 128, 'mlp__hidden_layer_sizes': (24, 64, 32), 'mlp__learning_rate': 'constant', 'mlp__learning_rate_init': 0.1}
pipeline_MLP_melhor = Pipeline([
     ('scaler', StandardScaler()),
     ('pca', PCA(n_components=0.95, whiten=True, random_state=42)),
     ('mlp', MLPClassifier(
         activation='logistic',
         batch_size=128,
         hidden_layer_sizes=(24, 64, 32),
         learning_rate='constant',
         learning_rate_init=0.1,
         random_state=42,
         max_iter=500
     ))
 ])


# FINAL
pipeline_MLP_melhor.fit(X_train, y_train)
y_pred_test = pipeline_MLP_melhor.predict(X_test)
y_proba_test = pipeline_MLP_melhor.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred_test)
acc = accuracy_score(y_test, y_pred_test)
print(f"Acurácia: {acc*100:.2f}%")
prec = precision_score(y_test, y_pred_test)
print(f"Precisão: {prec*100:.2f}%")
rec = recall_score(y_test, y_pred_test)
print(f"Recall: {rec*100:.2f}%")

