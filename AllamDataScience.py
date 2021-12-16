import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import joblib
import itertools
import cv2
from datasist.structdata import detect_outliers
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import  init_notebook_mode,  iplot
init_notebook_mode(connected=True)
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    cm = np.round(cm, 2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def MLPredictAcc(X, y, classes , scale = False , smote = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if (smote == True):
        sampler = SMOTE()
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    if (scale == True) :
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    models = {
        "XGB": XGBClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "GaussianNB" : GaussianNB(),
        "Perceptron" : Perceptron(),
        "LinearSVC" : LinearSVC(),
        "SGDClassifier" : SGDClassifier(),
        "LogisticRegression" : LogisticRegression()
    }
    modell = []
    modell_acc = []
    model_built = {}
    for name, model in models.items():
        print(f'Training Model {name} \n--------------')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cf = confusion_matrix(y_test, y_pred)
        acc_svc = round(accuracy_score(y_test, y_pred) * 100,2)
        modell.append(name)
        modell_acc.append(acc_svc)
        model_built[name]=model
        plot_confusion_matrix(cf, classes, title='{} cf with acc = {} %'.format(name,acc_svc))
        print('-' * 30)
    models = pd.DataFrame(
        {
            'Model': modell,
            'Score': modell_acc ,

        })
    models = models.sort_values(by='Score', ascending=False)
    models['Score'] = models['Score'].apply(lambda x : str(x) + " %")
    modelss = pd.DataFrame({
        "index ": [p for p in range(1,len(modell_acc)+1)],
         "model" : models['Model'],
         'Score': models['Score'],
    })

    if (scale == True):
        return modelss, model_built , scaler
    else:
        return modelss, model_built
def check_category_classes(df):
    return df.select_dtypes(include='O').columns.to_list()
def check_non_category_classes(df):
    return df.select_dtypes(exclude='O').columns.to_list()
def define_column_type(df):
    numerical_column =check_non_category_classes(df)
    categorical_column = check_category_classes(df)
    print("numerical_column", numerical_column)
    print("categorical_column", categorical_column)
    return numerical_column , categorical_column
def show_value_count_category_column(df , categorical_column):
    for name in categorical_column:
        df_count = pd.DataFrame(df[name].value_counts())
        print(df_count)
        print("*" * 50)
def allam_visualize_null_count(df):
    plt.figure(figsize=(12,8))
    print(df.isnull().sum())
    sns.heatmap(df.isnull())
def allam_plot_graph(x,y,xlabel="xlabel",ylabel="ylabel" , title = "plot grapg"):
    plt.figure(figsize=(12, 8))
    plt.style.use('ggplot')
    ax = plt.axes()
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title)
    plt.plot(x,y, color='blue', marker='o', markersize=11)
def allam_pie_graph(labels , values , title = None):
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                 insidetextorientation='radial' , title=title
                                 )] )

    fig.show()
def categoryColumnCountAndPercentage(df , categorical_column):
    for i in categorical_column:
        print("count and percentage of column {}".format(i))
        f = df[i].value_counts()
        allam_pie_graph(f.index.tolist(), f.tolist())
        print("*" * 100)
def categoryColumnMostAccurance(df,categorical_column):
    for i in categorical_column:
        print("most accurance of column {}".format(i))
        f = df[i].value_counts()
        allam_display_most_accurance(f.index.tolist(), f.tolist())
        print("*" * 100)
def found_text(m_purpose):
    bb=''
    for i in range(len(m_purpose)):
        if i <=len(m_purpose) -1 :
            bb += str(m_purpose[i]) + '% ,'
    bb = bb[0:-1]
    return bb.split(',')
def allam_compression_2_class(purp ,c1_purpose ,c2_purpose , c1_name= "type 1 " , c2_name = "type2" , title ="compression" , X_axis_name ="X" , Y_axis_name = "Y"):
    c1_txt = found_text(c1_purpose)
    c2_txt = found_text(c2_purpose)
    c1_pur = go.Bar(
        x=purp,
        y=c1_purpose,
        name=c1_name,
        text=c1_txt,
        textposition='auto',
    )
    c2_pur = go.Bar(
        x=purp,
        y=c2_purpose,
        name=c2_name,
        text=c2_txt,
        textposition='auto',
    )
    data = [c1_pur, c2_pur]
    layout = dict(
        title=title,
        xaxis=dict(title=X_axis_name),
        yaxis=dict(title=Y_axis_name)
    )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='grouped-bar-direct-labels')
def allam_display_most_accurance(things , values , title = "allam display most accurance"):
    data = [
        go.Scatterpolar(
            r=values,
            theta=things,
            fill='toself',
        )
    ]
    layout = go.Layout(
        title=title
    )
    fig = dict(data=data, layout=layout)
    iplot(fig)
def allam_bar_graph(purp ,c1_purpose , c1_name= "type 1 ", title ="compression" , X_axis_name ="X" , Y_axis_name = "Y"):
    c1_txt = found_text(c1_purpose)
    c1_pur = go.Bar(
        x=purp,
        y=c1_purpose,
        name=c1_name,
        text=c1_txt,
        textposition='auto',
    )
    data = [c1_pur]
    layout = dict(
        title=title,
        xaxis=dict(title=X_axis_name),
        yaxis=dict(title=Y_axis_name)
    )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='grouped-bar-direct-labels')
def allam_create_table(df ):
    colm = list(df.columns)
    ListOfColumnName = []
    for i in colm : ListOfColumnName.append(df[i])
    fig = go.Figure(data=[go.Table(
        header=dict(values= colm ,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=ListOfColumnName,
                   fill_color='lavender',
                   align='left'))
    ])
    fig.show()
def allam_sunburst_charts(df,path,values,color):
    fig = px.sunburst(df, path=path, values=values, color=color)
    fig.show()
def allam_histogram(df,x):
    fig = px.histogram(df, x=x)
    fig.show()
def allam_histogram_with_rug(df,x,color=None,):
    fig = px.histogram(df, x=x, color=color, marginal="rug",  hover_data=df.columns)
    fig.show()
def allam_histogram_with_violin(df,x,color=None,):
    fig = px.histogram(df, x=x, color=color, marginal="violin",  hover_data=df.columns)
    fig.show()
def allam_histogram_with_box(df,x,color=None,):
    fig = px.histogram(df, x=x, color=color, marginal="box", hover_data=df.columns)
    fig.show()
def allam_box_plot(df,x,y,color=None):
    fig = px.box(df, x=x, y=y, color=color)
    fig.update_traces(quartilemethod="exclusive")
    fig.show()
def allam_violin_plot(df,x,y,color=None):
    fig = px.violin(df, x=x, y=y, color=color, box=True, points="all", hover_data=df.columns)
    fig.show()
def allam_scatter_plot(df , x,y,color=None,size=None):
    fig = px.scatter(df, x=x, y=y, color=color, size=size)
    fig.show()
def allam_commpression_pointplot(df,x,y,clas,row=None,col=None):
    grid = sns.FacetGrid(df, row=row ,col=col , size=2.2, aspect=1.6)
    grid.map(sns.pointplot, x, y, clas, palette='deep')
    grid.add_legend()
def allam_commpression_barplot(df,x,y,row=None,col=None):
    grid = sns.FacetGrid(df, row=row, col=col)
    grid.map(sns.barplot, x, y)
    grid.add_legend()
def allam_commpression_histplot(df,x,row=None,col=None):
    grid = sns.FacetGrid(df, row=row, col=col, size=2.2, aspect=1.6)
    grid.map(plt.hist, x, alpha=.5, bins=20)
    grid.add_legend()
def allam_visualize_date(df,dateColumn , NumericalColumn):
    fig = px.line(df, x=dateColumn, y=NumericalColumn)
    fig.show()
def allam_visualize_corr(corr):
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt='0.2f')
def convert_pilimg_to_cv2img(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
def detect_outlier(df,numerical):
    for col in numerical:
        outliers = detect_outliers(df, 0, [col])
        df.drop(outliers, inplace=True)
        print("len outliner in {} = {}".format(col,len(outliers)) )
def make_encoding_dict(df):
    return dict(tuple(zip(df.value_counts().index.tolist(), [i for i in range (100)])))
def load_modelWithScaler(model_path,scaler_path ,data ,returnName=False,dictionary = None):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    dictionary = dictionary
    data = data
    prediction = model.predict(scaler.transform([data]))
    if (returnName == True):
        for name, age in dictionary.items():
            if age == prediction:
                return name
    else:
        return prediction
def imuter_nan(df,coll):
    imputer = KNNImputer()
    for col in coll:
        df[col] = imputer.fit_transform(df[[col]])