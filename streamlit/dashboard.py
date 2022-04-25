# Modules
from flask import jsonify
import requests
import json
import streamlit as st
import shap
import streamlit.components.v1 as components
import numpy as np

# Viz
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


if 'clicked' not in st.session_state:
    st.session_state['clicked'] = False

    
# Impression image (features globale)
threshold = 1 - 0.3


# Méthode pour afficher le force plot de shap
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height) 

    
#Recup top colonnes
def get_top_columns(shap_vals, f_names, num):
    l = []
    #https://github.com/slundberg/shap/issues/632
    for name in np.flip(np.argsort(np.abs(shap_vals))[-num:]):
        l.append(f_names[name])
    return l 
    
    
#Shap values de la classe 0 
expected_value = [-0.005649339905858291, 0.005649339905858291]


# Image features globales
image_url = r"C:\Users\Damien\Desktop\Data Scientist\P7\Images\dash3.png"
image_url2 = r"C:\Users\Damien\Desktop\Data Scientist\P7\Images\shap_importance.png"
#image_url = "https://banking-opc.herokuapp.com/images/dash3.png"
#image_url2 = "https://banking-opc.herokuapp.com/images/shap_importance.png"
image_logo = Image.open(image_url)
image_shap = Image.open(image_url2)
st.image(image_logo)


# 1- Affichage selectbox + Jauge
st.title('Dashboard Scoring Credit')
st.subheader("1. Prédictions de scoring client et comparaison à l'ensemble des clients")

# Récuperation ID client
client_IDs = requests.get(url="http://127.0.0.1:5000/ids")
ID_dict = client_IDs.json()
IDs = list(ID_dict.values())
id_input = st.selectbox('Selectionnez un ID client',IDs)

if st.button('Envoyez') or st.session_state.clicked: 
    st.session_state.clicked = True
    client_id = id_input
    URL = "http://127.0.0.1:5000/predict"   
    pydict = {'ID': client_id}
    jsondata = json.dumps(pydict)
    r = requests.post(url=URL, data=jsondata)  
    test = r.json()
    proba = float(r.json()["pred"])
        
    # Création jauge
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = proba,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Score"},
    gauge = { 'axis': {'range':[0,1]}}))

    if(proba > threshold):
        st.success("Le client a " + "{:.2%}".format(proba) + " de probabilité de remboursement.")
        fig.update_traces(gauge_bar_color="green")
    else:
        fig.update_traces(gauge_bar_color="red")
        st.error("Le client a " +"{:.2%}".format(1 - proba)+ " de probabilité de défaut de paiement !")

    # Affichage de la jauge
    st.write(fig)
    start = client_id
    
    
    # 2- Affichage TOP 10 variables
    st.subheader("2. Influence des variables sur le score du client– TOP 10")
    st.write("Conseil au chargé clientèle : Voici les variables ayant impacté la prédiction de solvabilité du client")
    
    # Récuperation top 10 features
    features = requests.get(url="http://127.0.0.1:5000/features")
    features_list = features.json()
    # Récuperation valeurs shap
    shap_data = requests.post(url="http://127.0.0.1:5000/shap", data=jsondata)  
    shap_list = shap_data.json()
    shap_array = np.asarray(shap_list)
    #Force plot
    st_shap(shap.plots.force(expected_value[0], shap_array, features_list))    

    
    # Affichage waterfall
    fig, ax = plt.subplots()
    shap.plots._waterfall.waterfall_legacy(expected_value[0], shap_array, feature_names=features_list)
    st.pyplot(fig, bbox_inches='tight',dpi=300,pad_inches=0)
    plt.clf()
    
    
    # 3- Importance variables tous clients
    st.subheader("3. Importance  des variables tout client confondu")
    st.write("Conseil au chargé clientèle : il peut être intéressant d'expliquer au client, quelles sont les variables qui influencent le plus pour l'obtention d'un prêt")
    
    # Affichage image
    st.image(image_shap, caption='L\'importance de chaque caractéristique dans la décision')

    
    # 4- Situation client sur top 10 des features
    st.subheader("4. Situation du client sur les 10 principales variables")
    st.text("Le rond bleu représente la valeur du client sélectionné")
    
    # Récupérer les top 10 colonnes
    top_cols = get_top_columns(shap_array, features_list, 7)
    top_cols_json = json.dumps(top_cols) 
    
    # Récupérer df avec top 10 colonnes
    df = requests.post(url="http://127.0.0.1:5000/dataframe", data=top_cols_json)  
    df_test = df.json()['data']    
    dframe = pd.read_json(df_test)
    
    # Récupérer df du client avec top 10 colonnes
    df_user_json = requests.post(url="http://127.0.0.1:5000/dataframeclient", data=jsondata)  
    df_user_dict = df_user_json.json()['dataUser']    
    df_user = pd.read_json(df_user_dict)
    
    # Affichage des figures
    fig, ax = plt.subplots()
    ax.set_ylim([0, 2])
    ax.set_xlim([0, 7])
    fig.set_figwidth(30)
    fig.set_figheight(20)
    sns.boxplot(data=dframe)
    sns.stripplot(data=df_user[top_cols], color='blue',linewidth=1, size=20)
    st.pyplot(fig, bbox_inches='tight',dpi=300,pad_inches=0)
    plt.clf()

    
    # 5- Viz variables selectionnées
    st.subheader("5. Visualisation d'une ou plusieurs variables sélectionnées")

    # Récupérer df
    df_json = requests.get(url="http://127.0.0.1:5000/df")
    df_dict = df_json.json()['df_graph']    
    df_tout = pd.read_json(df_dict)   
    
    # Liste déroulante
    Variable = st.selectbox("feature_names 1:", features_list) 
    fig, ax = plt.subplots()
    histplot = sns.histplot(data=df_tout, x=Variable, hue="target")
    histplot.axvline(float(df_user[Variable]), color='red')
    st.pyplot(fig, bbox_inches='tight',dpi=300,pad_inches=0)
    plt.clf()

    # Liste déroulante 2
    Variable2 = st.selectbox("feature_names 2:", features_list) 
    fig, ax = plt.subplots()
    histplot2 = sns.histplot(data=df_tout, x=Variable2, hue="target")
    histplot2.axvline(float(df_user[Variable2]), color='red')
    st.pyplot(fig, bbox_inches='tight',dpi=300,pad_inches=0)
    plt.clf()