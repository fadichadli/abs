import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="IA Pronos Football", page_icon="⚽", layout="centered")
st.title("⚽ Application de Pronostics Foot par IA")

@st.cache_data
def charger_donnees():
    df = pd.read_csv("data_ia.csv")
    df['Total_Goals'] = df['FTHG'] + df['FTAG']
    df['Over2.5'] = np.where(df['Total_Goals'] > 2.5, 1, 0)
    return df

try:
    df = charger_donnees()
    encoder_home = LabelEncoder()
    encoder_away = LabelEncoder()
    df['HomeTeam_Encoded'] = encoder_home.fit_transform(df['HomeTeam'])
    df['AwayTeam_Encoded'] = encoder_away.fit_transform(df['AwayTeam'])
    X = df[['HomeTeam_Encoded', 'AwayTeam_Encoded']]
    
    model_1X2 = RandomForestClassifier(n_estimators=200, random_state=42)
    model_1X2.fit(X, df['FTR'])
    model_goals = RandomForestClassifier(n_estimators=200, random_state=42)
    model_goals.fit(X, df['Over2.5'])
    liste_equipes = sorted(list(encoder_home.classes_))

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Équipe à domicile", liste_equipes)
    with col2:
        away_team = st.selectbox("✈️ Équipe à l'extérieur", liste_equipes)

    if st.button("📊 CALCULER LE PRONOSTIC", use_container_width=True):
        if home_team == away_team:
            st.warning("⚠️ Veuillez choisir deux équipes différentes !")
        else:
            home_encoded = encoder_home.transform([home_team])[0]
            away_encoded = encoder_away.transform([away_team])[0]
            X_match = np.array([[home_encoded, away_encoded]])
            probabilites = model_1X2.predict_proba(X_match)[0]
            proba_dict = dict(zip(model_1X2.classes_, probabilites))
            proba_over = model_goals.predict_proba(X_match)[0][1] * 100

            st.subheader(f"📊 Analyses : {home_team} VS {away_team}")
            st.write(f"🏠 Victoire {home_team} : {proba_dict.get('H', 0)*100:.1f}%")
            st.write(f"🤝 Match Nul : {proba_dict.get('D', 0)*100:.1f}%")
            st.write(f"✈️ Victoire {away_team} : {proba_dict.get('A', 0)*100:.1f}%")
            st.write(f"⚽ Plus de 2.5 buts : {proba_over:.1f}%")
except FileNotFoundError:
    st.error("Fichier de données manquant.")
