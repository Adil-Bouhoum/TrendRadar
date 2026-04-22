import streamlit as st
import pandas as pd
import joblib
from mastodon import Mastodon, StreamListener
from extractor import MastodonExtractor
from bs4 import BeautifulSoup
import time

st.set_page_config(page_title="TrendRadar Live", layout="wide")
st.title("🔥 TrendRadar : Détection de Viralité en Temps Réel")

# Chargement du modèle et du seuil (0.40)
@st.cache_resource
def load_assets():
    model = joblib.load('trendradar_xgb_model.pkl')
    cfg = joblib.load('trendradar_config.pkl')
    return model, cfg

model, cfg = load_assets()
extractor = MastodonExtractor()

# Gestion du Stream
class TrendListener(StreamListener):
    def on_update(self, status):
        content = BeautifulSoup(status['content'], "html.parser").get_text().lower()
        now = time.time()
        words = [w for w in content.split() if len(w) > 4 and w.isalpha()]
        
        for word in set(words):
            extractor.add_event(word, now, 
                                status['account']['followers_count'], 
                                status['account']['acct'].split('@')[-1], 
                                status['reblogs_count'])
            
            feats = extractor.get_features(word, now)
            if feats and feats[0] > 5: # Seuil de vélocité minimal pour analyse
                df_input = pd.DataFrame([feats], columns=cfg['features'])
                proba = model.predict_proba(df_input)[0][1]
                
                if proba >= cfg['threshold']:
                    st.toast(f"Tendance détectée : {word}", icon="🚨")
                    # Logique d'affichage ici...

if st.button("Lancer le Scan Mastodon"):
    st.info("Connexion au flux public...")
    # Remplacez par votre token Mastodon
    m = Mastodon(access_token="VOTRE_TOKEN", api_base_url='https://mastodon.social')
    m.stream_public(TrendListener())