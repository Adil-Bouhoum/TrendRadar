from mastodon import Mastodon

try:
    # Remplacez par votre jeton en dur juste pour le test
    m = Mastodon(
        access_token = "VOTRE_JETON_ICI",
        api_base_url = 'https://mastodon.social'
    )
    me = m.account_verify_credentials()
    print(f"✅ Connexion réussie ! Connecté en tant que : {me['username']}")
except Exception as e:
    print(f"❌ Erreur : {e}")