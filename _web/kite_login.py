"""
Daily Zerodha Kite Connect login → saves the access token for the dashboard.

Run once each trading day:
    python kite_login.py

It prints a login URL. Open it, log in to Zerodha, approve, and you'll be
redirected to a URL containing `request_token=...`. Paste that value back here.
The script exchanges it for an access_token and saves it to kite_token.json.

You enter your Zerodha credentials only on Zerodha's own login page — this
script never sees your password.

Requires:  pip install kiteconnect
Reads api_key / api_secret from .streamlit/secrets.toml ([kite]) or env vars
KITE_API_KEY / KITE_API_SECRET.
"""

import os
import sys

from kiteconnect import KiteConnect
import kite_helper


def read_secrets():
    api_key = os.environ.get("KITE_API_KEY")
    api_secret = os.environ.get("KITE_API_SECRET")
    path = os.path.join(".streamlit", "secrets.toml")
    if (not api_key or not api_secret) and os.path.exists(path):
        try:
            import tomllib
            with open(path, "rb") as f:
                data = tomllib.load(f)
            k = data.get("kite", {})
            api_key = api_key or k.get("api_key")
            api_secret = api_secret or k.get("api_secret")
        except Exception:
            pass
    return api_key, api_secret


def main():
    api_key, api_secret = read_secrets()
    if not api_key or not api_secret:
        print("Missing api_key / api_secret. Fill .streamlit/secrets.toml [kite] "
              "or set KITE_API_KEY / KITE_API_SECRET.")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    print("\n1) Open this URL, log in to Zerodha and approve:\n")
    print("   " + kite.login_url() + "\n")
    print("2) After approving you'll be redirected to a URL like:")
    print("   http://127.0.0.1/?request_token=XXXXXX&action=login&status=success\n")
    request_token = input("3) Paste the request_token value here: ").strip()

    try:
        sess = kite.generate_session(request_token, api_secret=api_secret)
        token = sess["access_token"]
    except Exception as e:
        print(f"Login failed: {e}")
        sys.exit(1)

    kite_helper.save_access_token(token)
    print(f"\n✅ Access token saved to {kite_helper.TOKEN_FILE}. "
          "The dashboard's Zerodha tab will now connect (valid until ~6am tomorrow).")


if __name__ == "__main__":
    main()
