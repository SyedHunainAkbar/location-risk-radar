# Deploying to Streamlit Community Cloud

We deploy `app/streamlit_app.py` from the GitHub repo
`SyedHunainAkbar/location-risk-radar`. The app reads only precomputed artifacts and
runs in Cached mode with no keys, so it is safe to deploy publicly before adding any
provider key.

## Exact clicks on share.streamlit.io

1. Go to https://share.streamlit.io and sign in with the GitHub account that owns
   (or can access) the repository.
2. Click **Create app** (top right), then choose **Deploy a public app from GitHub**.
3. In the deploy form set:
   - **Repository**: `SyedHunainAkbar/location-risk-radar`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
   - **App URL**: pick the subdomain you want (for example `location-risk-radar`).
4. Open **Advanced settings**:
   - **Python version**: select **3.11**.
   - **Secrets**: paste the TOML below (fill in a real key only if you want Live
     mode; leave blank for a safe Cached-mode deploy).
5. Click **Deploy**. The first build installs `requirements.txt` and starts the app.

### Secrets TOML to paste (Advanced settings -> Secrets)

```toml
# Leave keys empty for a Cached-mode (no-cost) public deploy.
# Add one real key to enable Live mode.
LLM_PROVIDER = "nvidia"
NVIDIA_API_KEY = ""
OPENAI_API_KEY = ""

# Abuse protection (all optional; these are the defaults).
LIVE_CALL_CAP = "25"
GATEWAY_MAX_TOKENS_PER_CALL = "1024"
GATEWAY_DAILY_BUDGET_USD = "2.0"
```

Do not set `VOYAGER_API_KEY`. Voyager requires the ASU VPN and is automatically
disabled on Streamlit Cloud; the sidebar shows it as disabled with a note.

## Post-deploy checklist

1. **Open in an incognito window** (a fresh session, no cached state).
2. Confirm the sidebar shows **Cached** mode by default and the note
   "No API key detected. Serving cached results" when no key is set.
3. **Load every page** and confirm none throws an error banner:
   - Portfolio Radar
   - Location Deep Dive
   - Risk Committee
   - Complaint Themes
   - Ask the Reviews
   - Model Lab
   - Methodology and Limitations
   (Until the pipeline artifacts are committed, pages show friendly
   "run pipeline X to produce it" notices rather than data. That is expected.)
4. If a key was set, switch the sidebar to **Live**, then:
   - Run **one Risk Committee** brief for a high-risk location and confirm it
     returns with a provider tag (for example `nvidia:meta/llama-3.3-70b-instruct`).
   - Ask **one Ask the Reviews** question and confirm a grounded answer with
     `[review_id]` citations.
   - Watch the sidebar: the live-call counter increments toward the cap and the
     daily budget bar updates.
5. **Check the app logs** (Manage app -> Logs on share.streamlit.io): confirm no
   tracebacks, no "No secrets found" errors, and no leaked key material.
6. Confirm the Voyager row reads "Voyager (disabled)" with the VPN-only note.

## Notes on data

The deployed app will render but show "artifact not available" notices until the
offline pipeline artifacts (survival scores, topics, RAG index, committee briefs)
are produced by `pipeline/01` through `pipeline/08` and committed under `artifacts/`
and `data/`. Deployment readiness (install, boot, Cached mode, abuse guards) does not
depend on those artifacts.
