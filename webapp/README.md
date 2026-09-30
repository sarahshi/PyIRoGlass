# PyIRoGlass web app

Browser front end for PyIRoGlass, with two tools:

- **Wafer thickness** from reflectance FTIR spectra (olivine, pyroxenes, basaltic glass, or any refractive index). Instant.
- **H₂O and CO₂** from transmission FTIR spectra (beta), for up to 5 spectra at a time. Each spectrum runs the full MCMC baseline fit (about 20 s on a laptop, roughly a minute on the free web server), and only one fit runs at a time. For larger datasets, use VICTOR, Google Colab, or a local install.

## Run locally

```
pip install -r webapp/requirements.txt
streamlit run webapp/streamlit_app.py
```

## Deploy (free)

1. Push to GitHub.
2. Go to https://share.streamlit.io, sign in with GitHub, and create an app.
3. Repository `sarahshi/PyIRoGlass`, branch `main`, main file path `webapp/streamlit_app.py`.

The example spectra come from `Inputs/COLAB_BINDER` in this repository. The app installs PyIRoGlass from PyPI (pinned in `requirements.txt`); after a new release, bump the pin and push.
