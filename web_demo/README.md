# Static Portfolio Site

This folder now contains a small personal portfolio site that can be hosted
for free on GitHub Pages, Netlify, Vercel, or your personal website.

Current tabs:

- `Profile`
- `Flight Delay Demo`
- `Wearable Device` placeholder
- `CGCNN Research` placeholder

## Files

- `index.html`
- `styles.css`
- `app.js`
- `data/demo_val_2021-11-28.json`

## Rebuild The Demo Data

If you want to refresh the exported replay bundle:

```bash
python 13_export_web_demo_data.py
```

That reads:

- `evaluation/demo_cache/demo_replay_val_2021-11-28.parquet`

and writes:

- `web_demo/data/demo_val_2021-11-28.json`

## Run Locally

Serve the folder with a simple static file server:

```bash
cd web_demo
python -m http.server 8080
```

Then open:

```text
http://127.0.0.1:8080
```

Windows-safe alternative from anywhere:

```powershell
python -m http.server 8080 --directory "C:\Users\user\Desktop\Airline_Graphs_Project\web_demo"
```

Important:

- open `http://127.0.0.1:8080/`
- do not open `http://127.0.0.1:8080/web_demo/`
- if you serve from the wrong folder, Python will return `404 File not found`

## Host For Free

### GitHub Pages

Push `web_demo/` to a GitHub repo or a `docs/` folder and enable GitHub Pages.

### Netlify / Vercel

Set the publish directory to:

```text
web_demo
```

## Notes

- The flight tab uses precomputed validation-day replay data, not live inference.
- The main profile page is based on the latest resume content.
- The wearable and CGCNN sections are intentionally placeholders for now.
- The whole site is meant for portfolio/demo use and costs `$0` to host.
- The current bundled day is `2021-11-28` from the validation split.
