# Bitcoin LSTM — Web App / تطبيق الويب

A config-driven Streamlit app that loads your trained LSTM **package** (model + scalers + config)
and shows the latest Bitcoin price, the next-day forecast, a signal, and a chart.
You can swap the model later **without editing the app**.

تطبيق Streamlit مبني على ملف إعدادات، يحمّل **حزمة** نموذجك (النموذج + المحجّمات + الإعدادات)
ويعرض السعر الحالي وتوقع اليوم التالي وإشارة ورسمًا بيانيًا. تقدر تبدّل النموذج لاحقًا **بدون تعديل الموقع**.

```
btc_app/
├── app.py             # the website / الموقع
├── features.py        # shared feature builder / بناء الخصائص (نفسه في التدريب)
├── requirements.txt
├── export_for_web.py  # paste into your notebook to export the package
└── models/
    ├── config.json    # points to the active model / يشير للنموذج الفعّال
    ├── model_v1.keras
    ├── fscaler_v1.pkl
    └── tscaler_v1.pkl
```

---

## Step 1 — Export the model package (in Colab)
افتح نوتبوك التدريب، وبعد تدريب النموذج الصق محتوى `export_for_web.py` كآخر خلية وشغّلها.
It creates `model_v1.keras`, `fscaler_v1.pkl`, `tscaler_v1.pkl`, `config.json`.

## Step 2 — Put the files in `models/`
انسخ الأربعة ملفات إلى مجلد `models/` داخل مشروع الموقع.

## Step 3 — Run locally / التشغيل محليًا
```bash
cd btc_app
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501 . يفتح على المتصفح محليًا.

## Step 4 — Deploy free (Streamlit Community Cloud) / النشر المجاني
1. Push this folder to a **GitHub** repo (include `models/`).
   - .keras + .pkl for a small LSTM are only a few MB — fine for GitHub (limit 100 MB/file).
2. Go to **share.streamlit.io** → sign in with GitHub → **New app**.
3. Pick your repo, branch `main`, main file `app.py` → **Deploy**.
4. You get a public URL to share with your supervisor.

Alternative / بديل: **Hugging Face Spaces** → New Space → SDK = *Streamlit* → upload the same files.

---

## Updating the model later / تحديث النموذج لاحقًا  ⭐
This is the whole point of the design.

- **Same features / look-back / outputs** (just retrained weights or new architecture):
  1. Re-run `export_for_web.py` with `VERSION = "v2"`.
  2. Copy the new files into `models/`.
  3. Edit `config.json` so `model_file`, `feature_scaler`, `target_scaler`, `version` point to v2.
  4. `git push` → the site auto-redeploys. **No app.py change.**

- **Changed the interface** (new features, different window, or switched
  regression ↔ classification): update `config.json` (features / look_back / task).
  Because the app reads these from config, most changes need **no code edit**. Only if you
  add a brand-new feature that isn't in `features.py`, add it there too (same file is used
  by training and the app, so they stay in sync).

بالعربي باختصار: بدّل الملفات في `models/` وحدّث `config.json`، ثم ادفع التغييرات.
طالما العقد (المدخلات/المخرجات) ثابت، الموقع ما يحتاج أي تعديل.

---

## Advanced — swap the model without redeploying / تبديل بدون إعادة نشر
Host the model files on Google Drive / a URL and have `app.py` download them at startup.
Then updating = replace the file at that URL; just restart the app. (Ask and I'll wire this in.)

> Educational thesis demo. Not financial advice. / عرض أكاديمي، ليس نصيحة مالية.
