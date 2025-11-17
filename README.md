
## Overview
The **Inclusive Emotional Movie Explorer** is an interactive visualization tool that maps movies based on **emotional similarity**.  
It combines the **MovieLens dataset** with the **NRC-VAD Lexicon**, which provides Valence, Arousal, and Dominance scores for English words.

By matching movie tags with VAD values, the app computes an emotional profile for each film and visualizes similarity using a **radial network layout**.

Built using **Python, Plotly, and Dash**, the project supports ADHD- and autism-friendly design principles through simplified modes, calming gradients, and reduced visual load.

---

## Features
- Radial layout of emotionally similar movies  
- Node color = emotional **valence**  
- Node size = audience engagement (**mean rating**)  
- Hover tooltips with emotional & genre metadata  
- Click any movie to recenter the network  
- Toggle between **Simplified** and **Intense** view modes  
- NRC-VAD–driven emotional modeling  
- Fully interactive **Dash** web app  

---

## Tech Stack
- Python  
- Pandas / NumPy  
- Scikit-Learn  
- Plotly  
- Dash  
- MovieLens Dataset  
- NRC-VAD Lexicon  

---

## How to Run Locally Render

```bash
pip install -r requirements.txt
python app.py

The app will launch at:  http://127.0.0.1:----
```
---
## **Render** 

[https://inclusive-visualization-designing-data.onrender.com
](https://inclusive-visualization-designing-data.onrender.com)
