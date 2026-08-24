# ☀️ Predictive Energy Informatics & Grid Balancing Dashboard

A production-ready, data-driven web application designed to forecast real-time solar grid availability for local distribution nodes in Kenya. This project serves as an end-to-end demonstration of leveraging Data Science and Machine Learning to optimize clean energy systems and mitigate climate instabilities.

👉 **[https://solar-power-forecaster-lewhamjmctznp7ifz9w65p.streamlit.app/])*

---

## 📈 System Architecture & Methodology

This platform bridges classical machine learning infrastructure with environmental telemetry through a robust data pipeline:

1. **Historical Telemetry Streaming:** Connects dynamically to the **NASA POWER API** to harvest 90 days of localized physical meteorological matrices (Surface Shortwave Downward Irradiance, Temperature at 2 Meters, and Wind Speed at 2 Meters).
2. **Predictive Engine Training:** Compiles a localized **Random Forest Regressor** ensemble model achieving up to a **99.05% out-of-sample explanatory variance (\(R^2\) background validation)** using cyclical temporal feature configurations.
3. **Live Predictive Forecasting:** Connects to the **Open-Meteo API** to ingest real-time future atmospheric forecasts for the chosen node.
4. **Fault-Tolerant Fallback Logic:** Implements an automated local sinusoidal backup model to simulate diurnal temperature variations if external API handshakes encounter server-side timeouts.

---

## 🚀 Technical Core & Dependencies

The backend architecture is built completely in Python, optimized for light deployment overhead on cloud-native spaces.

### File Structure
```text
├── app.py               # Main Streamlit web application & prediction logic
├── requirements.txt     # Cloud server production dependencies
└── README.md            # Technical documentation
```

### Dependencies (`requirements.txt`)
```text
streamlit
pandas
numpy
nasa-power
scikit-learn
matplotlib
requests
```

---

## 🛠️ How to Run Locally

To test the application infrastructure on your local development machine:

1. Clone this repository to your environment:
   ```bash
   git clone https://github.com
   cd YOUR_REPO_NAME
   ```

2. Install the production dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize the Streamlit local instance:
   ```bash
   streamlit run app.py
   ```

---

## 🎓 Academic Research Applications

This architecture is optimized to support academic research tracks at the intersection of **Data Science, Sustainable Energy, and Climate Policy**:
* **Smart Grid Infrastructure:** Providing predictive load-shedding models to prevent grid-destabilization during cloud-cover interruptions.
* **Climate Informatics:** Mapping macro-environmental changes onto localized micro-generation facilities.
* **Data Governance Frameworks:** Demonstrating an analytical sandbox using anonymized spatial data compliant with Kenya's Data Protection Act, 2019.
