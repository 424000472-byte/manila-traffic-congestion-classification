# Multi-Road Traffic Congestion Classification in Metro Manila Using Vehicle Volume and Travel Time

**Course:** Data Mining (Final Project)  
**Program & Section:** Bachelor of Science in Information Technology (BSIT 3.4)  
**Institution:** National Teachers College — Manila  

---

## Overview

This project is an end-to-end data mining and machine learning pipeline designed to classify real-time traffic congestion (`1 = Congested`, `0 = Clear`) across **10 major Metro Manila corridors**. Using live road segment telemetry from the **TomTom Traffic Flow API**, the system estimates hourly vehicle volume via macroscopic speed-density relationships, preprocesses raw multi-road records, trains a **Random Forest Classifier**, and deploys real-time congestion predictions on an interactive **Streamlit** web dashboard.

---

## Monitored Metro Manila Corridors

1. **España Blvd**
2. **Quezon Blvd (Quiapo)**
3. **Recto Ave**
4. **Taft Ave (Ermita)**
5. **Roxas Blvd (Manila)**
6. **Rizal Ave (Sta. Cruz)**
7. **Quezon Ave**
8. **EDSA (Ortigas)**
9. **EDSA (Guadalupe)**
10. **C5 (Bagong Ilog)**

---

## Repository Structure

```text
manila-traffic-congestion-classification/
├── config.example.py           # Template for API key, road coordinates, and volume formula
├── collect_manila_traffic.py   # Polls TomTom API every 2 mins to build manila_traffic_raw.csv
├── train_model.py              # Cleans data, trains Random Forest model, and saves graphs
├── app.py                      # Streamlit live Manila traffic monitor and evaluation dashboard
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Excludes venv/ and private config.py from version control
└── README.md                   # Project documentation
