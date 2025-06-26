# LumenALYZE

LumenALYZE is a full-stack machine learning platform built with FastAPI backend and Next.js TypeScript frontend. It provides a complete workflow for data analysis, from CSV file upload and preprocessing (handling missing values, normalization, outlier removal) to three core machine learning tasks: prediction (Random Forest and MLP), anomaly detection (Isolation Forest), and segmentation (K-Means clustering). The backend is modularly organized with config, models, services, and utils folders, while the frontend features interactive visualizations using Plotly.js, a results dashboard, and export to CSV/JSON/Excel. The app uses a custom color scheme (#014F39, #FBF7C7, #E8E8E6, #120F0A), Montserrat and Lora typography, comprehensive error handling, TypeScript type safety, and responsive design.

---

## How To Use

1. **Upload Your Data:**  
   - Click on the upload area or drag and drop your CSV/Excel file.
   - Supported formats: `.csv`, `.xlsx`, `.xls`.
   - Max file size: 50MB.
   - Ensure your file has at least one numeric column for analysis.

2. **Preview Your Data:**  
   - Review the data preview table for accuracy.
   - Verify column names and data types.
   - Check the number of rows and columns detected.

3. **Choose Analysis Type:**  
   - **Prediction:** Forecast values or classify data points.
   - **Anomaly Detection:** Identify unusual patterns or outliers.
   - **Segmentation:** Group similar data points into clusters.

4. **Configure Model (Optional):**  
   - Adjust model parameters as needed.
   - For anomaly detection, set the expected percentage of anomalies.

5. **View Results:**  
   - Results are shown in interactive charts and dashboards.
   - Export results to CSV, JSON, or Excel.

---

## Programming Languages & Technologies

- **Python** (Backend: FastAPI, scikit-learn, pandas, plotly, etc.)
- **TypeScript** (Frontend: Next.js, React, Plotly.js)
- **HTML/CSS** (UI styling and layout)

---

## Project Structure

### Backend

- **Entry Point:** `main.py`
- **Folder Structure:**
  - `config/`: Application settings and chart config
  - `models/`: Preprocessing, prediction, anomaly detection, segmentation, visualization
  - `services/`: Analytics, dashboard, report generation
  - `utils/`: File upload, JSON serialization, chart data formatting, export, logging
- **Dependencies:** See `requirements.txt`

### Frontend

- **Entry Point:** `src/app/page.tsx`
- **Styling:** `globals.css`, custom color scheme, Montserrat and Lora fonts
- **Components:**
  - `charts/`: Prediction, anomaly, cluster, and metrics charts
  - `dashboard/`: Results dashboard, export button, statistics panel
  - `visualization/`: Chart container, data visualization
- **Configuration:** `next.config.ts`, `package.json`, `tsconfig.json`

---

## Development Note

This project was developed by myself and assisted by Perplexity with the Sonet Claude 4.0 model and GitHub Copilot with the GPT-4.1 model.

---

## About

LumenALYZE is designed to make data analysis easy and accessible to everyone. With an intuitive interface and powerful algorithms, you can analyze data without in-depth programming skills. It features interactive charts, dashboards, and a user-friendly workflow.

> **Prediction, Anomaly Detection, and Segmentation**  
> **Visualization:** Interactive charts and dashboards  
> **User-Friendly:** No coding required – upload data and get insights in minutes