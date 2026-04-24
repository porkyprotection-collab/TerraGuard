# TerraGuard - Soil Erosion Detection

TerraGuard is an AI-powered Streamlit application for SDG 15 (Life on Land). It analyzes land and soil images, detects erosion risk, and delivers localized remediation guidance with a polished interactive dashboard.

## Features

- Multi-image upload support (1-5 images per analysis)
- Average field health scoring using a gauge visualization
- Simulated zero-shot erosion detection via Hugging Face CLIP model logic
- Adaptive remediation guide for:
  - Healthy Soil
  - Sheet Erosion (Mild)
  - Gully Erosion (Severe)
- Localized plant and stabilization suggestions based on region input
- Before/after comparison slider for severe erosion cases
- Downloadable PDF report summary
- Intel OpenVINO optimization placeholder and branding

## Tech Stack

- Python 3.14+
- Streamlit
- Transformers (Hugging Face)
- Pillow
- ReportLab
- Plotly
- `streamlit-image-comparison`
- `opencv-python-headless` for cloud compatibility

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/porkyprotection-collab/TerraGuard.git
   cd TerraGuard
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app locally:
```bash
streamlit run app.py
```

Then open the local URL provided by Streamlit in your browser.

## How It Works

1. Upload 1-5 images of land or soil.
2. Enter an optional region/country to get localized remediation suggestions.
3. TerraGuard computes an average erosion score and displays a visual gauge.
4. The app shows a field health assessment and a detailed action plan.
5. For severe cases, a comparison slider demonstrates restored land versus eroded land.
6. Generate a PDF report summarizing the results.

## Project Structure

- `app.py`: Main application file containing UI, detection logic, remediation guidance, and PDF generation.
- `requirements.txt`: Project dependencies.
- `README.md`: Project documentation.

## Notes

- `cv2` is intentionally removed in favor of `opencv-python-headless` for Streamlit Cloud compatibility.
- The CLIP-based detection is a simulated proof-of-concept; production use should be backed by a fine-tuned erosion dataset and optimized model.
- The current restoration image in the comparison slider is a placeholder for demonstration.

## Deployment

This app is best deployed on Streamlit Cloud for native Streamlit support. If deploying elsewhere, ensure the environment includes all required Python packages and supports Streamlit apps.

## Contribution

Contributions, bug reports, and feature requests are welcome. Submit issues or pull requests on the GitHub repository.

## License

Add your preferred license information here.
