# Scintill-AI
A research project aiming to apply machine learning models for regional Global Navigation Satellite System (**GNSS**) **ionospheric scintillation forecasting** at low latitudes.

## How can I run it?

- First, you need to clone the repo and install **dependencies** via [poetry](https://python-poetry.org/docs/) with `poetry install`

- To launch a web server and execute **jupyter notebooks**, (on Windows) you can run the `scripts/run-jupyter.ps1` script; otherwise, you can activate the virtual environment manually (via `poetry shell`) and then execute the `poetry run jupyter notebook` command

<!---
- To start an **[MLflow](https://mlflow.org/) tracking server**, (on Windows) you can run the `scripts/run-mlflow-ui.ps1` script; the **tracking UI** can be accessed locally by navigating to `http://localhost:5000/`

- Launch the **web app** via `streamlit run ./app/0_🏠_Home.py`
-->

## How can I help?

Contributions are what make the open source community an amazing place to learn, inspire, and create. Any contribution you make is **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature_amazing_feature`)
3. Commit your Changes (`git commit -m 'Add some amazing stuff'`)
4. Push to the Branch (`git push origin feature_amazing_feature`)
5. Open a Pull Request

<!---
An (hopefully) up-to-date list of things to do can be found [here](https://github.com/viventriglia/t-fors/blob/develop/todo.md?plain=1).
-->

## Data

- [INTERMAGNET](https://imag-data.bgs.ac.uk/GIN_V1/GINForms2?observatoryIagaCode=KOU&publicationState=Best+available&dataStartDate=2014-01-01&dataDuration=10&submitValue=Bulk+Download+...&request=DataView&samplesPerDay=minute): ground-based magnetometers in Kourou (KOU, French Guiana) and Tatuoca (TTB, Brazil)
