# Scintill-AI
A research project aiming to apply machine learning models for regional Global Navigation Satellite System (**GNSS**) **ionospheric scintillation forecasting** at low latitudes.

## Table of Contents

- [What is it?](#what-is-it)

- [How can I run it?](#how-can-i-run-it)

- [How can I help?](#how-can-i-help)

- [Data](#data) 

## What is it?

The ionosphere contains ionised particles that are generally homogeneously distributed, and GNSS receivers – which use the signals from satellites orbiting the Earth to calculate their locations – can account for their effect on satellite signals using models. However, problems arise when there are irregularities, *i.e.* **localised fluctuations in the electron density** of the ionosphere, which can distort the phase and amplitude of GNSS signals, producing fluctuations known as *scintillations*.

The appearance of scintillation is often deemed unpredictable. It varies throughout the day, with sunset triggering a sharp increase in ionospheric activity that can last several hours. Also, an increase in solar activity can produce scintillation events that can **degrade the quality of satellite signals**. In standard GNSS receivers, a mild scintillation can degrade position accuracy by up to several metres. More severe scintillation can cause cycle slips or, in the most extreme cases, total loss of signal lock. So, whether it comes to precision agriculture in Brazil, oil exploration in Alaska or a large construction project in Singapore, it is highly beneficial to forecast the onset of scintillation.

| ![Cars and a plane on a sunset sky](images/scintill_ai_cover.jpeg) | 
|:--:| 
| *Generated image of cars and an aircraft under a sunset sky* |

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

- [INTERMAGNET](https://imag-data.bgs.ac.uk/GIN_V1/GINForms2?observatoryIagaCode=KOU&publicationState=Best+available&dataStartDate=2014-01-01&dataDuration=10&submitValue=Bulk+Download+...&request=DataView&samplesPerDay=minute): ground-based magnetometers in Kourou (KOU, French Guiana) and Tatuoca (TTB, Brazil). To download the raw data, open a unix shell, cd into `scripts` and execute `./download_magnetometer_data.sh`, which will download the raw data into the `./data/in` directory.

- [ISMR](https://ismrquerytool.fct.unesp.br/is/index.php#): GNSS receiver in Presidente Prudente (PRU2, Brazil). The access to the API is exclusive for authorized researchers and a unique key is required. Your key is provided when you access [the webservice](https://ismrquerytool.fct.unesp.br/is/ismrtool/manual/mkdocs-ismrtools/webservice/); once you have it, place it in the `.env.shared` file and rename it to `.env.secret`.

- [OMNIWeb](https://omniweb.gsfc.nasa.gov/form/omni_min.html): field and plasma data shifted to the Earth's bow shock nose. To download the raw data, open a unix shell, cd into `scripts` and execute `./download_omniweb_data.sh`, which will download the raw data into the `./data/in/omniweb` directory. [Here](./scripts/download_omniweb_data_vars.md) a (partial) list of fields available for download.

- [GFZ Potsdam](https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt): solar indices data can be downloaded directly via the `scintill_ai.io` module.