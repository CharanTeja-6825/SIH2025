import kaggle

kaggle.api.authenticate()

path = kaggle.api.dataset_download_files("thedevastator/predicting-job-titles-from-resumes", unzip=True, path="../content/")


