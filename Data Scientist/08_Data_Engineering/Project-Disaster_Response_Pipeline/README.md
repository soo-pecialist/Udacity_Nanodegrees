# Disaster Response Pipeline Project

## Project Motivation

In this project, I made Disaster Response Pipeline Dashboard. Data is provided by [figure-eight](https://www.figure-eight.com/) and you can analyze messages with classifier working behind the scene. 

## File Description

    .
    ├── app     
    │   ├── run.py                           # Flask file that runs app
    │   └── templates   
    │       ├── go.html                      # Classification result page of web app
    │       └── master.html                  # Main page of web app    
    ├── data                   
    │   ├── disaster_categories.csv          # Dataset including all the categories  
    │   ├── disaster_messages.csv            # Dataset including all the messages
    │   └── process_data.py                  # Data cleaning
    ├── models
    │   └── train_classifier.py              # Train ML model           
    └── README.md


## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots
![Front page](https://raw.githubusercontent.com/soo-pecialist/Udacity_Nanodegrees/master/Data%20Scientist/08_Data_Engineering/Project-Disaster_Response_Pipeline/image/master.png)
![Example](https://raw.githubusercontent.com/soo-pecialist/Udacity_Nanodegrees/master/Data%20Scientist/08_Data_Engineering/Project-Disaster_Response_Pipeline/image/go.png)