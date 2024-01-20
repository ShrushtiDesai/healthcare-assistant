
from email.message import EmailMessage
import pymongo
from pymongo import MongoClient
import smtplib , ssl # two types - ssl, tls ssl - socket security layer and tls tranport layer security ssl - connection start tls - transport start 
from getpass import getpass # enter input will be not be echoed 
import pyttsx3 # test to speech or speech to text
import datetime  #  current date time 
import sys       # program exit with a message 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns               # graphical 
import matplotlib.pyplot as plt
import playsound
from sklearn.preprocessing import LabelEncoder as LabelEncoder # encodes class in 0 to n-1 classes
from warnings import filterwarnings
filterwarnings("ignore")
from sklearn.model_selection import train_test_split   # splits dataset into two - train and test 
from sklearn.tree import DecisionTreeClassifier          
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
import speech_recognition as sr # to recognizse voice
from gtts import gTTS
from email.mime.text import MIMEText                # to send mail 
from email.mime.multipart import MIMEMultipart

i=0
engine = pyttsx3.init() # initialized pytts

def speak(audio):         # function to read text given to speak()
    engine.say(audio)
    engine.runAndWait()

def takeCommand():            # to take voice input                         
    r = sr.Recognizer()
    with sr.Microphone() as source:        # mic as input source
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source, timeout=8, phrase_time_limit=5)
        said = ""            
        try:
            print("Recognizing...")
            said = r.recognize_google(audio, language ='en-in')  
            print("You said : "+said)
        
        except Exception as e:
            print(e)
            print("Unable to Recognize your voice.")
            speak("Unable to Recognize your voice.")

        return said


def diseases():
    train = pd.read_csv(r"C:\Users\Shrushti\Downloads\Training.csv\Training.csv")
    test = pd.read_csv(r"C:\Users\Shrushti\Downloads\Testing.csv")
    A = train
    B = test

    encoder = LabelEncoder()
    A["prognosis"] = encoder.fit_transform(A["prognosis"])

    A = A.drop(["Unnamed: 133"],axis=1)

    Y = A[["prognosis"]]
    X = A.drop(["prognosis"],axis=1)
    P = B.drop(["prognosis"],axis=1)
    #TRAINING IS SPLIT INOT TWO PARTS 
    xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=42)

    dtc= DecisionTreeClassifier(random_state=42)
    dtc_model = dtc.fit(xtrain,ytrain)
    tr_pred_dtc = dtc_model.predict(xtrain)
    ts_pred_dtc = dtc_model.predict(xtest)

    # Training and testing Naive Bayes Classifier
    nbc = GaussianNB()
    nbc_model = nbc.fit(xtrain, ytrain)
    tr_pred_nbc = nbc_model.predict(xtrain)
    ts_pred_nbc = nbc_model.predict(xtest)

    #prediction for full train dataset which was previously splited into two and then it will be compared with the test dataset 
    dtc_final_model = DecisionTreeClassifier()
    nbc_final_model = GaussianNB()
    dtc_final_model.fit(X,Y)
    nbc_final_model.fit(X,Y)

    test_X = B.iloc[:, :-1]
    test_Y = encoder.transform(B.iloc[:, -1])

    dtc_final_pred = dtc_final_model.predict(test_X)
    nbc_final_pred = nbc_final_model.predict(test_X)

    Final_prediction = [mode([i,j])[0][0] for i,j
                     in zip(dtc_final_pred,nbc_final_pred)]


    disease_counts = B["prognosis"].value_counts()
    temp_df = pd.DataFrame({
	    "Disease": disease_counts.index,
	    "Counts": disease_counts.values
    })

    symptoms = X.columns.values
    # Creating a symptom index dictionary to encode the input symptoms into numerical form
    symptom_index = {}
    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")]) # Skin Rash
        symptom_index[symptom] = index

    data_dict = {                                                               # skin rash,itching, nodal : 1
        "symptom_index":symptom_index,
        "predictions_classes":encoder.classes_
    }
    
    def predictDisease(symptoms):
        symptoms = symptoms.title()                 # Skin Rash 
        symptoms = symptoms.split(", ")
     
        # creating input data for the models
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
         
        # reshaping the input data and converting it       # 1 : fungal infection
        # into suitable format for model predictions
        input_data = np.array(input_data).reshape(1,-1)

        # generating individual outputs
        dtc_prediction = data_dict["predictions_classes"][dtc_final_model.predict(input_data)[0]]
        nbc_prediction = data_dict["predictions_classes"][nbc_final_model.predict(input_data)[0]]


        # making final prediction by taking mode of all predictions
        final_prediction = mode([dtc_prediction, nbc_prediction])[0]

        predictions = f'''
            dtc_model_prediction : {dtc_prediction}
            naive_bayes_prediction : {nbc_prediction}
            final_prediction : {final_prediction}
           '''

        return dtc_prediction

    speak("Enter your symptoms separated by commas")
    
    prediction = f'''{(predictDisease(input(" Enter your symptoms sepearated by comma(s) : ")))}'''

    speak(f'''there is a possibility that you might have {prediction}''')
    prediction = prediction.lower()

    if 'fungal infection' in prediction:
        print(f'''Fungal infections are common in humans and are usually not very serious if they are treated quickly and correctly.
                  It occur when an invading fungus takes over an area of the body and is too much for the immune system to handle
                  Anyone with a weakened immune system may be more likely to contract a fungal infection, as well as anyone who is taking antibiotics.
                  Cancer treatment and diabetes may also make a person more prone to fungal infections.
                  you can consult with a fungal infection specialist or a dermatologist.''')

        speak(f'''Fungal infections are common in humans and are usually not very serious if they are treated quickly and correctly.
                  It occur when an invading fungus takes over an area of the body and is too much for the immune system to handle
                  Anyone with a weakened immune system may be more likely to contract a fungal infection, as well as anyone who is taking antibiotics.
                  Cancer treatment and diabetes may also make a person more prone to fungal infections.
                  you can consult with a fungal infection specialist or a dermatologist.''')

    elif 'allergy' in prediction:
        print(f'''allergy is a condition in which the immune system reacts abnormally to a foreign substance
                  such as pollen, bee venom or pet dander — or a food that doesn't cause a reaction in most people.
                  When you have allergies, your immune system makes antibodies that identify a particular allergen as harmful, even though it isn't.
                  They are very common, you can consult an Allergist for treatment.''')

        speak(f'''allergy is a condition in which the immune system reacts abnormally to a foreign substance
                  such as pollen, bee venom or pet dander — or a food that doesn't cause a reaction in most people.
                  When you have allergies, your immune system makes antibodies that identify a particular allergen as harmful, even though it isn't.
                  They are very common, you can consult an Allergist for treatment.''')

    elif 'GERD' in prediction:
        print(f'''Gastroesophageal reflux disease (GERD) occurs when stomach acid repeatedly flows back into the tube connecting your mouth and stomach (esophagus).
                  This backwash (acid reflux) can irritate the lining of your esophagus.
                  Many people experience acid reflux from time to time. However, when acid reflux happens repeatedly over time, it can cause GERD. it is Very common as More than 10 million cases per year are found but it is good to 
                  get it checked. you can consult a Gastroenterologist for GERD.''')

        speak(f'''Gastroesophageal reflux disease (GERD) occurs when stomach acid repeatedly flows back into the tube connecting your mouth and stomach (esophagus).
                  This backwash (acid reflux) can irritate the lining of your esophagus.
                  Many people experience acid reflux from time to time. However, when acid reflux happens repeatedly over time, it can cause GERD. it is Very common as More than 10 million cases per year are found but it is good to 
                  get it checked. you can consult a Gastroenterologist for GERD.''')

    elif 'chronic cholestasis' in prediction:
        print(f'''Cholestasis is a liver disease. It occurs when the flow of bile from your liver is reduced or blocked. 
                  Bile is fluid produced by your liver that aids in the digestion of food, especially fats. When bile flow is altered, 
                  it can lead to a buildup of bilirubin. Bilirubin is a pigment produced by your liver and excreted from your body via bile,
                  in most cases it is caused by primary damage to the biliary epithelium. You can consult to a hepatologist or a general surgeon.''')

        speak(f'''Cholestasis is a liver disease. It occurs when the flow of bile from your liver is reduced or blocked. 
                  Bile is fluid produced by your liver that aids in the digestion of food, especially fats. When bile flow is altered, 
                  it can lead to a buildup of bilirubin. Bilirubin is a pigment produced by your liver and excreted from your body via bile,
                  in most cases it is caused by primary damage to the biliary epithelium. You can consult to a hepatologist or a general surgeon.''')

    elif 'drug reaction' in prediction:
        print(f'''A drug allergy is the reaction of the immune system to a medicine. Any medicine — nonprescription, prescription or herbal — can provoke a drug allergy. 
                  However, a drug allergy is more likely with certain medicines. A drug allergy is not the same as a drug side effect. 
                  A side effect is a known possible reaction to a medicine. Side effects to medicines are listed on their drug labels.
                  A drug allergy also is different from drug toxicity. Drug toxicity is caused by an overdose of medicine. 
                  Symptoms of a serious drug allergy often occur within an hour after taking a drug. you can with consult to an allergist or immunologist.''')

        speak(f'''A drug allergy is the reaction of the immune system to a medicine. Any medicine — nonprescription, prescription or herbal — can provoke a drug allergy. 
                  However, a drug allergy is more likely with certain medicines. A drug allergy is not the same as a drug side effect. 
                  A side effect is a known possible reaction to a medicine. Side effects to medicines are listed on their drug labels.
                  A drug allergy also is different from drug toxicity. Drug toxicity is caused by an overdose of medicine. 
                  Symptoms of a serious drug allergy often occur within an hour after taking a drug. you can with consult to an allergist or immunologist.''')

    elif 'peptic ulcer disease' in prediction:
        print(f'''Peptic ulcers are open sores that develop on the inside lining of your stomach and the upper portion of your small intestine. 
                  The most common symptom of a peptic ulcer is stomach pain.he most common causes of peptic ulcers are infection with the bacterium Helicobacter pylori (H. pylori)
                  and long-term use of nonsteroidal anti-inflammatory drugs (NSAIDs) such as ibuprofen (Advil, Motrin IB, others) and naproxen sodium (Aleve). 
                  Stress and spicy foods do not cause peptic ulcers. However, they can make your symptoms worse.The most common peptic ulcer symptom is burning stomach pain. 
                  You can consult with a gastroenterologist for this disease.''')

        speak(f'''Peptic ulcers are open sores that develop on the inside lining of your stomach and the upper portion of your small intestine. 
                  The most common symptom of a peptic ulcer is stomach pain.he most common causes of peptic ulcers are infection with the bacterium Helicobacter pylori (H. pylori)
                  and long-term use of nonsteroidal anti-inflammatory drugs (NSAIDs) such as ibuprofen (Advil, Motrin IB, others) and naproxen sodium (Aleve). 
                  Stress and spicy foods do not cause peptic ulcers. However, they can make your symptoms worse.The most common peptic ulcer symptom is burning stomach pain. 
                  You can consult with a gastroenterologist for this disease.''')

    elif 'aids' in prediction:
        print(f'''Acquired immunodeficiency syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV). 
                  By damaging your immune system, HIV interferes with your body's ability to fight infection and disease. 
                  HIV is a sexually transmitted infection (STI). It can also be spread by contact with infected blood and from illicit injection drug use or sharing needles. 
                  It can also be spread from mother to child during pregnancy, childbirth or breastfeeding. There's no cure for HIV/AIDS, 
                  but medications can control the infection and prevent progression of the disease. You can consult with a infectious disease specialist.''')

        speak(f'''Acquired immunodeficiency syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV). 
                  By damaging your immune system, HIV interferes with your body's ability to fight infection and disease. 
                  HIV is a sexually transmitted infection (STI). It can also be spread by contact with infected blood and from illicit injection drug use or sharing needles. 
                  It can also be spread from mother to child during pregnancy, childbirth or breastfeeding. There's no cure for HIV/AIDS, 
                  but medications can control the infection and prevent progression of the disease. You can consult with a infectious disease specialist.''')

    elif 'diabetes' in prediction:
        print(f'''Diabetes, also known as diabetes mellitus refers to a group of diseases that affect how the body uses blood sugar (glucose).
                  Glucose is an important source of energy for brain and for the cells that make up the muscles and tissues. 
                  The main cause of diabetes varies by type. But no matter what type of diabetes you have, 
                  it can lead to excess sugar in the blood and Too much sugar in the blood can lead to serious health problems. you can consult with a endocrinologist for diabetes.''')

        speak(f'''Diabetes, also known as diabetes mellitus refers to a group of diseases that affect how the body uses blood sugar (glucose).
                  Glucose is an important source of energy for brain and for the cells that make up the muscles and tissues. 
                  The main cause of diabetes varies by type. But no matter what type of diabetes you have, 
                  it can lead to excess sugar in the blood and Too much sugar in the blood can lead to serious health problems. you can consult with a endocrinologist for diabetes.''')

    elif 'gastroenteritis' in prediction:
        print(f'''Gastroenteritis commonly called “stomach flu.” It is not related to influenza, it is an inflammation of your stomach and intestines.
                  Stomach flu is typically spread by contact with an infected person or through contaminated food or water.
                  Diarrhoea, cramps, nausea, vomiting and low-grade fever are common symptoms.
                  Avoiding contaminated food and water and washing hands can often help prevent infection. 
                  Rest and rehydration are the mainstays of treatment. you can consult with a gastroenterologist for treatment.''')

        speak(f'''Gastroenteritis commonly called “stomach flu.” It is not related to influenza, it is an inflammation of your stomach and intestines.
                  Stomach flu is typically spread by contact with an infected person or through contaminated food or water.
                  Diarrhoea, cramps, nausea, vomiting and low-grade fever are common symptoms.
                  Avoiding contaminated food and water and washing hands can often help prevent infection. 
                  Rest and rehydration are the mainstays of treatment. you can consult with a gastroenterologist for treatment.''')

    elif 'bronchial asthma' in prediction:
        print(f'''Bronchial asthma is a chronic inflammatory disease of the airways characterized by bronchial hyperreactivity and a variable degree of airway obstruction. 
                  it is a chronic inflammatory disease of the airways characterized by bronchial hyperreactivity and a variable degree of airway obstruction. 
                  Asthma can't be cured, but its symptoms can be controlled. you can consult with a plumonologist or respiratory therapist.''')

        speak(f'''Bronchial asthma is a chronic inflammatory disease of the airways characterized by bronchial hyperreactivity and a variable degree of airway obstruction. 
                  it is a chronic inflammatory disease of the airways characterized by bronchial hyperreactivity and a variable degree of airway obstruction. 
                  Asthma can't be cured, but its symptoms can be controlled. you can consult with a plumonologist or respiratory therapist.''')

    elif 'hypertension' in prediction:
        print(f'''High blood pressure is a common condition that affects the body's arteries. 
                  It's also called hypertension.A condition in which the force of the blood against the artery walls is too high. The heart has to work harder to pump blood.
                  Blood pressure i measured in millimeters of mercury (mm Hg). In general, hypertension is a blood pressure reading of 130/80 mm Hg or higher. 
                  Blood pressure higher than 180/120 mm Hg is considered a hypertensive emergency or crisis. you can consult with a cardiologis or a nephrologist.''')

        speak(f'''High blood pressure is a common condition that affects the body's arteries. 
                  It's also called hypertension.A condition in which the force of the blood against the artery walls is too high. The heart has to work harder to pump blood.
                  Blood pressure i measured in millimeters of mercury (mm Hg). In general, hypertension is a blood pressure reading of 130/80 mm Hg or higher. 
                  Blood pressure higher than 180/120 mm Hg is considered a hypertensive emergency or crisis. you can consult with a cardiologis or a nephrologist.''')

    elif 'migraine' in prediction:
        print(f'''A migraine is a headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. 
                  It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine headaches are sometimes preceded by warning symptoms. 
                  Triggers include hormonal changes, certain food and drink, stress and exercise.
                  Nausea and sensitivity to light and sound are also common symptoms.
                  Preventive and pain-relieving medication can help manage migraine headaches. you can consult with a general physician or a neurologist.''')

        speak(f'''A migraine is a headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. 
                  It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine headaches are sometimes preceded by warning symptoms. 
                  Triggers include hormonal changes, certain food and drink, stress and exercise.
                  Nausea and sensitivity to light and sound are also common symptoms.
                  Preventive and pain-relieving medication can help manage migraine headaches. you can consult with a general physician or a neurologist.''')

    elif 'cervical spondylosis' in prediction:
        print(f'''Cervical spondylosis is a general term for age-related wear and tear affecting the spinal disks in your neck. 
                  As the disks dehydrate and shrink, signs of osteoarthritis develop, including bony projections along the edges of bones (bone spurs). 
                  Cervical spondylosis is very common and worsens with age. Most people experience no symptoms. When symptoms do occur, they typically include pain and stiffness in the neck. 
                  Seek medical attention if you notice a sudden onset of numbness or weakness, or loss of bladder or bowel control. You can consult with a rheumatologist or neurologist.''' )

        speak(f'''Cervical spondylosis is a general term for age-related wear and tear affecting the spinal disks in your neck. 
                  As the disks dehydrate and shrink, signs of osteoarthritis develop, including bony projections along the edges of bones (bone spurs). 
                  Cervical spondylosis is very common and worsens with age. Most people experience no symptoms. When symptoms do occur, they typically include pain and stiffness in the neck. 
                  Seek medical attention if you notice a sudden onset of numbness or weakness, or loss of bladder or bowel control. You can consult with a rheumatologist or neurologist.''')

    elif 'paralysis (brain hemorrhage)' in prediction:
        print(f'''Paralysis is the loss of the ability to move some or all of your body.
                  It can have lots of different causes, some of which can be serious. 
                  Depending on the cause, it may be temporary or permanent. dial 102 or 112 in case of emergency. you can consult a General physician or neurologist in severe cases.''')

        speak(f'''Paralysis is the loss of the ability to move some or all of your body.
                  It can have lots of different causes, some of which can be serious. 
                  Depending on the cause, it may be temporary or permanent. dial 102 or 112 in case of emergency. you can consult a General physician or neurologist in severe cases.''')

    elif 'jaundice' in prediction:
        print(f'''Jaundice is a condition in which the skin, sclera (whites of the eyes) and mucous membranes turn yellow. 
                  This yellow color is caused by a high level of bilirubin, a yellow-orange bile pigment. Bile is fluid secreted by the liver. 
                  Bilirubin is formed from the breakdown of red blood cells. Jaundice may occur if the liver can't efficiently process red blood cells as they break down. 
                  It's normal in healthy newborns and usually clears on its own. Symptoms include yellowing of the skin and whites of the eyes. 
                  You can consult with a genreal physician or gastroenterologist to know the root cause.''')

        speak(f'''Jaundice is a condition in which the skin, sclera (whites of the eyes) and mucous membranes turn yellow. 
                  This yellow color is caused by a high level of bilirubin, a yellow-orange bile pigment. Bile is fluid secreted by the liver. 
                  Bilirubin is formed from the breakdown of red blood cells. Jaundice may occur if the liver can't efficiently process red blood cells as they break down. 
                  It's normal in healthy newborns and usually clears on its own. Symptoms include yellowing of the skin and whites of the eyes. 
                  You can consult with a genreal physician or gastroenterologist to know the root cause.''')

    elif 'malaria' in prediction:
        print(f'''Malaria is a disease caused by a plasmodium parasite. The parasite is spread to humans through the bites of infected mosquitoes. 
                  People who have malaria usually feel very sick with a high fever and shaking chills. The severity of malaria varies based on the species of plasmodium.
                  Prevention includes Mosquito nets, insect repellent, mosquito control, medications cleanliness and covering all container holding water. 
                  you can consult with a infectious disease specialist.''')

        speak(f'''Malaria is a disease caused by a plasmodium parasite. The parasite is spread to humans through the bites of infected mosquitoes. 
                  People who have malaria usually feel very sick with a high fever and shaking chills. The severity of malaria varies based on the species of plasmodium.
                  Prevention includes Mosquito nets, insect repellent, mosquito control, medications cleanliness and covering all container holding water. 
                  you can consult with a infectious disease specialist.''')

    elif 'chicken pox' in prediction:
        print(f'''chicken pox is an infectious disease causing a mild fever and a rash of itchy inflamed pimples which turn to blisters and then loose scabs. 
                  It is caused by the herpes zoster virus and mainly affects children.Chickenpox can be prevented by a vaccine. 
                  Treatment usually involves relieving symptoms, although high-risk groups may receive antiviral medication. 
                  you can consult with a general physician or a pediatrician.''')

        speak(f'''chicken pox is an infectious disease causing a mild fever and a rash of itchy inflamed pimples which turn to blisters and then loose scabs. 
                  It is caused by the herpes zoster virus and mainly affects children.Chickenpox can be prevented by a vaccine. 
                  Treatment usually involves relieving symptoms, although high-risk groups may receive antiviral medication. 
                  you can consult with a general physician or a pediatrician.''')

    elif 'dengue' in prediction:
        print(f'''Dengue is an infection caused by a virus. You can get it if an infected mosquito bites you. Dengue does not spread from person to person. 
                  It is common in warm, wet areas of the world. Symptoms include a high fever, headaches, joint and muscle pain, vomiting, and a rash. 
                  In some cases, dengue turns into dengue hemorrhagic fever, which causes bleeding from your nose, gums, or under your skin. 
                  It can also become dengue shock syndrome, which causes massive bleeding and shock. These forms of dengue are life-threatening.
                  There is no specific treatment. Most people with dengue recover within 2 weeks. 
                  Until then, drinking lots of fluids, resting and taking non-aspirin fever-reducing medicines might help. 
                  People with the more severe forms of dengue usually need to go to the hospital and get fluids. you can consult with an infectious disease speacialist.''')

        speak(f'''Dengue is an infection caused by a virus. You can get it if an infected mosquito bites you. Dengue does not spread from person to person. 
                  It is common in warm, wet areas of the world. Symptoms include a high fever, headaches, joint and muscle pain, vomiting, and a rash. 
                  In some cases, dengue turns into dengue hemorrhagic fever, which causes bleeding from your nose, gums, or under your skin. 
                  It can also become dengue shock syndrome, which causes massive bleeding and shock. These forms of dengue are life-threatening.
                  There is no specific treatment. Most people with dengue recover within 2 weeks. 
                  Until then, drinking lots of fluids, resting and taking non-aspirin fever-reducing medicines might help. 
                  People with the more severe forms of dengue usually need to go to the hospital and get fluids. you can consult with an infectious disease speacialist.''')

    elif 'typhoid' in prediction:
        print(f'''Typhoid fever is a bacterial infection that can spread throughout the body, affecting many organs. Without prompt treatment, 
                  it can cause serious complications and can be fatal. It's caused by a bacterium called Salmonella typhi. 
                  it is an infection that spreads through contaminated food and water.Vaccines are recommended in areas where typhoid fever is common.
                  Symptoms include high fever, headache, stomach pain, weakness, vomiting and loose stools.
                  Treatment includes antibiotics and fluids. you can consult with a infectious disease speacialsit.''')

        speak(f'''Typhoid fever is a bacterial infection that can spread throughout the body, affecting many organs. Without prompt treatment, 
                  it can cause serious complications and can be fatal. It's caused by a bacterium called Salmonella typhi. 
                  it is an infection that spreads through contaminated food and water.Vaccines are recommended in areas where typhoid fever is common.
                  Symptoms include high fever, headache, stomach pain, weakness, vomiting and loose stools.
                  Treatment includes antibiotics and fluids. you can consult with a infectious disease speacialsit.''')

    elif 'hepatitis a' in prediction:
        print(f'''hepatitis A is an highly contagious liver infection caused by the hepatitis A virus.
                  Hepatitis A is preventable by vaccine. It spreads from contaminated food or water or contact with someone who is infected.
                  Symptoms include fatigue, nausea, abdominal pain, loss of appetite and low-grade fever.
                  The condition clears up on its own in one or two months. Rest and adequate hydration can help. you can consult with a hepatologist.''')

        speak(f'''hepatitis A is an highly contagious liver infection caused by the hepatitis A virus.
                  Hepatitis A is preventable by vaccine. It spreads from contaminated food or water or contact with someone who is infected.
                  Symptoms include fatigue, nausea, abdominal pain, loss of appetite and low-grade fever.
                  The condition clears up on its own in one or two months. Rest and adequate hydration can help. you can consult with a hepatologist.''')

    elif 'hepatitis b' in prediction:
        print(f'''Hepatitis B is a serious liver infection caused by the hepatitis B virus that's easily preventable by a vaccine.
                  This disease is most commonly spread by exposure to infected bodily fluids.
                  Symptoms are variable and include yellowing of the eyes, abdominal pain and dark urine. Some people, particularly children, don't experience any symptoms. 
                  In chronic cases, liver failure, cancer or scarring can occur.
                  The condition often clears up on its own. Chronic cases require medication and possibly a liver transplant. you can consult with a hepatologist.''')

        speak(f'''Hepatitis B is a serious liver infection caused by the hepatitis B virus that's easily preventable by a vaccine.
                  This disease is most commonly spread by exposure to infected bodily fluids.
                  Symptoms are variable and include yellowing of the eyes, abdominal pain and dark urine. Some people, particularly children, don't experience any symptoms. 
                  In chronic cases, liver failure, cancer or scarring can occur.
                  The condition often clears up on its own. Chronic cases require medication and possibly a liver transplant. you can consult with a hepatologist.''')

    elif 'hepatitis c' in prediction:
        print(f'''Hepatitis C is an infection caused by a virus that attacks the liver and leads to inflammation.
                  The virus is spread by contact with contaminated blood; for example, from sharing needles or from unsterile tattoo equipment.
                  Most people have no symptoms. Those who do develop symptoms may have fatigue, nausea, loss of appetite and yellowing of the eyes and skin.
                  Hepatitis C is treated with antiviral medication. In some people, newer medicines can eradicate the virus. you can consult with a hepatologist.''')

        speak(f'''Hepatitis C is an infection caused by a virus that attacks the liver and leads to inflammation.
                  The virus is spread by contact with contaminated blood; for example, from sharing needles or from unsterile tattoo equipment.
                  Most people have no symptoms. Those who do develop symptoms may have fatigue, nausea, loss of appetite and yellowing of the eyes and skin.
                  Hepatitis C is treated with antiviral medication. In some people, newer medicines can eradicate the virus. you can consult with a hepatologist.''')

    elif 'hepatitis d' in prediction:
        print(f'''Hepatits D is a serious liver disease caused by infection with the hepatitis D virus.
                  Hepatitis D only occurs amongst people who are infected with the Hepatitis B virus. Transmission requires contact with infectious blood. 
                  At-risk populations include intravenous drug abusers and men who have sex with men.
                  Symptoms include abdominal pain, nausea and fatigue.
                  There are few treatments specifically for hepatitis D, although different regimes may be tried. Management also focuses on supportive care. 
                  you can consult with a hepatologist.''')

        speak(f'''Hepatitis D is a serious liver disease caused by infection with the hepatitis D virus.
                  Hepatitis D only occurs amongst people who are infected with the Hepatitis B virus. Transmission requires contact with infectious blood. 
                  At-risk populations include intravenous drug abusers and men who have sex with men.
                  Symptoms include abdominal pain, nausea and fatigue.
                  There are few treatments specifically for hepatitis D, although different regimes may be tried. Management also focuses on supportive care. 
                  you can consult with a hepatologist.''')

    elif 'hepatitis e' in prediction:
        print(f'''Hepatitis E is a liver disease caused by the hepatitis E virus.
                  The hepatitis E virus is mainly transmitted through drinking water contaminated with faecal matter.
                  Symptoms include jaundice, lack of appetite and nausea. In rare cases, it may progress to acute liver failure.
                  Hepatitis E usually resolves on its own within four to six weeks. Treatment focuses on supportive care, rehydration and rest. 
                  you can consult with a hepatologist.''')

        speak(f'''Hepatitis E is a liver disease caused by the hepatitis E virus.
                  The hepatitis E virus is mainly transmitted through drinking water contaminated with faecal matter.
                  Symptoms include jaundice, lack of appetite and nausea. In rare cases, it may progress to acute liver failure.
                  Hepatitis E usually resolves on its own within four to six weeks. Treatment focuses on supportive care, rehydration and rest. 
                  you can consult with a hepatologist.''')

    elif 'alcoholic hepatitis' in prediction:
        print(f'''Alcoholic Hepatits is a liver inflammation caused by drinking too much alcohol.
                  Alcoholic hepatitis can occur in people who drink heavily for many years.
                  Symptoms include yellow skin and eyes along with increasing stomach size due to fluid accumulation.
                  Treatment involves hydration, nutritional care and stopping alcohol use. Steroid drugs can help reduce liver inflammation. 
                  you can consult with a hepatologist.''')

        speak(f'''Alcoholic Hepatits is a liver inflammation caused by drinking too much alcohol.
                  Alcoholic hepatitis can occur in people who drink heavily for many years.
                  Symptoms include yellow skin and eyes along with increasing stomach size due to fluid accumulation.
                  Treatment involves hydration, nutritional care and stopping alcohol use. Steroid drugs can help reduce liver inflammation. 
                  you can consult with a hepatologist.''')

    elif 'tuberculosis' in prediction:
        print(f'''Tuberculosis (TB) is caused by a bacterium called Mycobacterium tuberculosis. 
                  The bacteria usually attack the lungs, but TB bacteria can attack any part of the body such as the kidney, spine, and brain. 
                  Not everyone infected with TB bacteria becomes sick.Most people infected with the bacteria that cause tuberculosis don't have symptoms. 
                  When symptoms do occur, they usually include a cough (sometimes blood-tinged), weight loss, night sweats and fever.
                  Treatment isn't always required for those without symptoms. Patients with active symptoms will require a long course of treatment involving multiple antibiotics. 
                  you can consult with a pulmonologist.''')

        speak(f'''Tuberculosis (TB) is caused by a bacterium called Mycobacterium tuberculosis. 
                  The bacteria usually attack the lungs, but TB bacteria can attack any part of the body such as the kidney, spine, and brain. 
                  Not everyone infected with TB bacteria becomes sick.Most people infected with the bacteria that cause tuberculosis don't have symptoms. 
                  When symptoms do occur, they usually include a cough (sometimes blood-tinged), weight loss, night sweats and fever.
                  Treatment isn't always required for those without symptoms. Patients with active symptoms will require a long course of treatment involving multiple antibiotics. 
                  you can consult with a pulmonologist.''')

    elif 'common cold' in prediction:
        print(f'''Common cold is a common viral infection of the nose and throat.
                  In contrast to the flu, a common cold can be caused by many different types of viruses. 
                  The condition is generally harmless and symptoms usually resolve within two weeks.
                  Symptoms include a runny nose, sneezing and congestion. High fever or severe symptoms are reasons to see a doctor, especially in children.
                  Most people recover on their own within two weeks. Over-the-counter products and home remedies can help control symptoms. 
                  you can consult with a general physician or pediatrician.''')

        speak(f'''Common cold is a common viral infection of the nose and throat.
                  In contrast to the flu, a common cold can be caused by many different types of viruses. 
                  The condition is generally harmless and symptoms usually resolve within two weeks.
                  Symptoms include a runny nose, sneezing and congestion. High fever or severe symptoms are reasons to see a doctor, especially in children.
                  Most people recover on their own within two weeks. Over-the-counter products and home remedies can help control symptoms. 
                  you can consult with a general physician or pediatrician.''')

    elif 'pneumonia' in prediction:
        print(f'''Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), 
                  causing cough with phlegm or pus, fever, chills, and difficulty breathing. The infection can be life-threatening to anyone, but particularly to infants, 
                  children and people over 65. Symptoms include a cough with phlegm or pus, fever, chills and difficulty breathing.
                  Antibiotics can treat many forms of pneumonia. Some forms of pneumonia can be prevented by vaccines. 
                  you can consult with a pulmonologist or respiratory therapist.''')

        speak(f'''Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), 
                  causing cough with phlegm or pus, fever, chills, and difficulty breathing. The infection can be life-threatening to anyone, but particularly to infants, 
                  children and people over 65. Symptoms include a cough with phlegm or pus, fever, chills and difficulty breathing.
                  Antibiotics can treat many forms of pneumonia. Some forms of pneumonia can be prevented by vaccines. 
                  you can consult with a pulmonologist or respiratory therapist.''')

    elif 'dimorphic hemmorhoids(piles)' in prediction:
        print(f'''Piles (haemorrhoids) are lumps inside and around your bottom that cause discomfort and bleeding. 
                  They often get better on their own after a few days.Haemorrhoids are usually caused by straining during bowel movements, obesity or pregnancy.
                  Discomfort is a common symptom, especially during bowel movements or when sitting. Other symptoms include itching and bleeding.
                  A high-fibre diet can be effective, along with stool softeners. 
                  In some cases, a medical procedure to remove the haemorrhoid may be needed to provide relief. you can consult with a proctologist.''')

        speak(f'''Piles (haemorrhoids) are lumps inside and around your bottom that cause discomfort and bleeding. 
                  They often get better on their own after a few days.Haemorrhoids are usually caused by straining during bowel movements, obesity or pregnancy.
                  Discomfort is a common symptom, especially during bowel movements or when sitting. Other symptoms include itching and bleeding.
                  A high-fibre diet can be effective, along with stool softeners. 
                  In some cases, a medical procedure to remove the haemorrhoid may be needed to provide relief. you can consult with a proctologist.''')

    elif 'heart attack' in prediction:
        print(f'''Heart attack is also called as myocardial infarction. A heart attack is a medical emergency. dial 112 in emergency. 
                  A heart attack usually occurs when a blood clot blocks blood flow to the heart. Without blood, tissue loses oxygen and dies.
                  Symptoms include tightness or pain in the chest, neck, back or arms, as well as fatigue, lightheadedness, abnormal heartbeat and anxiety. 
                  Women are more likely to have atypical symptoms than men.
                  Treatment ranges from lifestyle changes and cardiac rehabilitation to medication, stents and bypass surgery. 
                  you can consult with a cardiologist.''')

        speak(f'''Heart attack is also called as myocardial infarction. A heart attack is a medical emergency. dial 112 in emergency.
                  A heart attack usually occurs when a blood clot blocks blood flow to the heart. Without blood, tissue loses oxygen and dies.
                  Symptoms include tightness or pain in the chest, neck, back or arms, as well as fatigue, lightheadedness, abnormal heartbeat and anxiety. 
                  Women are more likely to have atypical symptoms than men.
                  Treatment ranges from lifestyle changes and cardiac rehabilitation to medication, stents and bypass surgery. 
                  you can consult with a cardiologist.''')

    elif 'varicose veins' in prediction:
        print(f'''Varicose veins are twisted, Gnarled, enlarged veins, most commonly appearing in the legs and feet.
                  Varicose veins are generally benign. The cause of this condition is not known.
                  For many people, there are no symptoms and varicose veins are simply a cosmetic concern. 
                  In some cases, they cause aching pain and discomfort or signal an underlying circulatory problem.
                  Treatment involves compression stockings, exercise or procedures to close or remove the veins. you can consult with a general surgeon.''')

        speak(f'''Varicose veins are twisted, Gnarled, enlarged veins, most commonly appearing in the legs and feet.
                  Varicose veins are generally benign. The cause of this condition is not known.
                  For many people, there are no symptoms and varicose veins are simply a cosmetic concern. 
                  In some cases, they cause aching pain and discomfort or signal an underlying circulatory problem.
                  Treatment involves compression stockings, exercise or procedures to close or remove the veins. you can consult with a general surgeon.''')

    elif 'hypothyroidism' in prediction:
        print(f'''Hypothyroidism is also called as underactive thyroid. Hypothyroidism results when the thyroid gland fails to produce enough hormones. 
                  Hypothyroidism may be due to a number of factors, including: Autoimmune disease. 
                  The most common cause of hypothyroidism is an autoimmune disorder known as Hashimoto's thyroiditis. 
                  it disrupts things such as heart rate, body temperature and all aspects of metabolism. Hypothyroidism is most prevalent in older women.
                  Major symptoms include fatigue, cold sensitivity, constipation, dry skin and unexplained weight gain.
                  Treatment consists of thyroid hormone replacement. you can consult with an endocrinologist.''')

        speak(f'''Hypothyroidism is also called as underactive thyroid. Hypothyroidism results when the thyroid gland fails to produce enough hormones. 
                  Hypothyroidism may be due to a number of factors, including: Autoimmune disease. 
                  The most common cause of hypothyroidism is an autoimmune disorder known as Hashimoto's thyroiditis. 
                  it disrupts things such as heart rate, body temperature and all aspects of metabolism. Hypothyroidism is most prevalent in older women.
                  Major symptoms include fatigue, cold sensitivity, constipation, dry skin and unexplained weight gain.
                  Treatment consists of thyroid hormone replacement. you can consult with an endocrinologist.''')

    elif 'hyperthyroidism' in prediction:
        print(f'''Hyperthyroidism is also called as overactive thyroid. Hyperthyroidism is the production of too much thyroxine hormone. 
                  It can increase metabolism.Symptoms include unexpected weight loss, rapid or irregular heartbeat, sweating and irritability, 
                  although the elderly often experience no symptoms.Treatments include radioactive iodine, medication and sometimes surgery.
                  you can consult with an endocrinologist.''')

        speak(f'''Hyperthyroidism is also called as overactive thyroid. Hyperthyroidism is the production of too much thyroxine hormone. 
                  It can increase metabolism.Symptoms include unexpected weight loss, rapid or irregular heartbeat, sweating and irritability, 
                  although the elderly often experience no symptoms.Treatments include radioactive iodine, medication and sometimes surgery.
                  you can consult with an endocrinologist.''')

    elif 'hypoglycemia' in prediction:
        print(f'''Hypoglycemia is also called as low blood sugar. Hypoglycemia is a condition in which your blood sugar (glucose) level is lower than the standard range. 
                  Diabetes treatment and other conditions can cause hypoglycaemia. Confusion, heart palpitations, shakiness and anxiety are symptoms.
                  Consuming high-sugar foods or drinks, such as orange juice or regular fizzy drinks, can treat this condition. 
                  Alternatively, medication can be used to raise blood sugar levels. 
                  It's also important that a doctor identifies and treats the underlying cause. you can consult with a endocrinologist.''')

        speak(f'''Hypoglycemia is also called as low blood sugar. Hypoglycemia is a condition in which your blood sugar (glucose) level is lower than the standard range. 
                  Diabetes treatment and other conditions can cause hypoglycaemia. Confusion, heart palpitations, shakiness and anxiety are symptoms.
                  Consuming high-sugar foods or drinks, such as orange juice or regular fizzy drinks, can treat this condition. 
                  Alternatively, medication can be used to raise blood sugar levels. 
                  It's also important that a doctor identifies and treats the underlying cause. you can consult with a endocrinologist.''')

    elif 'osteoarthritis' in prediction:
        print(f'''Osteoarthritis is the most common form of arthritis. 
                  The wearing down of the protective tissue at the ends of bones (cartilage) occurs gradually and worsens over time. 
                  Although osteoarthritis can damage any joint, the disorder most commonly affects joints in your hands, knees, hips and spine. 
                  joint pain in the hands, neck, lower back, knees or hips is the most common symptom.
                  Medication, physiotherapy and sometimes surgery can help reduce pain and maintain joint movement. you can consult with a rheumatologist.''')

        speak(f'''Osteoarthritis is the most common form of arthritis. 
                  The wearing down of the protective tissue at the ends of bones (cartilage) occurs gradually and worsens over time. 
                  Although osteoarthritis can damage any joint, the disorder most commonly affects joints in your hands, knees, hips and spine. 
                  joint pain in the hands, neck, lower back, knees or hips is the most common symptom.
                  Medication, physiotherapy and sometimes surgery can help reduce pain and maintain joint movement. you can consult with a rheumatologist.''')

    elif 'arthritis' in prediction:
        print(f'''Arthritis is Inflammation of one or more joints, causing pain and stiffness that can worsen with age.
                  Different types of arthritis exist, each with different causes including wear and tear, infections and underlying diseases.
                  Symptoms include pain, swelling, reduced range of motion and stiffness.
                  Medication, physiotherapy or sometimes surgery helps reduce symptoms and improve quality of life. you can consult with rheumatologist and physiotherapist.''')

        speak(f'''Arthritis is Inflammation of one or more joints, causing pain and stiffness that can worsen with age.
                  Different types of arthritis exist, each with different causes including wear and tear, infections and underlying diseases.
                  Symptoms include pain, swelling, reduced range of motion and stiffness.
                  Medication, physiotherapy or sometimes surgery helps reduce symptoms and improve quality of life. you can consult with rheumatologist and physiotherapist.''')

    elif '(vertigo) paroymsal positional vertigo' in prediction:
        print(f'''Benign paroxysmal positional vertigo (BPPV) is one of the most common causes of vertigo — 
                  the sudden sensation that you're spinning or that the inside of your head is spinning. BPPV causes brief episodes of mild to intense dizziness. 
                  It is usually triggered by specific changes in your head's position such as tipping the head up or down. 
                  It's rarely serious unless it increases the risk of falling. People can experience dizziness, a spinning sensation (vertigo), lightheadedness, 
                  unsteadiness, loss of balance and nausea. Treatment includes a series of head movements that shift particles in the ears. 
                  you can consult with a neurologist or otolaryngologist.''')

        speak(f'''Benign paroxysmal positional vertigo (BPPV) is one of the most common causes of vertigo — 
                  the sudden sensation that you're spinning or that the inside of your head is spinning. BPPV causes brief episodes of mild to intense dizziness. 
                  It is usually triggered by specific changes in your head's position such as tipping the head up or down. 
                  It's rarely serious unless it increases the risk of falling. People can experience dizziness, a spinning sensation (vertigo), lightheadedness, 
                  unsteadiness, loss of balance and nausea. Treatment includes a series of head movements that shift particles in the ears. 
                  you can consult with a neurologist or otolaryngologist.''')

    elif 'acne' in prediction:
        print(f'''Acne is a skin condition that occurs when hair follicles plug with oil and dead skin cells.
                  Acne is most common in teenagers and young adults.
                  Symptoms range from uninflamed blackheads to pus-filled pimples or large, red and tender bumps.
                  Treatments include over-the-counter creams and cleanser, as well as prescription antibiotics. you can consult with a dermatologist.''')

        speak(f'''Acne is a skin condition that occurs when hair follicles plug with oil and dead skin cells.
                  Acne is most common in teenagers and young adults.
                  Symptoms range from uninflamed blackheads to pus-filled pimples or large, red and tender bumps.
                  Treatments include over-the-counter creams and cleanser, as well as prescription antibiotics. you can consult with a dermatologist.''')

    elif 'urinary tract infecetion' in prediction:
        print(f'''A urinary tract infection (UTI) is an infection in any part of the urinary system. the kidneys, bladder or urethra.
                  Urinary tract infections are more common in women. They usually occur in the bladder or urethra, but more serious infections involve the kidney.
                  A bladder infection may cause pelvic pain, increased urge to urinate, pain with urination and blood in the urine. 
                  A kidney infection may cause back pain, nausea, vomiting and fever. Common treatment is with antibiotics. you can consult with a urologist.''')

        speak(f'''A urinary tract infection (UTI) is an infection in any part of the urinary system. the kidneys, bladder or urethra.
                  Urinary tract infections are more common in women. They usually occur in the bladder or urethra, but more serious infections involve the kidney.
                  A bladder infection may cause pelvic pain, increased urge to urinate, pain with urination and blood in the urine. 
                  A kidney infection may cause back pain, nausea, vomiting and fever. Common treatment is with antibiotics. you can consult with a urologist.''')

    elif 'psoriasis' in prediction:
        print(f'''Psoriasis is a skin disease that causes a rash with itchy, scaly patches, most commonly on the knees, elbows, trunk and scalp. 
                  Psoriasis is a common, long-term (chronic) disease with no cure. It can be painful, interfere with sleep and make it hard to concentrate.
                  Treatment aims to remove scales and stop skin cells from growing so quickly. Topical ointments, light therapy and medication can offer relief. 
                  you can consult with a dermatologist who has experience treating psoriasis.''')

        speak(f'''Psoriasis is a skin disease that causes a rash with itchy, scaly patches, most commonly on the knees, elbows, trunk and scalp. 
                  Psoriasis is a common, long-term (chronic) disease with no cure. It can be painful, interfere with sleep and make it hard to concentrate.
                  Treatment aims to remove scales and stop skin cells from growing so quickly. Topical ointments, light therapy and medication can offer relief. 
                  you can consult with a dermatologist who has experience treating psoriasis.''')

    elif 'impetigo' in prediction:
        print(f'''Impetigo is a common and highly contagious skin infection that mainly affects infants and young children.
                  It usually appears as reddish sores on the face, especially around the nose and mouth and on the hands and feet. 
                  Over about a week, the sores burst and develop honey-colored crusts. Antibiotics shorten the infection and can help prevent spread to others. 
                  you can consult with a dermatologist.''')

        speak(f'''Impetigo is a common and highly contagious skin infection that mainly affects infants and young children.
                  It usually appears as reddish sores on the face, especially around the nose and mouth and on the hands and feet. 
                  Over about a week, the sores burst and develop honey-colored crusts. Antibiotics shorten the infection and can help prevent spread to others. 
                  you can consult with a dermatologist.''')

    else:
        speak("out of my knowledge, sorry!")
        print("No disease matched!!")




def booking_details_email():
        def booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1):
            # connecting tot the mongodb atlas 
            cluster = pymongo.MongoClient("mongodb+srv://shrushti:mBWnaLmIvjQZfzo2@cluster-test.yceejon.mongodb.net/?retryWrites=true&w=majority")
            db = cluster.test
            db = cluster["Booking_Appointments"]
            collection = db["Appointment_details"]

            port = 465
            sender_email = "healthcare.assistant8@gmail.com"
            password = "vfxsrrcqtmceguza"#getpass("Type your password and press enter : ")
            #receiver_email = [input("Enter your email here-> : ")] #"shrushtidesai02@gmail.com"  
            query = takeCommand().lower()
            if(len(query)==0):
                query = takeCommand().lower()

            if 'yes' in query or 'ok' in query or 'sure' in query or 'yah' in query:
                i=i+1
                speak("Tell me the patient name")
                patient_name = takeCommand().lower()
                if(len(patient_name)==0):
                    patient_name = takeCommand().lower()

                speak("Enter your email id to recieve booking and doctor details")
                #receiver_email = "shrushtidesai02@gmail.com" #[input("Enter your email address : ")]
                date = datetime.datetime.now()
                collection.insert_one({"id":(str(i)), "patient_name":(patient_name),"DRname":(dr_name), "Date":(date), "DRtype": (dr_type)})

                receiver_email =  input("Enter your email here-> : ")  #"shrushtidesai02@gmail.com"     #[input("Enter your email here-> : ")]

                message = MIMEMultipart("alternative")
                message["Subject"] = f'''Doctor and booking details {(dr_type)}'''
                message["From"] = sender_email
                message["To"] = receiver_email

                # Create     the plain-text and HTML version of your message
                text = f"""\
                Details of the doctor
                Doctor name : {(dr_name)}
                Info : {(dr_info)}
                Address : {(dr_add)}
                Contact : {(dr_contact)}
                link for booking appointment:-
                {(link1)}"""
                html = f"""\
                <html>  
                <body>    
                <h3>DOCTOR DETAILS</h3>
                <h4>Doctor name :</h4> {(dr_name)}
                <br>
                <h4>Info :</h4> {(dr_info)}
                <br>
                <h4>Address :</h4> {(dr_add)}
                <br>
                <h4>contact :</h4> {(dr_contact)}
                <br>
                <h3>Link for booking appointment</h3>
                <h4>link :</h4> click on this -> <a href="{(link1)}">Book</a>
                </p>    
                </body>
                </html>     
                """     

# Turn t    hese into plain/html MIMEText objects
                part1 = MIMEText(text, "plain")
                part2 = MIMEText(html, "html")

# Add HT    ML/plain-text parts to MIMEMultipart message
# The em    ail client will try to render the last part first
                message.attach(part1)
                message.attach(part2)

# Create     secure connection with server and send email
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                    server.login(sender_email, password)
                    server.sendmail(
                    sender_email, receiver_email, message.as_string()
                             )
                speak("email has been send!")
                print("done")

            else:
                speak("ok")
    
        def doctor():
        
            speak("Tell me what are your problems")

            while True:
                query = takeCommand().lower()

        #done
                if 'allergy' in query or 'allergist' in query:
                    speak("I think you need to go for a Allergist")
                    print("I think you need to go for a Allergist")
                    speak("They treat immune system disorders such as asthma, eczema, food allergies, insect sting allergies, and some autoimmune diseases.")
                    print("They treat immune system disorders such as asthma, eczema, food allergies, insect sting allergies, and some autoimmune diseases.")
                    speak("You can contact Doctor Sarika Verma")
                    print("You can contact Dr Sarika Verma")
                    speak("Below are the address and contact number of Doctor Sarika Verma")
                    print("Address: Shree Sai Clinic and Parvatibai Shankarrao Chavan Hospital and Research Centre,Near Patkar College, Off S.V .Road, Goregaon West, Mumbai, Maharashtra 400104")
                    print("Contact: 020 7117 3187 Ext. 312")

                    dr_type = "Allergist"
                    dr_name = "Dr. Sarika Verma"
                    dr_add = "Shree Sai Clinic and Parvatibai Shankarrao Chavan Hospital and Research Centre,Near Patkar College, Off S.V .Road, Goregaon West, Mumbai, Maharashtra 400104"
                    dr_contact = "020 7117 3187 Ext. 312"
                    dr_info =f'''MBBS, DNB - ENT
                                 ENT/ Otorhinolaryngologist, Pediatric Otorhinolaryngologist, Otologist/ Neurotologist
                                 23 Years Experience Overall  (15 years as specialist)  
                                 Dr. Sarika Verma is an ENT Surgeon and Allergy Specialist'''
                    link1 = "https://www.practo.com/gurgaon/doctor/dr-sarika-verma-ear-nose-throat-ent-specialist-2" 

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
        #done
                elif 'heart' in query or 'cardiac' in query or 'cardiologist' in query:
                    speak("I think you need to go for a Cardiologist")
                    print("I think you need to go for a Cardiologist")
                    speak("You might see them for heart failure, a heart attack high blood pressure or an irregular heartbeat")
                    print("You might see them for heart failure, a heart attack, high blood pressure or an irregular heartbeat.")
                    speak("You can contact Doctor Kamales Kumar Saha")
                    print("You can contact Dr Kamales Kumar Saha")
                    speak("Below are the address and contact details of Doctor Kamales Kumar Saha")
                    print("Address: HeartClinic 5 accord classic 6 th floor near station, anupam stationary building, Goregaon, Mumbai, Maharashtra 400063")
                    print("Contact: 099773 45555")

                    dr_type = "Cardiologist"
                    dr_name = "Dr. Kamales Kumar Saha"
                    dr_add = "HeartClinic 5 accord classic 6 th floor near station, anupam stationary building, Goregaon, Mumbai, Maharashtra 400063"
                    dr_contact = "099773 45555"
                    dr_info = f'''36 Years Experience Overall  (27 years as specialist) 
                                 Cardiothoracic Vascular Surgery, Cardiologist'''  
                    link1 = "https://www.practo.com/mumbai/doctor/kamales-kumar-saha-cardiac-surgeon"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
          #done         
                elif 'small Intestine' in query or 'colon' in query or 'bottom' in query or 'colon rectal specialist' in query or 'proctologist' in query:
                    speak("I think you need to go for a Proctologist also know as Colon and Rectal Surgeons")
                    print("I think you need to go for a Proctologist also know as Colon and Rectal Surgeons")
                    speak("You would see these doctors for problems with your small intestine,colon, and bottom. They can treat colon cancer, hemorrhoids, and inflammatory bowel disease")
                    print("You would see these doctors for problems with your small intestine,colon, and bottom. They can treat colon cancer, hemorrhoids, and inflammatory bowel disease")
                    speak("You can contact Doctor Jiten Chowdhry")
                    print("You can contact Dr. Jiten Chowdhry")
                    speak("Below are the address and contact details of Doctor Jiten Chowdhry")
                    print("Address: Arunodaya Tower, Jangal Mangal Rd, Bhandup, Kokan Nagar, Bhandup West, Mumbai, Maharashtra 400078")
                    print("Contact: +919867333568")

                    dr_type = "Proctologiat"
                    dr_name = "Dr. Jiten Chowdhry"
                    dr_add = "Arunodaya Tower, Jangal Mangal Rd, Bhandup, Kokan Nagar, Bhandup West, Mumbai, Maharashtra 400078"
                    dr_contact = "+919867333568"
                    dr_info =f'''General Surgeon, Laparoscopic Surgeon, Proctologist
                                 29 Years Experience Overall  (23 years as specialist)
                                 MS, FAIS, FMAS, FISC (USA), Dip.Lap. Surgery (SAGES), (FISC) Fellowship International Society of Coloproctology
                                 Surgical Experience of 24 Years''' 
                    link1 = "https://www.drjitenchowdhry.com/"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
                
        #done
                elif 'ill' in query or 'injured' in query or 'critical care' in query or 'critical care specialist' in query:
                    speak("I think you need to go for a Critical care medicine specialist")
                    print("I think you need to go for a Critical care medicine specialist")
                    speak("They care for people who are critically ill or injured, often heading intensive care units in hospitals. You might see them if your heart or other organs are failing or if you’ve been in an accident.")
                    print("They care for people who are critically ill or injured, often heading intensive care units in hospitals. You might see them if your heart or other organs are failing or if you’ve been in an accident.")
                    speak("You can contact Doctor Vatsal Kothari")
                    print("You can contact Dr. Vatsal Kothari")
                    speak("Below are the address and contact number of Doctor Vatsal Kothari")
                    print("Address: Kokilaben Hospital, Rao Saheb, Achutrao Patwardhan Marg, Four Bungalows, Andheri West, Mumbai, Maharashtra 400053")
                    print("Contact: +91 20485 52994 Ext. 160")

                    dr_type = "Intensive Medicicne Specialist"
                    dr_name = "Dr. Vatsal Kothari"
                    dr_add = "Kokilaben Hospital, Rao Saheb, Achutrao Patwardhan Marg, Four Bungalows, Andheri West, Mumbai, Maharashtra 400053"
                    dr_contact = "+91 20485 52994 Ext. 160"
                    dr_info = f'''Dr. Vatsal Kothari is a Intensive Medicine Specialist and Internal Medicine
                                  MBBS, MD - General Medicine
                                  Internal Medicine
                                  27 Years Experience Overall (26 years as specialist)'''
                    link1 = "https://www.kokilabenhospital.com/professionals/vatsalkothari.html"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
        #done
                elif 'skin' in query or 'dermatologist' in query:
                    speak("I think you need to go for a Dermatologists")
                    print("I think you need to go for a Dermatologists")
                    speak("They care for people who Have problems with your skin, hair, nails. Do you have moles, scars, acne, or skin allergies? Dermatologists can help.")
                    print("They care for people who Have problems with your skin, hair, nails. Do you have moles, scars, acne, or skin allergies? Dermatologists can help.")
                    speak("You can contact Doctor Jaishree Sharad")
                    print("You can contact Dr. Jaishree Sharad")
                    speak("Below are the address and contact number of Doctor Jaishree Sharad ")
                    print("Address: Skinfiniti, Cosmetic Dermatology Clinic G 62, Satra Plaza, Sector 19D, Palm Beach Road,  Vashi, Navi Mumbai Landmark: Behind Mercedes Showroom")
                    print("Contact: 022 4893 2701 Ext. 369")

                    dr_type = "Dermatologist"
                    dr_name = "Dr. Jaishree Sharad"
                    dr_add = "Skinfiniti, Cosmetic Dermatology Clinic G 62, Satra Plaza, Sector 19D, Palm Beach Road,  Vashi, Navi Mumbai Landmark: Behind Mercedes Showroom"
                    dr_contact = "022 4893 2701 Ext. 369"
                    dr_info = f'''MBBS, Diploma in Dermatology
                                  Therapist Cosmetologist Dermatologist
                                  23 Years Experience Overall (20 years as specialist)'''
                    link1 = "https://www.practo.com/navi-mumbai/doctor/dr-jaishree-sharad-vashi-1-dermatologist"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
        #done
                elif 'hormones' in query or 'metabolism' in query or 'endocrinologist' in query:
                    speak("I think you need to go for a Endocrinologists")
                    print("I think you need to go for a Endocrinologists")
                    speak("These are experts on hormones and metabolism. They can treat conditions like diabetes, thyroid problems, infertility, and calcium and bone disorders.")
                    print("These are experts on hormones and metabolism. They can treat conditions like diabetes, thyroid problems, infertility, and calcium and bone disorders.")
                    speak("You can contact Doctor Benny Negalur")
                    print("You can contact Dr. Benny Negalur")
                    speak("Below are the address and contact number of Doctor Benny Negalur")
                    print("Address: Dr. Negalur's Diabetes & Thyroid Specialities Center, 2nd Floor, Swastik High-Point, Gloria Apartments, Thane West, Thane, Maharashtra 400615 Landmark: Opposite Harmony Residences")
                    print("Contact: 020 6732 5118 Ext. 525")


                    dr_type = "Endocrinologists"
                    dr_name = "Dr. Benny Negalur"
                    dr_add = "Dr. Negalur's Diabetes & Thyroid Specialities Center, 2nd Floor, Swastik High-Point, Gloria Apartments, Thane West, Thane, Maharashtra 400615 Landmark: Opposite Harmony Residences"
                    dr_contact = "020 6732 5118 Ext. 525"
                    dr_info = f'''MBBS, MD - Medicine, Fellowship in Diabetes (UK)
                                  Consultant Physician
                                  38 Years Experience Overall'''
                    link1 = "https://www.practo.com/thane/doctor/dr-benny-negalur-diabetologist"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
        #done
                elif 'digestive organs' in query or 'gastroenterologist' in query:
                    speak("I think you need to go for a Gastroenterologists")
                    print("I think you need to go for a Gastroenterologists")
                    speak("They are specialists in digestive organs, including the stomach, bowels, pancreas, liver, and gallbladder. You might see them for abdominal pain, ulcers, diarrhea, jaundice, or cancers in your digestive organs.")
                    print("They are specialists in digestive organs, including the stomach, bowels, pancreas, liver, and gallbladder. You might see them for abdominal pain, ulcers, diarrhea, jaundice, or cancers in your digestive organs.")
                    speak("You can contact Doctor Keyur A Sheth")
                    print("You can contact Dr.Keyur A Sheth")
                    print("Below are the address and contact number of Doctor Keyur A Sheth ")
                    speak("Below are the address and contact number of Doctor")
                    print("Address: Ameeta Nursing Home 1st Floor, Ramgiri Building, Chembur East, Mumbai, Maharashtra 400071 Landmark: Opposite Natraj Cinema.")
                    print("Contact: 020 7117 3182 Ext. 100")


                    dr_type = "Gastroenterologists"
                    dr_name = "Dr. Keyur A Sheth"
                    dr_add = "Ameeta Nursing Home 1st Floor, Ramgiri Building, Chembur East, Mumbai, Maharashtra 400071 Landmark: Opposite Natraj Cinema. "
                    dr_contact = "020 7117 3182 Ext. 100"
                    dr_info = f'''MBBS, DNB - General Medicine, DNB - Gastroenterology, CCST - Gastroenterology
                                  Gastroenterologist
                                  20 Years Experience Overall (9 years as specialist)'''
                    link1 = "https://www.justdial.com/Mumbai/Dr-Keyur-A-Sheth-Opposite-Natraj-Cinema-and-Chembur-Station-Chembur-East/022PXX22-XX22-160202130634-L4J1_BZDET"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
        #done
                elif 'routine checkups' in query or 'physician' in query:
                    speak("I think you need to go for a Physician")
                    print("I think you need to go for a Physician")
                    speak("They care for the whole family, including children, adults, and the elderly. They do routine checkups and screening tests, give you flu and immunization shots, and manage diabetes and other ongoing medical conditions.")
                    print("They care for the whole family, including children, adults, and the elderly. They do routine checkups and screening tests, give you flu and immunization shots, and manage diabetes and other ongoing medical conditions.")
                    speak("You can contact Doctor Mili S. Joshi")
                    print("You can contact Dr. Mili S. Joshi")
                    speak("Below are the address and contact number of Doctor Mili S. Joshi")
                    print("Address: Dr. Mili Joshi's Clinic, 4, Ulacon CHS Limited, St. Anthony's Road, Vakola,  Santacruz East, Mumbai, Maharashtra 400055 Landmark: Near Charle's School")
                    print("Contact: 020 7117 9000 Ext. 322")


                    dr_type = "Physician"
                    dr_name = "Dr. Mili S. Joshi"
                    dr_add = "Dr. Mili Joshi's Clinic, 4, Ulacon CHS Limited, St. Anthony's Road, Vakola,  Santacruz East, Mumbai, Maharashtra 400055 Landmark: Near Charle's School"
                    dr_contact = "020 7117 9000 Ext. 322"
                    dr_info = f'''MBBS, Diploma in Child Health (DCH)
                                  Pediatrician, General Physician
                                  25 Years Experience Overall  (20 years as specialist)
                                  Dr. Mili Joshi is an experienced Pediatrician and General Physician with an experience of 19 years'''
                    link1 = "https://www.practo.com/mumbai/doctor/dr-mili-s-joshi-pediatrician-pediatrician"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
                
                
        #done
                elif 'blood' in query or 'spleen' in query or 'lymph glands' in query or 'hematologist' in query:
                    speak("I think you need to go for a Hematologists")
                    print("I think you need to go for a Hematologists")
                    speak("These are specialists in diseases of the blood, spleen, and lymph glands, like sickle cell disease, anemia, hemophilia, and leukemia.")
                    print("These are specialists in diseases of the blood, spleen, and lymph glands, like sickle cell disease, anemia, hemophilia, and leukemia.")
                    speak("You can contact Doctor Samir Shah")
                    print("You can contact Dr. Samir Shah")
                    speak("Below are the address and contact number of Doctor Samir Shah")
                    print("Address: Jaslok Hospital,15, Dr. G.Deshmukh Marg, Pedder Road, Mumbai, Maharashtra 400026. Landmark: Near Mahalakshmi Temple")
                    print("Contact: 020 4856 6752 Ext. 570")

                    dr_type = "Hematologists"
                    dr_name = "Dr. Samir Shah"
                    dr_add = "Jaslok Hospital,15, Dr. G.Deshmukh Marg, Pedder Road, Mumbai, Maharashtra 400026. Landmark: Near Mahalakshmi Temple"
                    dr_contact = "020 4856 6752 Ext. 570"
                    dr_info = f'''Pediatric Hematologist,
                                  M.B.B.S, MD ( GENERAL MEDICINE) MRCP (GENERAL MEDICINE), MRCPath (Haematology), U.K
                                  20 years of experience'''
                    link1 = "https://www.bhatiahospital.org/doctors/bookappointment/t?doctors=dr-samir-shah"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
        #done
                elif 'infections' in query or 'infectious disease' in query:
                    speak("I think you need to go for a Infectious Disease Specialists")
                    print("I think you need to go for a Infectious Disease Specialists")
                    speak("They diagnose and treat infections in any part of your body, like fevers, Lyme disease, pneumonia, tuberculosis, and HIV and AIDS.")
                    print("They diagnose and treat infections in any part of your body, like fevers, Lyme disease, pneumonia, tuberculosis, and HIV and AIDS.")
                    speak("You can contact Doctor Vidyullata Koparkar")
                    print("You can contact Dr. Vidyullata Koparkar")
                    speak("Below are the address and contact number of Doctor Vidyullata Koparkar")
                    print("Address: 5, Shroff Bungalow, Opp Pandya Hospital, near Sodawala Municipal School, Sodawala Ln, Borivali West, Mumbai, Maharashtra 400092")
                    print("Contact: +91 090762 36902")

                    dr_type = "Infectious Disease Specialist"
                    dr_name = "Dr. Vidyullata Koparkar"
                    dr_add = "5, Shroff Bungalow, Opp Pandya Hospital, near Sodawala Municipal School, Sodawala Ln, Borivali West, Mumbai, Maharashtra 400092"
                    dr_contact = "+91 090762 36902"          
                    dr_info = f'''FNB - Infectious Disease, MBBS, DNB - General Medicine
                                  Infectious Diseases Physician, General Physician
                                  11 Years Experience Overall  (5 years as specialist)'''
                    link1 = "https://www.justdial.com/Mumbai/Dr-Vidyullata-Koparkar-Above-Shara-Sagar-Hotel-Borivali-West/022PXX22-XX22-191116090927-B2Q1_BZDET"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i)
                    break
                
        #done
                elif 'heridatary disorders' in query or 'genes' in query or 'genetics' in query:
                    speak("I think you need to go for a Medical Geneticists")
                    print("I think you need to go for a Medical Geneticists")
                    speak("They diagnose and treat hereditary disorders passed down from parents to children. These doctors may also offer genetic counseling and screening tests.")
                    print("They diagnose and treat hereditary disorders passed down from parents to children. These doctors may also offer genetic counseling and screening tests.")
                    speak("You can contact Doctor Shruti Bajaj")
                    print("You can contact Dr. Shruti Bajaj")
                    speak("Below are the address and contact number of Doctor Shruti Bajaj")
                    print("Address: Suchak Hospital, 186, Manchubhai Road, Opposite, Malad Subway, Malad East, Mumbai, Maharashtra 400097")
                    print("Contact: 091360 17545")

                    dr_type = "Medical Geneticists"
                    dr_name = "Dr. Shruti Bajaj"
                    dr_add = "Suchak Hospital, 186, Manchubhai Road, Opposite, Malad Subway, Malad East, Mumbai, Maharashtra 400097"
                    dr_contact = "091360 17545"
                    dr_info = f'''MD Pediatrics, Fellowship in Clinical Genetics (MUHS, KEM Hospital, Mumbai)
                                   Founder & Director: The Purple Gene Clinic'''
                    link1 = "https://www.geneticsinindia.com/askthedoctor.php"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
        #done
                elif 'kidney' in query or 'high blood pressure' in query or 'nephrologist' in query:
                    speak("I think you need to go for a Nephrologists")
                    print("I think you need to go for a Nephrologists")
                    speak("They treat kidney diseases as well as high blood pressure and fluid and mineral imbalances linked to kidney disease.")
                    print("They treat kidney diseases as well as high blood pressure and fluid and mineral imbalances linked to kidney disease.")
                    speak("You can contact Doctor Sharad Sheth")
                    print("You can contact Dr.Sharad Sheth")
                    speak("Below are the address and contact number of Doctor Sharad Sheth")
                    print("Address: Mallika Hospital, No. 52, Sharma Estate, Swami Vivekanand Road, Collectors Colony, Jogeshwari West, Mumbai, Maharashtra 400102. Landmark: Next To Dewan Shopping Centre & Next To Mina Hotel.")
                    print("Contact: 020 7117 3188 Ext. 012")

                    dr_type = "Nephrologist"
                    dr_name = "Dr. Sharad Sheth"
                    dr_add = "Mallika Hospital, No. 52, Sharma Estate, Swami Vivekanand Road, Collectors Colony, Jogeshwari West, Mumbai, Maharashtra 400102. Landmark: Next To Dewan Shopping Centre & Next To Mina Hotel."
                    dr_contact = "020 7117 3188 Ext. 012"
                    dr_info = f'''MBBS, MD - Medicine, MNAMS - Nephrology
                                  Nephrologist/Renal Specialist
                                  46 Years Experience Overall  (39 years as specialist)'''
                    link1 = "https://www.practo.com/mumbai/doctor/dr-sharad-sheth-nephrologist"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
        #done
                elif 'brain' in query or 'spinal Cord' in query or 'nerves' in query or 'neurologist' in query:
                    speak("I think you need to go for a Neurologists")
                    print("I think you need to go for a Neurologists")
                    speak("These are specialists in the nervous system, which includes the brain, spinal cord, and nerves. They treat strokes, brain and spinal tumors, epilepsy, Parkinson's disease, and Alzheimer's disease.")
                    print("These are specialists in the nervous system, which includes the brain, spinal cord, and nerves. They treat strokes, brain and spinal tumors, epilepsy, Parkinson's disease, and Alzheimer's disease.")
                    speak("You can contact Doctor Pravina Ushakant Shah")
                    print("You can contact Dr. Pravina Ushakant Shah")
                    speak("Below are the address and contact number of Doctor Pravina Ushakant Shah")
                    print("Address: Fortis Hospital Mulund, Mulund - Goregaon Link Rd, Nahur West, Industrial Area, Bhandup West, Mumbai, Maharashtra 400078.")
                    print("Contact: 020 7117 7302 Ext. 472")

                    dr_type = "Neurologist"
                    dr_name = "Dr. Pravina Ushakant Shah"
                    dr_add = "Fortis Hospital Mulund, Mulund - Goregaon Link Rd, Nahur West, Industrial Area, Bhandup West, Mumbai, Maharashtra 400078."
                    dr_contact = "020 7117 7302 Ext. 472"
                    dr_info = f'''MBBS, MD - General Medicine, DM - Neurology
                                  Neurologist
                                  47 Years Experience Overall  (45 years as specialist)'''
                    link1 = "https://www.practo.com/mumbai/doctor/dr-pravina-ushakant-shah-neurologist"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
         #done           
                elif 'pregnancy' in query or 'childbirth' in query or 'gynecologist' in query or 'obstetrician' in query:
                    speak("I think you need to go for a Obstetricians and Gynecologists")
                    print("I think you need to go for a Obstetricians and Gynecologists")
                    speak("Often called OB/GYNs, these doctors focus on women's health, including pregnancy and childbirth. They do Pap smears, pelvic exams, and pregnancy checkups. OB/GYNs are trained in both areas. But some of them may focus on women's reproductive health (gynecologists), and others specialize in caring for pregnant women (obstetricians).")
                    print("Often called OB/GYNs, these doctors focus on women's health, including pregnancy and childbirth. They do Pap smears, pelvic exams, and pregnancy checkups. OB/GYNs are trained in both areas. But some of them may focus on women's reproductive health (gynecologists), and others specialize in caring for pregnant women (obstetricians).")
                    speak("You can contact Doctor Mohini Vachhani")
                    print("You can contact Dr. Mohini Vachhani")
                    speak("Below are the address and contact number of Doctor Mohini Vachhani")
                    print("Address: In Sync Gynae Care, 302, Vinayak Chambers, 4th Road, Khar West, Mumbai, Landmark: Near Kabutarkhana & Opp Amor Building.")
                    print("Contact: 020 7117 3190 Ext. 861")

                    dr_type = "Gynecologist"
                    dr_name = "Dr. Mohini Vachhani"
                    dr_add = "In Sync Gynae Care, 302, Vinayak Chambers, 4th Road, Khar West, Mumbai, Landmark: Near Kabutarkhana & Opp Amor Building."
                    dr_contact = "020 7117 3190 Ext. 861"
                    dr_info = f'''Fellow of Royal College of Obstetricians and Gynaecologists FRCOG (London), MRCOG(UK), 
                                  MD - Obstetrics & Gynaecology, DGO
                                  Gynecologist, Obstetrician'''
                    link1 = "https://www.practo.com/mumbai/doctor/dr-mohini-vachhani-gynecologist-obstetrician"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break

        #done
                elif 'cancer' in query or 'oncologist' in query or 'chemo' in query:
                    speak("I think you need to go for a Oncologists")
                    print("I think you need to go for a Oncologists")
                    speak("These internists are cancer specialists. They do chemotherapy treatments and often work with radiation oncologists and surgeons to care for someone with cancer.")
                    print("These internists are cancer specialists. They do chemotherapy treatments and often work with radiation oncologists and surgeons to care for someone with cancer.")
                    speak("You can contact Doctor Suresh Advani")
                    print("You can contact Dr. Suresh Advani")
                    speak("Below are the address and contact number of Doctor Suresh Advani")
                    print("Address: International Oncology (Oncology Clinic), Hillside Road, Hiranandani Gardens, IIT Area, Powai, Mumbai, Maharashtra 400076.")
                    print("Contact: 020 7118 8518 Ext. 541")

                    dr_type = "Oncologist"
                    dr_name = "Dr. Suresh Advani"
                    dr_add = "International Oncology (Oncology Clinic), Hillside Road, Hiranandani Gardens, IIT Area, Powai, Mumbai, Maharashtra 400076."
                    dr_contact = "020 7118 8518 Ext. 541"
                    dr_info = f'''MBBS, DM - Oncology
                                  Medical Oncologist
                                  49 Years Experience Overall'''
                    link1 = "https://drsureshadvani.in/make-appointment/"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break
                
        #done
                elif 'eye' in query or 'eyes' in query or 'opthalmologist' in query:
                    speak("I think you need to go for a Ophthalmologists")
                    print("I think you need to go for a Ophthalmologists")
                    speak("You call them eye doctors. They can prescribe glasses or contact lenses and diagnose and treat diseases like glaucoma. Unlike optometrists, they’re medical doctors who can treat every kind of eye condition as well as operate on the eyes.")
                    print("You call them eye doctors. They can prescribe glasses or contact lenses and diagnose and treat diseases like glaucoma. Unlike optometrists, they’re medical doctors who can treat every kind of eye condition as well as operate on the eyes.")
                    speak("You can contact Doctor Chinmaya Sahu")
                    print("You can contact Dr. Chinmaya Sahu")
                    speak("Below are the address and contact number of Doctor Chinmaya Sahu")
                    print("Address: Sahu Eye Hospital, 1st Floor, Almar Arcade , Orlem, Marve Road, Malad West, Mumbai, Maharashtra 400064. Landmark: Above Punjab National Bank.")
                    print("Contact: 020 7117 7422 Ext. 039")

                    dr_type = "Ophthalmologist"
                    dr_name = "Dr. Chinmaya Sahu"
                    dr_add = "Sahu Eye Hospital, 1st Floor, Almar Arcade , Orlem, Marve Road, Malad West, Mumbai, Maharashtra 400064. Landmark: Above Punjab National Bank."
                    dr_contact = "020 7117 7422 Ext. 039"            
                    dr_info = f'''MBBS, MS - Ophthalmology, DOMS
                                  Ophthalmologist/ Eye Surgeon
                                  19 Years Experience Overall  (14 years as specialist)'''
                    link1 = "https://www.practo.com/mumbai/doctor/dr-chinmaya-sahu-ophthalmologist-ophthalmologist"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break            
                
        #done
                elif 'ears' in query or 'nose' in query or 'throat' in query or 'sinuses' in query or 'head' in query or 'neck' in query or 'respiratory system' in query or 'otorhinolaryngologist' in query:
                    speak("I think you need to go for a Otorhinolaryngologists")
                    print("I think you need to go for a Otorhinolaryngologists")
                    speak("They treat diseases in the ears, nose, throat, sinuses, head, neck, and respiratory system. They also can do reconstructive and plastic surgery on your head and neck.")
                    print("They treat diseases in the ears, nose, throat, sinuses, head, neck, and respiratory system. They also can do reconstructive and plastic surgery on your head and neck.")
                    speak("You can contact Doctor Shailesh Pandey")
                    print("You can contact Dr. Shailesh Pandey")
                    speak("Below are the address and contact number of Doctor Shailesh Pandey")
                    print("Address: Dr Pandey's Prime ENT Clinic, C-4, 1st Floor, Ratan Deep Building, SV Road, Andheri West, Mumbai, Maharashtra 400058 Landmark: Next to Shoppers Stop.")
                    print("Contact: 020 4855 4748 Ext. 145")

                    dr_type = "Otorhinolaryngologist"
                    dr_name = "Dr. Shailesh Pandey"
                    dr_add = "Dr Pandey's Prime ENT Clinic, C-4, 1st Floor, Ratan Deep Building, SV Road, Andheri West, Mumbai, Maharashtra 400058 Landmark: Next to Shoppers Stop."
                    dr_contact = "020 4855 4748 Ext. 145" 
                    dr_info = f'''MBBS, Diploma in Otorhinolaryngology (DLO)
                                  ENT/ Otorhinolaryngologist, Pediatric Otorhinolaryngologist
                                  24 Years Experience Overall  (22 years as specialist)'''
                    link1 = "https://www.practo.com/mumbai/doctor/shailesh-pandey-ear-nose-throat-ent-specialist"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break     
                
        #DONE
                elif 'tests urine' in query  and 'blood fluid samples' in query or 'pathology' in query or 'pathologist' in query:
                    speak("I think you need to go for a Pathologists")
                    print("I think you need to go for a Pathologists")
                    speak("These lab doctors identify the causes of diseases by examining body tissues and fluids under microscopes.")
                    print("These lab doctors identify the causes of diseases by examining body tissues and fluids under microscopes.")
                    speak("You can contact Doctor Munjal Shah,")
                    print("You can contact Dr. Munjal Shah,")
                    speak("Below are the address and contact number of Doctor Munjal Shah,")
                    print("Address: Paras papthology LLP, 8/59, Old Anand Nagar, Service Rd, Near Vakola Highway Signal, Santacruz East, Mumbai, Maharashtra 400055.")
                    print("Contact: +91 70452 80906")

                    dr_type = "Pathologist"
                    dr_name = "Dr. Munjal Shah,"
                    dr_add = "Paras pathology LLP, 8/59, Old Anand Nagar, Service Rd, Near Vakola Highway Signal, Santacruz East, Mumbai, Maharashtra 400055."
                    dr_contact = "+91 70452 80906"              
                    dr_info = f'''MBBS, MD (Pathology) Pathologist '''
                    link1 = "https://www.paraspathology.com/drmunjalshah.html"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break 
                
        #done
                elif 'treat neck' in query or 'back pain' in query or 'sports' in query or 'spinal cord injuries' in query or ' physiotherapist' in query:
                    speak("I think you need to go for a Physiotherapists")
                    print("I think you need to go for a Physiotherapists")
                    speak("These specialists in physical medicine and rehabilitation treat neck or back pain and sports or spinal cord injuries as well as other disabilities caused by accidents or diseases.")
                    print("These specialists in physical medicine and rehabilitation treat neck or back pain and sports or spinal cord injuries as well as other disabilities caused by accidents or diseases.")
                    speak("You can contact Doctor Niraj Jha")
                    print("You can contact Dr. Niraj Jha")
                    speak("Below are the address and contact number of Doctor Niraj Jha")
                    print("Address: Dr. Jha's Physio World, A-1, 3-B, 1st Floor, Sukh Shantiniketan Co-op. Housing Society, L.B.S. Marg, Ghatkopar West, Mumbai, Maharashtra 400086.  Landmark: Near Shreyas Cinema.")
                    print("Contact: 020 7117 3185 Ext. 830")

                    dr_type = "Physiotherapist"
                    dr_name = "Dr. Niraj Jha"
                    dr_add = "Dr. Jha's Physio World, A-1, 3-B, 1st Floor, Sukh Shantiniketan Co-op. Housing Society, L.B.S. Marg, Ghatkopar West, Mumbai, Maharashtra 400086.  Landmark: Near Shreyas Cinema."
                    dr_contact = "020 7117 3185 Ext. 830"  
                    dr_info = f'''BPTh/BPT, MPTh/MPT - Neurological Physiotherapy
                                  TherapistNeuro Physiotherapist, Physiotherapist
                                  16 Years Experience Overall  (8 years as specialist)'''
                    link1 = "https://www.practo.com/mumbai/therapist/dr-niraj-jha-physiotherapist"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break 
                
        #done
                elif 'rebuild' in query and 'face' in query or 'plastic' in query:
                    speak("I think you need to go for a Plastic Surgeons")
                    print("I think you need to go for a Plastic Surgeons")
                    speak("You might call them cosmetic surgeons. They rebuild or repair your skin, face, hands, breasts, or body.")
                    print("You might call them cosmetic surgeons. They rebuild or repair your skin, face, hands, breasts, or body.")
                    speak("You can contact Doctor Parag Telang")
                    print("You can contact Dr. Parag Telang")
                    speak("Below are the address and contact number of Doctor Parag Telang")
                    print("Address: Designer Bodyz, 401-402, Vastu Precinct, Opp Mercedes showroom, Sundervan, Lokhandwala Road, Andheri West, Mumbai, Maharashtra 400053.")
                    print("Contact: +91 075067 10258")

                    dr_type = "Plastic Surgeon"
                    dr_name = "Dr. Parag Telang,"
                    dr_add = "Designer Bodyz, 401-402, Vastu Precinct, Opp Mercedes showroom, Sundervan, Lokhandwala Road, Andheri West, Mumbai, Maharashtra 400053."
                    dr_contact = "+91 075067 10258"  
                    dr_info = f'''MBBS, MS - General Surgery, MCh - Plastic Surgery, Fellowship in Aesthetic Medicine (FAM)
                                  Plastic Surgeon
                                  14 Years Experience Overall  (12 years as specialist)'''
                    link1 = "https://www.drparagtelang.com/book-an-appointment"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break 
                
        #done
                elif 'ankles' in query or 'feet' in query or 'foot' in query or 'podiatrist' in query:
                    speak("I think you need to go for a Podiatrists")
                    print("I think you need to go for a Podiatrists")
                    speak("They care for problems in your ankles and feet. That can include injuries from accidents or sports or from ongoing health conditions like diabetes.")
                    print("They care for problems in your ankles and feet. That can include injuries from accidents or sports or from ongoing health conditions like diabetes.")
                    speak("You can contact Doctor Shah")
                    print("You can contact Dr.Shah")
                    speak("Below are the address and contact number of Doctor Shah")
                    print("Address: Orthofit Healthcare Pvt.ltd, 9th Floor, Mahalaxmi Chambers, 22, Bhulabhai Desai Marg, Mahalaxmi West, Mumbai, Maharashtra 400026")
                    print("Contact: 084549 20321 or 020 7117 3201 Ext. 476")

                    dr_type = "Podiatrist"
                    dr_name = "Dr. Shah"
                    dr_add = "Orthofit Healthcare Pvt.ltd, 9th Floor, Mahalaxmi Chambers, 22, Bhulabhai Desai Marg, Mahalaxmi West, Mumbai, Maharashtra 400026"
                    dr_contact = "084549 20321"             
                    dr_info = ""
                    link1 = "https://orthofit.in/contact/"
                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break 
                
        #done
                elif 'mental' in query or 'emotional' in query  or 'addictive disorders' in query or 'psychiatrist' in query:
                    speak("I think you need to go for a Psychiatrists")
                    print("I think you need to go for a Psychiatrists")
                    speak("These doctors work with people with mental, emotional, or addictive disorders. They can diagnose and treat depression, schizophrenia, substance abuse, anxiety disorders, and sexual and gender identity issues.")
                    print("These doctors work with people with mental, emotional, or addictive disorders. They can diagnose and treat depression, schizophrenia, substance abuse, anxiety disorders, and sexual and gender identity issues.")
                    speak("You can contact Doctor Ajit Dandekar")
                    print("You can contact Dr. Ajit Dandekar")
                    speak("Below are the address and contact number of Doctor Ajit Dandekar")
                    print("Address: C/3, Viral Apartment, Opp Shoppers Stop, S V Road, Bharucha Baug, Parsi Colony, Andheri West, Mumbai, Maharashtra 400047")
                    print("Contact: 022 2628 7788")

                    dr_type = "Psychiatrist"
                    dr_name = "Dr. Ajit Dandekar"
                    dr_add = "C/3, Viral Apartment, Opp Shoppers Stop, S V Road, Bharucha Baug, Parsi Colony, Andheri West, Mumbai, Maharashtra 400047"
                    dr_contact = "022 2628 7788" 
                    dr_info = f'''MBBS, MD - Psychological Medicine
                                  Psychiatrist
                                  41 Years Experience Overall '''
                    link1 ="https://www.practo.com/mumbai/doctor/dr-ajit-dandekar-9-psychiatrist"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break 
                
        #done
                elif 'lungs' in query or 'pulmonologist' in query:
                    speak("I think you need to go for a Pulmonologists")
                    print("I think you need to go for a Pulmonologists")
                    speak("You would see these specialists for problems like lung cancer, pneumonia, asthma, emphysema, and trouble sleeping caused by breathing issues.")
                    print("You would see these specialists for problems like lung cancer, pneumonia, asthma, emphysema, and trouble sleeping caused by breathing issues.")
                    speak("You can contact Doctor Dilip Maydeo")
                    print("You can contact Dr.Dilip Maydeo")
                    speak("Below are the address and contact number of Doctor Dilip Maydeo")
                    print("Address: Dr. Dilip Maydeo Clinic, 9, Nilkanth Shopping Centre, Camalane, Ghatkopar West, Mumbai, Maharashtra 400086. Landmark: Opposite SNDT Womens College.")
                    print("Contact: +91 20485 54700 Ext. 405")

                    dr_type = "Pulmonologist"
                    dr_name = "Dr. Dilip Maydeo"
                    dr_add = "Dr. Dilip Maydeo Clinic, 9, Nilkanth Shopping Centre, Camalane, Ghatkopar West, Mumbai, Maharashtra 400086. Landmark: Opposite SNDT Womens College."
                    dr_contact = "+91 20485 54700 Ext. 405" 
                    dr_info = f'''MBBS, MD - Tuberculosis and Chest Diseases
                                  Pulmonologist
                                  39 Years Experience Overall  (38 years as specialist) '''
                    link1 = "https://www.practo.com/mumbai/doctor/dr-dilip-v-maydeo-general-physician"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i, dr_info,link1)
                    break   
                
        #done
                elif 'joint' in query or 'muscle' in query or 'bone' in query or 'tendon' in query or 'arthritis' in query or 'rheumatologist' in query:
                    speak("I think you need to go for a Rheumatologists")
                    print("I think you need to go for a Rheumatologists")
                    speak("They specialize in arthritis and other diseases in your joints, muscles, bones, and tendons. You might see them for your osteoporosis (weak bones), back pain, gout, tendinitis from sports or repetitive injuries, and fibromyalgia.")
                    print("They specialize in arthritis and other diseases in your joints, muscles, bones, and tendons. You might see them for your osteoporosis (weak bones), back pain, gout, tendinitis from sports or repetitive injuries, and fibromyalgia.")
                    speak("You can contact Doctor Mahesh Maheshwari")
                    print("You can contact Dr. Mahesh Maheshwari")
                    speak("Below are the address and contact number of Doctor Mahesh Maheshwari")
                    print("Address: Sai Hospital Sector 15, Above Sanman Hotel, Nerul east, Navi Mumbai, Maharashtra 400706. Landmark: Near Railway station.")
                    print("Contact: +91 20485 52186 Ext. 055")

                    dr_type = "Rheumatologist"
                    dr_name = "Dr. Mahesh Maheshwari"
                    dr_add = "Sai Hospital Sector 15, Above Sanman Hotel, Nerul east, Navi Mumbai, Maharashtra 400706. Landmark: Near Railway station."
                    dr_contact = "+91 20485 52186 Ext. 055" 
                    dr_info = f'''MBBS, MS - Orthopaedics
                                  Orthopedic surgeon
                                  33 Years Experience Overall  (29 years as specialist)  '''
                    link1 ="https://www.practo.com/navi-mumbai/doctor/dr-mahesh-maheshwari-orthopedist-2"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break           
                
        #done
                elif 'tumors' in query or 'appendices' in query or 'gallbladders' in query or 'repair hernias' in query or 'surgery' in query or 'surgeon' in query:
                    speak("I think you need to go for a Surgeons")
                    print("I think you need to go for a Surgeons")
                    speak("These doctors can operate on all parts of your body. They can take out tumors, appendices, or gallbladders and repair hernias. Many surgeons have subspecialties, like cancer, hand, or vascular surgery.")
                    print("These doctors can operate on all parts of your body. They can take out tumors, appendices, or gallbladders and repair hernias. Many surgeons have subspecialties, like cancer, hand, or vascular surgery.")
                    speak("You can contact Doctor Rajshree Murudkar")
                    print("You can contact Dr. Rajshree Murudkar")
                    speak("Below are the address and contact number of Doctor Rajshree Murudkar")
                    print("Address: Lifewave Hospital, A/5, Sukh Sagar Mehal, Bachani Nagar, Malad east, Mumbai, Maharashtra 400097. Landmark: Near Children’s Academy High School.")
                    print("Contact: 022 4890 2319 Ext. 298")

                    dr_type = "General Surgeon"
                    dr_name = "Dr. Rajshree Murudkar"
                    dr_add = "Lifewave Hospital, A/5, Sukh Sagar Mehal, Bachani Nagar, Malad east, Mumbai, Maharashtra 400097. Landmark: Near Children’s Academy High School."
                    dr_contact = "022 4890 2319 Ext. 298" 
                    dr_info = f'''MS - General Surgery, MBBS
                                  General Surgeon
                                  25 Years Experience Overall  (21 years as specialist)'''
                    link1 = "https://www.practo.com/mumbai/doctor/rajshree-murudkar-general-surgeon"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i,dr_info,link1)
                    break 
                
        #done
                elif 'urinary tract' in query or 'urinary' in query or ' urologist' in query:
                    speak("I think you need to go for a Urologists")
                    print("I think you need to go for a Urologists")
                    speak("These are surgeons who care for men and women for problems in the urinary tract, like a leaky bladder. They also treat male infertility and do prostate exams.")
                    print("These are surgeons who care for men and women for problems in the urinary tract, like a leaky bladder. They also treat male infertility and do prostate exams.")
                    speak("You can contact Doctor Ravindra D Hodarkar")
                    print("You can contact Dr. Ravindra D Hodarkar")
                    speak("Below are the address and contact number of Doctor Ravindra D Hodarkar")
                    print("Address: Upkar Clinic, C Wing, Ground Floor, Satyam Shopping Centre, M G Road, Near Somaiya College, Ghatkopar East, Mumbai - 400077")
                    print("Contact: 020 7117 3184 Ext. 809")

                    dr_type = "Urologist"
                    dr_name = "Dr. Ravindra D Hodarkar"
                    dr_add = "Upkar Clinic, C Wing, Ground Floor, Satyam Shopping Centre, M G Road, Near Somaiya College, Ghatkopar East, Mumbai - 400077"
                    dr_contact = "020 7117 3184 Ext. 809"             
                    dr_info = f'''MBBS, MS - General Surgery, MCh - Urology, DNB - Urology/Genito - Urinary Surgery
                                  Urologist, General Physician
                                  45 Years Experience Overall  (39 years as specialist)'''
                    link1 = "https://www.practo.com/mumbai/doctor/dr-r-d-hodarkar-urologist-1"

                    speak("Do you want to book an appoinment")
                    print("Do you want to book an appoinment?")
                    print("Yes or No")

                    booking(dr_type,dr_name,dr_add,dr_contact,i)
                    break 
                
                elif 'back' in query or 'menu' in query:
                    speak("redirecting to menu")
                    menu(option)

                else:
                    speak("Cannot find any doctor for this")
                    speak("please specify the body part in which you have problem")
        doctor()

def usrname():
    speak("What should i call you")   #
    uname = takeCommand()
    speak(f'''Welcome {(uname)}''')

def wishMe():
    usrname
    hour = int(datetime.datetime.now().hour)
    if hour>= 0 and hour<12:
        speak("Good Morning !")

    elif hour>= 12 and hour<18:
        speak("Good Afternoon !")

    else:
        speak("Good Evening !")

    assname =("Healthcare Assistant") #name of the assistant
    speak(f'''I am your {(assname)}''')


def menu(option):
        
        if '1' in option or 'disease prediction' in option or 'predict' in option or 'disease' in option or 'one' in option:
            print("1. Disease Prediction")
            speak("ok i will predict the disease")
            return diseases()

        elif 'two' in option or 'booking' in option or 'booking details' in option or 'mail' in option or 'Receive Doctor booking details through mail' in option or 'doctor' in option or 'details' in option:
            print("2. Get Doctor booking details through mail.")
            speak("ok i will send you mail regarding booking details")
            return booking_details_email()

        elif 'three' in option or 'third' in option or 'exit' in option or 'nothing' in option:
            speak("my favourite option. ok bye, have a good day!.")
            sys.exit("my favourite option ;) ok bye, have a good day!")

        else:
            print("invalid option selected")
            speak("sorry i can't do that")
            option = takeCommand().lower()
    
if __name__=="__main__" :
        usrname()
        wishMe()
        speak("i can do the following things")
        print("################### Heealthcare assistant #####################\n")
        print("1. Disease Prediction\n")
        speak("disease prediction")
        print("2. Send Doctor booking details through mail.\n")
        speak("Send Doctor booking details through mail.")
        print("3.exit or nothing.\n")
        speak("exit or nothing")
        print("################################################################\n")
        while(1):
            print("################### Healthcare assistant #####################\n")
            print("1. Disease Prediction\n")
            print("2. Send Doctor booking details through mail.\n")
            print("3.exit or nothing.\n")
            print("################################################################\n")
    
            option = takeCommand().lower()
            if(len(option)==0):
                 option = takeCommand().lower()

            if '1' in option or 'disease prediction' in option or 'predict' in option or 'disease' in option or 'one' in option:
                print("1. Disease Prediction")
                speak("ok i will predict the disease")
                diseases()


            elif 'two' in option or 'booking' in option or 'booking details' in option or 'mail' in option or 'Receive Doctor booking details through mail' in option or 'doctor' in option or 'details' in option:
                print("2. Get Doctor booking details through mail.")
                speak("ok i will send you mail regarding booking details")
                booking_details_email()


            elif 'three' in option or 'third' in option or 'exit' in option or 'nothing' in option:
                speak("my favourite option. ok bye, have a good day!.")
                sys.exit("my favourite option ;) ok bye, have a good day!")

            else:
                print("invalid option selected")
                speak("sorry i can't do that")
                
