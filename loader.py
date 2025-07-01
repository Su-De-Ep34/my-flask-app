import os
import pandas as pd
import json
attributes = {}
model_names = ['ast','hrt','lng',
               'lvr','tyd','obs',
               'dia','brn','kdy',
               'alz']
for model_name in model_names:
    df = pd.read_csv(f"C:/Machine Learning/RuralAreaDiseasePredictor/datasets/{model_name}.csv")
    features = df.columns
    attributes[model_name] = list(features)

with open('C:/Machine Learning/RuralAreaDiseasePredictor/datasets/attributes.json' , 'w') as f:
    json.dump(attributes , f)
