import pandas as pd
from langdetect import detect
import re
data = pd.read_csv('../Datasets/combined_data.csv')

X = data.iloc[:,1]
y = data.iloc[:,0]
num_examples = X.shape[0]
file = open("../Datasets/cleaned_combined_data.csv", "w+",  encoding='utf-8')
file.write("label,text\n")
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown' 

multiline = re.compile('.*\n.*')
for index in range(num_examples):
    lang = detect_language(X[index])
    if lang == 'en':
            label = str(y[index])
            text = X[index].replace('"', '""')  # Escape double quotes within the text
            file.write(f'{label},"{text}"\n')

file.close()