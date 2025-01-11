import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
data = pd.read_csv('../Datasets/cleaned_combined_data.csv')

num_rows, num_cols = data.shape
print("Number of Rows: ", num_rows)
print("Number of Cols: ", num_cols)

X = data.iloc[:,1]
y = data.iloc[:,0]

value_counts = y.value_counts()
labels = ["(0)- Not Spam","(1) - Spam"]

plt.bar(value_counts.index, value_counts.values, width=1.0, edgecolor= 'black')
plt.title('Classes Histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks([0, 1], labels= labels)
plt.show()

plt.pie(y.value_counts(), labels= labels, autopct = "%.2f%%")
plt.show()

spam= X[y==1]
print("Spam shape: ", spam.shape)
print(data.head())
words= []

# Language Detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'  # Handle cases where language detection fails

langs = {}

for example in X:
    lang = detect_language(example)
    if lang not in langs.keys():
        langs[lang] = 1
    else:
        langs[lang] += 1


plt.bar(langs.keys(), langs.values())
plt.title('Language Frequency')
plt.xlabel('Language')
plt.ylabel('Frequency')
plt.show()