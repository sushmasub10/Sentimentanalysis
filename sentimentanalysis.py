import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt 


data = pd.read_csv("/home/user/Documents/ML_projetcs/NLP/project/Reviews.csv")
data = data.head(10000)

example = data['Text'][50]

tokens = nltk.word_tokenize(example)
tokens = tokens[:10]
tagged = nltk.pos_tag(tokens)


entities = nltk.chunk.ne_chunk(tagged)
print(entities)


sia = SentimentIntensityAnalyzer()

sia.polarity_scores('im so happy')
sia.polarity_scores("This oatmeal is not good. Its mushy, soft, I dont like it. Quaker Oats is the way to go.")


sia.polarity_scores(example)

res = {}
for i, row in tqdm(data.iterrows(), total=len(data)):
    
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)
df = pd.DataFrame(res).T
df = df.reset_index().rename(columns={'index':id})
result = pd.concat([df, data], axis=1)

ax = sns.barplot(data = result, x = 'Score', y='compound')
ax.set_title("compound score by amazon star review")
plt.show()


ax = sns.barplot(data = result, x = 'Score', y='pos')
ax.set_title("compound score by amazon star review")
plt.show()


fig , axs = plt.subplots(1,3, figsize=(15,5))
sns.barplot(data = result, x = 'Score', y='compound', ax = axs[0])
sns.barplot(data = result, x = 'Score', y='pos', ax = axs[1])
sns.barplot(data = result, x = 'Score', y='neg', ax = axs[2])
plt.show()