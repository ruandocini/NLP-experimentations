import pandas as pd 
import nltk as nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import seaborn as sns
import matplotlib.pyplot as plt 

def list_steamer(list):
    ps = PorterStemmer()
    steamed = [ps.stem(word) for word in list]
    return steamed

def domain_trash_removal(domain, list):
    clean = [word for word in list if word not in domain]
    return clean

wiki_data = pd.read_csv("Wikipedia/data/en-books-dataset.csv", nrows=100)

stop_words = list(stopwords.words("English"))
stop_words.append('Wikibooks')

wiki_data["words"] = [word_tokenize(str(line)) for line in wiki_data["title"]]
wiki_data["clean_words"] = [set(word).difference(stop_words) for word in wiki_data["words"]]

wiki_data["steamed_words"] = [list_steamer(sequence) for sequence in wiki_data["clean_words"]]

punctuation = string.punctuation
wiki_data["removed_puntuation"] = [domain_trash_removal(punctuation,w_colect) for w_colect in wiki_data["steamed_words"]]

trash_list = ['\'s', 'the', ')', ":", '?', '(', ',', '.', 'A', '9', '...', '-']
wiki_data["removed_puntuation"] = [domain_trash_removal(trash_list,w_colect) for w_colect in wiki_data["steamed_words"]]

just_words = pd.concat([pd.DataFrame(x) for x in wiki_data["removed_puntuation"]])
just_words["count"] = 1

just_words = just_words.rename({0:"Words"},axis=1)
count_words = just_words.groupby(by="Words").sum().sort_values(by="count",ascending=False)

most_important = count_words.head(15)

# sns.barplot(x=most_important.index, y=most_important["count"],palette='rocket')
# plt.show()

print(wiki_data)


