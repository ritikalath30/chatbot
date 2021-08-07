import numpy as np
import tensorflow as tf
import nltk
from tensorflow.keras import layers,models
from string import punctuation
import pickle
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from flask import Flask
from flask.templating import render_template
from flask.wrappers import request

print(nltk.download('stopwords'))
print(nltk.download('wordnet'))


model=models.load_model('/files/model/Induction_Chatbot_6.model')
feature_indices_dict={}
with open('/files/model/feature_indices_3.txt','rb') as f:
  feature_indices_dict=pickle.load(f)
answers=[]
with open('/files/model/answers_3.txt','rb') as f:
  answers=pickle.load(f)
bot_responses=[]
with open('/files/model/bot_responses_3.txt','rb') as f:
  bot_responses=pickle.load(f)


app = Flask(__name__)

# @app.route('/')
# def man():
#     return render_template('index.html')

@app.route('/',methods=["GET","POST"])
def home():
    if request.method == "POST":
        # data = request.form.get("question")
        data1 = predict('hello')
        print(data1)
        return render_template('index.html',data1=data1)
    else:
        return render_template('index.html',data1='')

def predict(text):
  sentences=[text]
  # print(sentences)
  lower_sent=[]
  for sent in sentences:
    sent=sent.lower()
    lower_sent.append(sent)
  punct_sent=[]
  for comm in lower_sent:
    for w in comm:
      if w in punctuation:
        comm=comm.replace(w,'')
    punct_sent.append(comm)
  comm_words=[]
  for comm in punct_sent:
    comm_words.append(comm.split())
  cleaned_text=[]
  for comm in comm_words:
    words=[]
    for w in comm:
      if w not in set(stopwords.words('english')):
        words.append(w)
    cleaned_text.append(words)
  lemmatizer=WordNetLemmatizer()
  lemm_text=[]
  for text in cleaned_text:
    words=[]
    for w in text:
      words.append(lemmatizer.lemmatize(w))
    lemm_text.append(words)
  stemmer=SnowballStemmer('english')
  stem_text=[]
  for text in lemm_text:
    words=''
    for w in text:
      words=words+stemmer.stem(w)+' '
    words=words.replace('colleg','')
    words=words.replace('facil','')
    words=words.replace('campus','')
    stem_text.append(words)
  # print(stem_text[0])
  classes={0:'greetings', 1:'overview',2:'conduct', 3:'academics', 4:'resources',5:'activities',6:'goodbye'}
  vectorizer=TfidfVectorizer()
  tfidf_matrix=vectorizer.fit_transform(stem_text)
  X=tfidf_matrix.toarray()
  feature_arr=vectorizer.get_feature_names()
  feature_arr_dict={}
  for ind,name in enumerate(feature_arr):
    feature_arr_dict[name]=ind
  test_tens=[[0]*784]
  for i in feature_arr:
    try:
      test_tens[0][feature_indices_dict[i]]=X[0][feature_arr_dict[i]]
    except:
      pass
  test_tens=tf.convert_to_tensor(np.array(test_tens))
  predictions=model.predict(test_tens)
  index=np.argmax(predictions)
  responses_list=[]
  responses_list=bot_responses[index][:]
  responses_list.insert(0,stem_text[0])
  count_vectorize=CountVectorizer().fit_transform(responses_list)
  vectors=count_vectorize.toarray()
  csim=cosine_similarity(vectors)
  answer_index=np.argmax(csim[0][1:])
  print(answers[index][answer_index])
  return classes[index]
            

if __name__ == "__main__":
    app.run(debug=True)