##Developed by Chandra Parashian Napitupulu
##IDE via Google Colab

import zipfile,os
%cd '.../.../MyDrive/...'
dir='/.../.../.../.../Sentiment_Twitter_Covid'
if not os.path.isdir(dir):
  os.makedirs(dir)
 
 ##Unzip
local_zip = 'sentiment_covid_twitter.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/.../.../.../.../Sentiment_Twitter_Covid')
zip_ref.close()

##Import csv
import pandas as pd
df_training_twitter=pd.read_csv('.../data_training_twitter+vaccine.csv',encoding='latin1')
df_testing_twitter=pd.read_csv('.../data_testing_twitter+vaccine.csv',encoding='latin1')

df_twitter=df_training_twitter
df_twitter['label']=df_twitter.label-1
df_twitter=df_twitter.drop(columns='tweet_id')
df_testing_twitter=df_testing_twitter.drop(columns='tweet_id')
df_testing_twitter['label']=df_testing_twitter.label-1
df_twitter.head()

##Describe the data
print('Size of training data :', len(df_twitter))
print('Size of testing data:',len(df_testing_twitter))
df_twitter.info()

#1.EDA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import more_itertools as miter
def count_target(df):
  target=[]
  target_name=[]
  for x in set(df):
    count=df[df==x].count()
    target.append(count)
    target_name.append(x)
  return target_name,target
def text(x,y):
  for i in range(len(x)):
    plt.text(i, y[i]//2, y[i], ha = 'center')

#1.1 Number of target class
x,y=count_target(df_twitter.label)
label=['Negative','Neutral','Positive']
df_target=pd.DataFrame({"class":x,"height":y})
df_target['class']=label
plt.figure(figsize=(8,8))
plt.bar(x=df_target['class'],height=df_target.height)
text(df_target['class'],df_target['height'])
plt.xlabel("Class")
plt.ylabel("Number of Class")
plt.title('Number of Target Class',fontweight='bold')

#1.2 Distribution of length words
text=df_twitter.assign(len_text=df_twitter['tweet_text'].apply(lambda x:len(str(x).split())))
ds=text['len_text'].describe()
ds['mode']=text['len_text'].mode()[0]
ds
plt.figure(figsize=(8,8))
sns.distplot(text.len_text)
plt.title('Distribution of length text',fontweight='bold')
plt.xticks(np.arange(0,1,max(text['len_text'])))


# 1.3 Distribution of words for each class
text=df_twitter.assign(len_text=(df_twitter['tweet_text'].apply(lambda x: len(str(x).split()))),label=df_twitter['label'])
plt.figure(figsize=(10,10))
sns.displot(text,x='len_text',hue='label',kind='kde',fill=True,legend=False)
plt.xlabel('Length of text')
plt.title('Distribution length of text for each label class',fontweight='bold')
plt.legend(labels=['Positive','Neutral','Negative'],title='Target Class',loc='center right')

text[text['label']==0]['len_text'].describe()

# 1.4 Wordcloud without punctuation
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import re

df_0=df_twitter[df_twitter['label']==0].tweet_text
df_0=df_0.apply(lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",x)).values
t0=list(miter.collapse([x.split() for x in df_0]))

df_1=df_twitter[df_twitter['label']==1].tweet_text
df_1=df_1.apply(lambda x:re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",x)).values
t1=list(miter.collapse([x.split() for x in df_1]))

df_2=df_twitter[df_twitter['label']==2].tweet_text
df_2=df_2.apply(lambda x:re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",x)).values
t2=list(miter.collapse([x.split() for x in df_2]))

img=np.array(Image.open('/content/drive/MyDrive/Icon for Wordcloud Text/twitter.JPG'))
wordtext1=WordCloud(height=1800,width=2500,mask=img,max_words=max([len(t0),len(t1),len(t2)]),background_color='white').generate(str(df_0))
wordtext2=WordCloud(height=1800,width=2500,mask=img,max_words=max([len(t0),len(t1),len(t2)]),background_color='white').generate(str(df_1))
wordtext3=WordCloud(height=1800,width=2500,mask=img,max_words=max([len(t0),len(t1),len(t2)]),background_color='white').generate(str(df_2))

plt.figure(figsize=(10,10))
plt.imshow(wordtext1,interpolation="bilinear")
plt.title('Negative Words',fontweight='bold')
plt.axis('off')
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(wordtext2,interpolation="bilinear")
plt.title('Neutral Words',fontweight='bold')
plt.axis('off')
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(wordtext3,interpolation="bilinear")
plt.title('Positive Words',fontweight='bold')
plt.axis('off')
plt.show()

# 1.5 Percentage of reaction through links url
urls =[line for line,x in enumerate(df_twitter['tweet_text']) if re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+', x)]
df_urls1=df_twitter.loc[urls,['tweet_text','label']]
x,y=count_target(df_urls1.label)
label=['Negative','Neutral','Positive']
df_urls=pd.DataFrame({"class":label,"height":y})

plt.figure(figsize=(10,10))
patches,ax=plt.pie(data=df_urls,
                   x="height",
                   labels="class",
                   wedgeprops={'linewidth' : 7, 'edgecolor' : 'white'},
                   textprops={'fontsize': 14,'fontweight':'bold'})
percentage=round(df_urls['height']*100/sum(df_urls['height']),2)
for i,text in enumerate(ax):
  x=text
  x1,y1=x.get_position()
  if (x.get_text()=='Neutral'):
    plt.annotate("%.2f"%percentage[i]+"%",
             xy=x.get_position(),
             xytext=(x1-0.268,y1-0.11),
             fontsize=14)
  else:
    plt.annotate("%.2f"%percentage[i]+"%",
             xy=x.get_position(),
             xytext=(x1,y1-0.11),
             fontsize=14)
gcf=plt.gcf()
gcf.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))
plt.title("Precentage of attached URL Text Based On Their Reaction", fontweight='bold')

# 1.6 Counting url from each text 
count_url=[]
df_count=df_twitter.tweet_text
for id,text in enumerate(df_count):
  count = 0
  for sub_text in text.split():
    if (re.match('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+',sub_text)):
      count=count+1
  count_url.append(count)

df_count_url=df_twitter.assign(count_url=count_url)
df_count=df_count_url.groupby(['label','count_url']).size().reset_index(name='size_url')

figure,ax=plt.subplots(nrows=1,ncols=3,figsize=(17,7))
for i in set(df_count['label']):
  if (i==0):
    bar=ax[i].bar(data=df_count[df_count['label']==i],
                x='count_url',
                height='size_url',
                color='blue')
    for idx,x in enumerate(bar.patches):
      ax[i].annotate(x.get_height(),xy=(idx,x.get_height()+0.35),ha='center',fontweight='bold')
    ax[i].set_title('Negative Reaction',fontweight='bold')
    x_index=df_count[df_count['label']==i]['count_url'].values
    ax[i].set_xticks(x_index)
    
  elif (i==1):
    bar=ax[i].bar(data=df_count[df_count['label']==i],
                x='count_url',
                height='size_url',
                color='orange')
    for idx,x in enumerate(bar.patches):
      ax[i].annotate(x.get_height(),xy=(idx,x.get_height()+0.35),ha='center',fontweight='bold')
    ax[i].set_title('Neutral Reaction',fontweight='bold')
    x_index=df_count[df_count['label']==i]['count_url'].values
    ax[i].set_xticks(x_index)
  
  else:
    bar=ax[i].bar(data=df_count[df_count['label']==i],
                x='count_url',
                height='size_url',
                color='red')
    for idx,x in enumerate(bar.patches):
      ax[i].annotate(x.get_height(),xy=(idx,x.get_height()+0.35),ha='center',fontweight='bold')
    ax[i].set_title('Positive Reaction',fontweight='bold')
    x_index=df_count[df_count['label']==i]['count_url'].values
    ax[i].set_xticks(x_index)
figure.suptitle('Group Number of URL Based on Reactions',fontweight='bold')

##Check 1-Gram, 2-gram and 3-gram
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')

df_gram=df_twitter['tweet_text']
df_gram=df_gram.apply(lambda x: re.sub('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+','',str(x)))
df_gram=df_gram.apply(lambda x: re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)','',str(x)))
df_gram=df_gram.apply(lambda x: x.lower())
df_gram=df_gram.apply(lambda x: re.sub(r'\d+','',str(x)))


set_stopword=set(stopwords.words('english'))
stopword={'im','iam','youre','youare','theyare','theyre',')hes','shes','its',
            'iwas','theywere','youwere','hewas','shewas','itwas',
            'ive','ihave','youve','youhave','theyve','theyhave','hehas','shehas','ithas',
            'ill','youll','youwill','iwill','theyll','theywill','hewill','shewillll','itll','etc','wouldnt'}
set_stopword.update(stopword)
c_vector=CountVectorizer(stop_words=set_stopword)
df_vectorize=c_vector.fit_transform(df_gram)
df_1gram=pd.DataFrame(df_vectorize.sum(axis=0),columns=c_vector.get_feature_names()).T.sort_values(0,ascending=False).reset_index()
df_1gram.rename(columns={'index':'1_gram',0:'total'},inplace=True)

c_vector=CountVectorizer(stop_words=set_stopword,ngram_range=(2,2))
df_vectorize=c_vector.fit_transform(df_gram)
df_2gram=pd.DataFrame(df_vectorize.sum(axis=0),columns=c_vector.get_feature_names()).T.sort_values(0,ascending=False).reset_index()
df_2gram.rename(columns={'index':'2_gram',0:'total'},inplace=True)

c_vector=CountVectorizer(stop_words=set_stopword,ngram_range=(3,3))
df_vectorize=c_vector.fit_transform(df_gram)
df_3gram=pd.DataFrame(df_vectorize.sum(axis=0),columns=c_vector.get_feature_names()).T.sort_values(0,ascending=False).reset_index()
df_3gram.rename(columns={'index':'3_gram',0:'total'},inplace=True)


df_1gram['total'].describe()
df_2gram['total'].describe()
df_3gram['total'].describe()
