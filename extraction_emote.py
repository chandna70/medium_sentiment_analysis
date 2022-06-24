##Extracting emoticon text 
emot=['smile','laugh','sad']
for i in emot:
  x=[]
  if i =='smile':
    for s in df_twitter['tweet_text']:
      if re.search(':\)|:-\)',s.lower()):
        x.append(1)
      else:
        x.append(0)
  elif i =='laugh':
    for s in df_twitter['tweet_text']:
      if re.search(':D|:-D|haha|lol|lmao',s.lower()):
        x.append(1)
      else:
        x.append(0)
  elif i =='sad':
    for s in df_twitter['tweet_text']:
      if re.search(':\(|:-\(',s.lower()):
        x.append(1)
      else:
        x.append(0)
  df_twitter[i+'_emot']=x  
