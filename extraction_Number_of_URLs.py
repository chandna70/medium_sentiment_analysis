##Extracting amount of URLs
import regex as re
count_url=[]
df_count=df_twitter.tweet_text
for id,text in enumerate(df_count):
  count = 0
  for sub_text in text.split():
    if (re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+',sub_text)):
      count=count+1
  count_url.append(count)

df_count_url=df_twitter.assign(count_url=count_url)
