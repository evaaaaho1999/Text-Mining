import pandas as pd
import matplotlib.pyplot as plt
import nltk
import gdown
import time

import pyimgur
im = pyimgur.Imgur("2e6a7fc89850e3d")


from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *

import tempfile, os
from config import client_id, client_secret, album_id, access_token, refresh_token, line_channel_access_token, \
    line_channel_secret

import requests
import json

from linebot import (
    LineBotApi, WebhookHandler
)

# import time
import re #python的re模組 #用來匹配特定符號


app = Flask(__name__)

#line information
line_bot_api = LineBotApi('NxZu9744E8dIyEHsE9nY5IcHbKCXOe+WJa389T2LceI6SO8mKrNO1G4id4julbt0fqYOeMm1/hGgyPrb/ulRM/6dJjJyX2BuZlcJB9krWsg+Bw+LgwgPAgBY8u0rizEaUGI4Aw1pNV0K/6P0hbYT4AdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('56b4082afaf189bdd445205cc1f57557')

static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')


###########################################################################################
nltk.download('punkt')
def get_df():
    # df_query
    url = "https://drive.google.com/u/0/uc?id=1-24aWX9oqnhYm65onxV5Zm4PbXppvtGD&export=download"
    gdown.download(url, "news_en.csv", quiet=True)
    return pd.read_csv("news_en.csv")
def get_doc_all(df_query):
    # doc_all
    doc_all={}
    for i in range(len(df_query)):
        tokens = nltk.word_tokenize(df_query.news[i])
        token_filtered = [w.lower() for w in tokens if w.isalpha()]
        doc_all[i]=token_filtered
    return doc_all
# Declare all function 
# create tf function
def tf(term, token_doc):
    tf = token_doc.count(term)/len(token_doc)
    return tf

# create function to calculate how many doc contain the term 
def numDocsContaining(word, token_doclist):
    doccount = 0
    for doc_token in token_doclist:
        if doc_token.count(word) > 0:
            doccount +=1
    return doccount
  
import math
# create function to calculate  Inverse Document Frequency in doclist - this list of all documents
def idf(word, token_doclist):
    n = len(token_doclist)
    df = numDocsContaining(word, token_doclist)
    return math.log10(n/df)

#define a function to do cosine normalization a data dictionary
def cos_norm(dic): # dic is distionary data structure
    import numpy as np
    dic_norm={}
    factor=1.0/np.sqrt(sum([np.square(i) for i in dic.values()]))
    for k in dic:
        dic_norm[k] = dic[k]*factor
    return dic_norm

#create function to calculate normalize tfidf 
def compute_tfidf(token_doc,bag_words_idf):
    tfidf_doc={}
    for word in set(token_doc):
        tfidf_doc[word]= tf(word,token_doc) * bag_words_idf[word]   
    tfidf_norm = cos_norm(tfidf_doc)
    return tfidf_norm

# create normalize term frequency
def tf_norm(token_doc):
    tf_norm={}
    for term in token_doc:
        tf = token_doc.count(term)/len(token_doc)
        tf_norm[term]=tf
    tf_max = max(tf_norm.values())
    for term, value in tf_norm.items():
        tf_norm[term]= 0.5 + 0.5*value/tf_max
    return tf_norm
  
def compute_tfidf_query(query_token,bag_words_idf):
    tfidf_query={}
    tf_norm_query = tf_norm(query_token)
    for term, value in tf_norm_query.items():
        tfidf_query[term]=value*bag_words_idf[term]   
    return tfidf_query
def add_tfidf_of_query(tfidf_query):
    tfidf["query"]
def get_tfidf_query(input_query,bag_words,bag_words_idf):
    # tfidf_query,query_token
    query=str(input_query)
    query=query.lower()
    query_token_raw= nltk.word_tokenize(query)
    query_token = [term for term in query_token_raw if term in bag_words]
    if len(query_token) >0 :
        tfidf_query =compute_tfidf_query(query_token,bag_words_idf) #calculate tfidf for query text
        return tfidf_query,query_token
    else:  
        tp == "No relative news."
        return tp

def web_crawer():
    import bs4
    from bs4 import BeautifulSoup as soup
    from urllib.request import urlopen

    news_url="https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US"
    Client=urlopen(news_url)
    xml_page=Client.read()
    Client.close()

    soup_page=soup(xml_page,"xml")
    #soup_page=soup(xml_page,"html.parser")
    news_list=soup_page.findAll("item")
    # Print news title, url and publish date
    news_title=[]
    website=[]
    today_news=[]
    i=0
    for news in news_list:
        news_title.append(news.title.text)
        website.append(news.link.text)
        i+=1
    
    return news_title,website
def query_and_answer(input_query="trump"):
    news_title,website = web_crawer()
    df_query = get_df()
    doc_all = get_doc_all(df_query)
    
    #create bag words
    bag_words =[] # declare bag_words is a list
    for doc in doc_all.keys():
        bag_words += doc_all[doc]
    bag_words=set(bag_words)


    bag_words_idf={} # declare "bag_words_idf" data structure is dictionary 
    bag_words_len = len(bag_words)
    bag_word_10 = round(bag_words_len/10,0)

    i=0
    for word in bag_words:
        i+=1
        bag_words_idf[word]= idf(word,doc_all.values())
    tfidf={} 
    for doc in doc_all.keys():
        tfidf[doc]= compute_tfidf(doc_all[doc],bag_words_idf)
    if (get_tfidf_query(input_query,bag_words,bag_words_idf)!=0):
        tfidf_query,query_token = get_tfidf_query(input_query,bag_words,bag_words_idf)
    else:
        return 0

    # add tfidf of query text to tfidf of all doc and convert to dataframe
    tfidf["query"]=tfidf_query

    import pandas as pd
    tfidf_df = pd.DataFrame(tfidf).transpose()
    tfidf_df= tfidf_df.fillna(0) # replace all NaN by zero

    from scipy.spatial.distance import cosine
    cosine_sim ={}
    for row in tfidf_df.index:
        if row != "query":
            cosine_sim[row]= 1-cosine(tfidf_df.loc[row],tfidf_df.loc["query"])

    # the top 10 relevant document
    if len(query_token) >0 :
        cosine_sim_top10 = dict(sorted(cosine_sim.items(), key=lambda item: item[1],reverse=True)[:10])
        # print(cosine_sim_top10)

    returns=list(cosine_sim_top10.keys())[0]

    return (news_title[returns],website[returns])
##########################################################################################################





# def MakePlot():
#     # 取得 struct_time 格式的時間
#     t = time.localtime()

#     # 依指定格式輸出
#     result = time.strftime("%Y%m%d", t)
#     path = result+".png"

#     import matplotlib.pyplot as plt
#     plt.plot([1,2,3],[1,2,3])

#     path = result+".png"
#     plt.savefig(path)


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    # print("body:",body)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'ok'


@handler.add(MessageEvent, message=(ImageMessage, TextMessage))
def handle_message(event):
    
    message = event.message.text
    
    if isinstance(event.message, TextMessage):
        
        # 文字雲
        if event.message.text == "See today's trend":
            
            url_cloud= "https://raw.githubusercontent.com/Keycatowo/financial-sentimental-linebot/master/image/wordcloud.png"
            
            image_message = ImageSendMessage(
                original_content_url=url_cloud,
                preview_image_url=url_cloud
            )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            return 0
        
        # 綜合比較圖(4合1)
        elif event.message.text == "Today's financial index and sentimental index":
            
            url_pic = "https://i.imgur.com/yMsXJUA.png"
            
            image_message = ImageSendMessage(
                original_content_url=url_pic,
                preview_image_url=url_pic
            )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0
        

        #相關新聞/Query
        elif event.message.text == "Search for today's relative news":
            line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(
                        text="Please type your keyword :\n\" # + keyword \"\n( ex: #trump )"
                        )
                ])
            return 0


        elif(re.match(r'^#.', message)): 

            text_received = message.split('#')
            news_query = text_received[1]  # this is "input"
            
      
            tp = query_and_answer(str(news_query))
            
            if(tp==0):

                line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(
                        text="(No relative news.)")
                ])
                return 0
            
            else:         
                output_title = tp[0]
                output_link = tp[1]

                line_bot_api.reply_message(
                    event.reply_token, [
                        TextSendMessage(
                            text=output_link) 
                    ])
                return 0
        

        
        #股票資訊
        elif event.message.text == "Search for today's stock information":
            line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(
                        text="Please type stock code :\n \" @ + stock code \"\n( ex: @3231、@AAPL)")
                ])
            return 0
        
        elif(re.match(r'^@.', message)): 

            text_received = message.split('@')
            stock_name = text_received[1]  # this is "input"

            #美股
            if stock_name.isalpha()==True:
                
                urlst1="https://query1.finance.yahoo.com/v7/finance/download/"
                urlst2=stock_name #user's input
                urlst3="?period1="
                urlst4="&period2="
                urlst5="&interval=1d&events=history"
                t2=round(time.time())
                t1=t2-86400*7

                url_ST_US=urlst1 +urlst2+urlst3+str(t1) + urlst4+ str(t2) + urlst5
                gdown.download(url_ST_US,"stock price us.csv", quiet=False)
                ST_US_data = pd.read_csv('stock price us.csv')
                

                if (str(ST_US_data)=='Empty DataFrame\nColumns: [404 Not Found: No data found,  symbol may be delisted]\nIndex: []'):
                    
                    line_bot_api.reply_message(
                        event.reply_token, [
                            TextSendMessage(
                                text="(Not found ...QQ)")
                        ])
                    return 0
                    
                else:
                    date=[]
                    for i in range (len(ST_US_data.Date)):
                        date.append(ST_US_data.Date[i])
                    print(date)
                    plt.figure()
                    x_value=date
                    y_value=ST_US_data[['Open','Close','High', 'Low']]
                    for i in y_value:
                        plt.plot(x_value,ST_US_data[str(i)],label = str(i))
                    plt.legend(loc="upper right")
                
                    path = "stock_US.png"
                    plt.savefig(path)

                    uploaded_image = im.upload_image(path, title="Uploaded with PyImgur")
                    stock_US_image = str(uploaded_image.link)

                    image_message = ImageSendMessage(
                        original_content_url=stock_US_image,
                        preview_image_url=stock_US_image
                    )
                    
                    line_bot_api.reply_message(
                        event.reply_token, image_message)
                    
                    return 0

            #台股
            else:           
                urlst1="https://query1.finance.yahoo.com/v7/finance/download/"
                urlst2=stock_name
                urlst3=".TW?period1="
                urlst4="&period2="
                urlst5="&interval=1d&events=history"
                t2=round(time.time())
                t1=t2-86400*7

                url_ST_TW=urlst1 +urlst2+urlst3+str(t1) + urlst4+ str(t2) + urlst5
                gdown.download(url_ST_TW,"stock price tw.csv", quiet=False)
                ST_TW_data = pd.read_csv('stock price tw.csv')
                

                if (str(ST_TW_data)=='Empty DataFrame\nColumns: [404 Not Found: No data found,  symbol may be delisted]\nIndex: []'):
                    
                    line_bot_api.reply_message(
                        event.reply_token, [
                            TextSendMessage(
                                text="(Not found...QQ)")
                        ])
                    return 0
                
                else:
                    date=[]
                    for i in range (len(ST_TW_data.Date)):
                        date.append(ST_TW_data.Date[i])
                    print(date)
                    plt.figure()
                    x_value=date
                    y_value=ST_TW_data[['Open','Close','High', 'Low']]
                    for i in y_value:
                        plt.plot(x_value,ST_TW_data[str(i)],label = str(i))
                    plt.legend(loc="upper right")
                    # plt.show()
                    path = "stock_TW.png"
                    plt.savefig(path)

                    uploaded_image = im.upload_image(path, title="Uploaded with PyImgur")
                    stock_TW_image = str(uploaded_image.link)

                    image_message = ImageSendMessage(
                        original_content_url=stock_TW_image,
                        preview_image_url=stock_TW_image
                    )
                    
                    line_bot_api.reply_message(
                        event.reply_token, image_message)
                    
                    return 0
    

        elif event.message.text=="1":
            url_pic = "https://i.imgur.com/x2ORUF2.jpg"
            image_message = ImageSendMessage(
            original_content_url=url_pic,
            preview_image_url=url_pic
            )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0
            
        elif event.message.text=="2":
            url_pic = "https://i.imgur.com/JZvc5wd.jpg"
            image_message = ImageSendMessage(
                original_content_url=url_pic,
                preview_image_url=url_pic
                )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0

        elif event.message.text=="3":
            url_pic = "https://i.imgur.com/Kt5eXpw.jpg"
            image_message = ImageSendMessage(
                original_content_url=url_pic,
                preview_image_url=url_pic
                )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0

        elif event.message.text=="4":
            url_pic = "https://i.imgur.com/dh70pWk.jpg"
            image_message = ImageSendMessage(
                original_content_url=url_pic,
                preview_image_url=url_pic
                )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0

        elif event.message.text=="5":
            url_pic = "https://i.imgur.com/yUtBGWa.jpg"
            image_message = ImageSendMessage(
                original_content_url=url_pic,
                preview_image_url=url_pic
                )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0

        elif event.message.text=="6":
            url_pic = "https://i.imgur.com/TnaQ1pJ.jpg"
            image_message = ImageSendMessage(
                original_content_url=url_pic,
                preview_image_url=url_pic
                )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0

        elif event.message.text=="7" :
            url_pic = "https://i.imgur.com/qtyXGfX.jpg"
            image_message = ImageSendMessage(
                original_content_url=url_pic,
                preview_image_url=url_pic
                )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0

        elif event.message.text=="8":
            url_pic = "https://i.imgur.com/YYz2BPw.jpg"
            image_message = ImageSendMessage(
                original_content_url=url_pic,
                preview_image_url=url_pic
                )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0

        elif event.message.text=="9":
            url_pic = "https://i.imgur.com/pTTp4KI.jpg"
            image_message = ImageSendMessage(
                original_content_url=url_pic,
                preview_image_url=url_pic
                )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0

        elif event.message.text=="10":
            url_pic = "https://i.imgur.com/t3wxmcE.jpg"
            image_message = ImageSendMessage(
                original_content_url=url_pic,
                preview_image_url=url_pic
                )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0

        elif event.message.text=="group2":
            url_pic = "https://i.imgur.com/iQebUVJ.jpg"
            image_message = ImageSendMessage(
                original_content_url=url_pic,
                preview_image_url=url_pic
                )
            
            line_bot_api.reply_message(
                event.reply_token, image_message)
            
            return 0

        #處理其他資訊:重複使用者的話
        else:
            
            line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(
                        text="Please click the menu to get information")
                ])
            
            return 0



if __name__ == '__main__':
    app.run()

