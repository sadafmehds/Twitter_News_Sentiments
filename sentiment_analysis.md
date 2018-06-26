

```python
import json
import tweepy
import numpy as np
from config import consumer_key, consumer_secret, access_token, access_token_secret

import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Accounts
agencies=['BBCWorld','CBSNews','CNN','FoxNews','nytimes']

#empty lists of needed data
date=[]
agency_name=[]
text=[]
comp=[]
neutral=[]
negative=[]
positive=[]

#go through past 5 pages for last 100 tweets
for x in range(5):
    #go through every target account/news agency
    for agency in agencies:
        public_tweets=api.user_timeline(agency,page=x)
            # Loop through all tweets 
        for tweet in public_tweets:
            date.append(tweet['created_at'])
            agency_name.append(tweet['user']['name'])
            text.append(tweet['text'])
                
            comp.append(analyzer.polarity_scores(tweet["text"])["compound"])
            neutral.append(analyzer.polarity_scores(tweet["text"])["neu"])
            negative.append(analyzer.polarity_scores(tweet["text"])["neg"])
            positive.append(analyzer.polarity_scores(tweet["text"])["pos"])
                
#place all data in a dictionary and make dataframe
data_dicti={'Tweet Date':date,
            'News Agency':agency_name,
            'Tweet Text':text,
            'Compound Score':comp,
           'Neutral Score':neutral,
           'Negative Score':negative,
           'Positive Score':positive}

data_df=pd.DataFrame(data_dicti)
data_df


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet Date</th>
      <th>News Agency</th>
      <th>Tweet Text</th>
      <th>Compound Score</th>
      <th>Neutral Score</th>
      <th>Negative Score</th>
      <th>Positive Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tue Jun 26 16:28:50 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>Noura Hussein: Appeals court overturns death s...</td>
      <td>-0.5574</td>
      <td>0.536</td>
      <td>0.348</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tue Jun 26 16:02:33 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>No BBC Sports News Correspondents were harmed ...</td>
      <td>-0.6486</td>
      <td>0.762</td>
      <td>0.238</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tue Jun 26 15:54:27 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>RT @BBCSport: FT #AUS 0-2 #PER \n\n18' ⚽ Andre...</td>
      <td>0.5859</td>
      <td>0.863</td>
      <td>0.000</td>
      <td>0.137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tue Jun 26 15:51:40 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>RT @BBCSport: FT: #DEN 0-0 #FRA \n\nFrom start...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tue Jun 26 15:49:50 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>Guernsey airline Waves cancels flights for two...</td>
      <td>-0.2263</td>
      <td>0.808</td>
      <td>0.192</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tue Jun 26 15:41:08 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>RT @BBCSport: The rags to riches story of the ...</td>
      <td>0.4939</td>
      <td>0.674</td>
      <td>0.106</td>
      <td>0.220</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Tue Jun 26 15:32:13 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>Teenager helps deaf and blind man to communica...</td>
      <td>-0.0258</td>
      <td>0.629</td>
      <td>0.189</td>
      <td>0.182</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Tue Jun 26 15:25:01 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>Marree Man: The enduring mystery of a giant ou...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Tue Jun 26 15:19:53 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>Ottawa Bluesfest preparations obstructed by ne...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Tue Jun 26 15:12:14 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>Ethiopia offers Eritrea chance to end Africa's...</td>
      <td>-0.4404</td>
      <td>0.576</td>
      <td>0.281</td>
      <td>0.144</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Tue Jun 26 15:03:43 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>New Zealand man shot after 'flying to US to at...</td>
      <td>-0.4767</td>
      <td>0.780</td>
      <td>0.220</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Tue Jun 26 15:03:20 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>It's a big day at the World Cup and we've alre...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Tue Jun 26 14:55:35 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>RT @BBCMonitoring: VAR! What is it good for? #...</td>
      <td>0.4926</td>
      <td>0.738</td>
      <td>0.000</td>
      <td>0.262</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Tue Jun 26 14:50:28 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>Trump travel ban: What does this ruling mean? ...</td>
      <td>-0.5574</td>
      <td>0.690</td>
      <td>0.310</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Tue Jun 26 14:47:23 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>RT @BBCSport: HT #AUS 0-1 #PER \n\n18' ⚽ Andre...</td>
      <td>0.5719</td>
      <td>0.856</td>
      <td>0.000</td>
      <td>0.144</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Tue Jun 26 14:41:59 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>Eritrea and Ethiopia open first high-level tal...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Tue Jun 26 14:34:20 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>DR Congo's Kasai crisis: War crimes committed ...</td>
      <td>-0.7845</td>
      <td>0.498</td>
      <td>0.398</td>
      <td>0.104</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Tue Jun 26 14:26:32 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>US Supreme Court upholds Trump's travel ban ht...</td>
      <td>0.0000</td>
      <td>0.455</td>
      <td>0.273</td>
      <td>0.273</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Tue Jun 26 14:24:13 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>RT @BBCSport: GOAL! #AUS 0-1 #PER\n\nWhat a st...</td>
      <td>-0.2695</td>
      <td>0.910</td>
      <td>0.090</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Tue Jun 26 14:06:48 +0000 2018</td>
      <td>BBC News (World)</td>
      <td>"Most people don't talk about organ donations ...</td>
      <td>0.6705</td>
      <td>0.744</td>
      <td>0.000</td>
      <td>0.256</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Tue Jun 26 17:14:50 +0000 2018</td>
      <td>CBS News</td>
      <td>Heather Locklear arrested for allegedly batter...</td>
      <td>-0.6908</td>
      <td>0.637</td>
      <td>0.363</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Tue Jun 26 17:05:38 +0000 2018</td>
      <td>CBS News</td>
      <td>The Supreme Court's ruling "shows the attacks ...</td>
      <td>-0.3400</td>
      <td>0.639</td>
      <td>0.226</td>
      <td>0.135</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Tue Jun 26 17:04:09 +0000 2018</td>
      <td>CBS News</td>
      <td>WATCH: President Trump attends lunch with cong...</td>
      <td>0.5574</td>
      <td>0.833</td>
      <td>0.000</td>
      <td>0.167</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Tue Jun 26 16:54:39 +0000 2018</td>
      <td>CBS News</td>
      <td>Sean Spicer TV show, "Sean Spicer's Common Gro...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Tue Jun 26 16:34:56 +0000 2018</td>
      <td>CBS News</td>
      <td>Years ago, a royal helped a couple survive the...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Tue Jun 26 16:14:42 +0000 2018</td>
      <td>CBS News</td>
      <td>"Jurassic World" star Chris Pratt surprises ch...</td>
      <td>0.2960</td>
      <td>0.455</td>
      <td>0.200</td>
      <td>0.345</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Tue Jun 26 15:30:06 +0000 2018</td>
      <td>CBS News</td>
      <td>WATCH LIVE: The Council on American–Islamic Re...</td>
      <td>0.5574</td>
      <td>0.825</td>
      <td>0.000</td>
      <td>0.175</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Tue Jun 26 15:23:25 +0000 2018</td>
      <td>CBS News</td>
      <td>WATCH: Vice President Mike Pence is meeting wi...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Tue Jun 26 14:59:35 +0000 2018</td>
      <td>CBS News</td>
      <td>Second shooting at California state park where...</td>
      <td>-0.6705</td>
      <td>0.780</td>
      <td>0.220</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Tue Jun 26 14:52:03 +0000 2018</td>
      <td>CBS News</td>
      <td>"We are disappointed to hear about today's rul...</td>
      <td>-0.4767</td>
      <td>0.846</td>
      <td>0.154</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>Tue Jun 26 08:38:00 +0000 2018</td>
      <td>Fox News</td>
      <td>During Monday's press briefing, @PressSec Sara...</td>
      <td>-0.5423</td>
      <td>0.811</td>
      <td>0.189</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>471</th>
      <td>Tue Jun 26 08:30:00 +0000 2018</td>
      <td>Fox News</td>
      <td>.@GovMikeHuckabee: "When you deny one person's...</td>
      <td>-0.0284</td>
      <td>0.695</td>
      <td>0.174</td>
      <td>0.131</td>
    </tr>
    <tr>
      <th>472</th>
      <td>Tue Jun 26 08:23:00 +0000 2018</td>
      <td>Fox News</td>
      <td>Navy SEALS join search in Thailand for boys so...</td>
      <td>0.2960</td>
      <td>0.855</td>
      <td>0.000</td>
      <td>0.145</td>
    </tr>
    <tr>
      <th>473</th>
      <td>Tue Jun 26 08:15:00 +0000 2018</td>
      <td>Fox News</td>
      <td>.@ IngrahamAngle: "Publicly shaming conservati...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>474</th>
      <td>Tue Jun 26 08:07:00 +0000 2018</td>
      <td>Fox News</td>
      <td>On the Senate floor Monday, @SenSchumer rebuke...</td>
      <td>-0.1779</td>
      <td>0.904</td>
      <td>0.096</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>475</th>
      <td>Tue Jun 26 08:00:00 +0000 2018</td>
      <td>Fox News</td>
      <td>Rep. @SteveScalise: "This is about civility. Y...</td>
      <td>0.6249</td>
      <td>0.797</td>
      <td>0.000</td>
      <td>0.203</td>
    </tr>
    <tr>
      <th>476</th>
      <td>Tue Jun 26 07:53:00 +0000 2018</td>
      <td>Fox News</td>
      <td>Irate customer caught on surveillance camera v...</td>
      <td>-0.5994</td>
      <td>0.698</td>
      <td>0.302</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>477</th>
      <td>Tue Jun 26 07:45:01 +0000 2018</td>
      <td>Fox News</td>
      <td>.@POTUS on @HillaryClinton: "She blamed everyb...</td>
      <td>-0.6908</td>
      <td>0.711</td>
      <td>0.289</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>478</th>
      <td>Tue Jun 26 07:38:00 +0000 2018</td>
      <td>Fox News</td>
      <td>On Twitter Monday, Former President @GeorgeHWB...</td>
      <td>0.4019</td>
      <td>0.847</td>
      <td>0.000</td>
      <td>0.153</td>
    </tr>
    <tr>
      <th>479</th>
      <td>Tue Jun 26 07:30:00 +0000 2018</td>
      <td>Fox News</td>
      <td>.@GovMikeHuckabee: "My daughter handled this v...</td>
      <td>0.5499</td>
      <td>0.827</td>
      <td>0.000</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>480</th>
      <td>Tue Jun 26 04:58:28 +0000 2018</td>
      <td>The New York Times</td>
      <td>A plan to build an AirTrain between Manhattan ...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>481</th>
      <td>Tue Jun 26 04:42:03 +0000 2018</td>
      <td>The New York Times</td>
      <td>5 years ago, the Supreme Court struck down a k...</td>
      <td>0.3818</td>
      <td>0.781</td>
      <td>0.078</td>
      <td>0.141</td>
    </tr>
    <tr>
      <th>482</th>
      <td>Tue Jun 26 04:22:04 +0000 2018</td>
      <td>The New York Times</td>
      <td>RT @nytopinion: In Episode One of @realtrumpbi...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>483</th>
      <td>Tue Jun 26 04:11:25 +0000 2018</td>
      <td>The New York Times</td>
      <td>Five Takeaways From Turkey’s Election https://...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>484</th>
      <td>Tue Jun 26 04:02:02 +0000 2018</td>
      <td>The New York Times</td>
      <td>Amid increasing concerns about privacy violati...</td>
      <td>-0.5267</td>
      <td>0.833</td>
      <td>0.167</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>485</th>
      <td>Tue Jun 26 03:47:01 +0000 2018</td>
      <td>The New York Times</td>
      <td>From our friends at @Watching: What to remembe...</td>
      <td>0.4767</td>
      <td>0.819</td>
      <td>0.000</td>
      <td>0.181</td>
    </tr>
    <tr>
      <th>486</th>
      <td>Tue Jun 26 03:32:08 +0000 2018</td>
      <td>The New York Times</td>
      <td>Sleep apnea can be downright deadly, and not j...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>487</th>
      <td>Tue Jun 26 03:21:05 +0000 2018</td>
      <td>The New York Times</td>
      <td>RT @nytvideo: A New York Times investigation f...</td>
      <td>-0.4939</td>
      <td>0.862</td>
      <td>0.138</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>488</th>
      <td>Tue Jun 26 03:11:03 +0000 2018</td>
      <td>The New York Times</td>
      <td>8 of the tech industry’s most influential comp...</td>
      <td>0.7425</td>
      <td>0.718</td>
      <td>0.000</td>
      <td>0.282</td>
    </tr>
    <tr>
      <th>489</th>
      <td>Tue Jun 26 03:02:03 +0000 2018</td>
      <td>The New York Times</td>
      <td>Airline stopover programs can double your dest...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>490</th>
      <td>Tue Jun 26 02:55:49 +0000 2018</td>
      <td>The New York Times</td>
      <td>RT @jmartNYT: For two years, Democrats have wr...</td>
      <td>-0.1779</td>
      <td>0.922</td>
      <td>0.078</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>491</th>
      <td>Tue Jun 26 02:47:05 +0000 2018</td>
      <td>The New York Times</td>
      <td>In a lab in Philadelphia, scientists are study...</td>
      <td>-0.6908</td>
      <td>0.810</td>
      <td>0.190</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>492</th>
      <td>Tue Jun 26 02:32:08 +0000 2018</td>
      <td>The New York Times</td>
      <td>Here's every team's path to the next round (ex...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>493</th>
      <td>Tue Jun 26 02:17:06 +0000 2018</td>
      <td>The New York Times</td>
      <td>In a 5-4 ruling, the Supreme Court upholds Tex...</td>
      <td>0.5574</td>
      <td>0.795</td>
      <td>0.000</td>
      <td>0.205</td>
    </tr>
    <tr>
      <th>494</th>
      <td>Tue Jun 26 02:02:02 +0000 2018</td>
      <td>The New York Times</td>
      <td>How China got Sri Lanka to hand over a port: A...</td>
      <td>0.4939</td>
      <td>0.856</td>
      <td>0.000</td>
      <td>0.144</td>
    </tr>
    <tr>
      <th>495</th>
      <td>Tue Jun 26 01:51:01 +0000 2018</td>
      <td>The New York Times</td>
      <td>In more than 20 of the most competitive House ...</td>
      <td>0.4927</td>
      <td>0.827</td>
      <td>0.000</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>496</th>
      <td>Tue Jun 26 01:41:01 +0000 2018</td>
      <td>The New York Times</td>
      <td>RT @nytopinion: We're suffering from a breakdo...</td>
      <td>-0.2302</td>
      <td>0.694</td>
      <td>0.179</td>
      <td>0.126</td>
    </tr>
    <tr>
      <th>497</th>
      <td>Tue Jun 26 01:30:15 +0000 2018</td>
      <td>The New York Times</td>
      <td>A scientist was fatally shot while camping wit...</td>
      <td>-0.6369</td>
      <td>0.811</td>
      <td>0.189</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>498</th>
      <td>Tue Jun 26 01:17:04 +0000 2018</td>
      <td>The New York Times</td>
      <td>RT @readercenter: Dean Baquet, our executive e...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>499</th>
      <td>Tue Jun 26 01:02:03 +0000 2018</td>
      <td>The New York Times</td>
      <td>Evening Briefing: Here's what you need to know...</td>
      <td>0.0000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 7 columns</p>
</div>




```python
#convert date strings to datetime format
data_df['Tweet Date'] = pd.to_datetime(data_df["Tweet Date"])

# Sort by tweet date
data_df.sort_values("Tweet Date", inplace=True)
data_df.reset_index(drop=True, inplace=True)

data_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet Date</th>
      <th>News Agency</th>
      <th>Tweet Text</th>
      <th>Compound Score</th>
      <th>Neutral Score</th>
      <th>Negative Score</th>
      <th>Positive Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-06-25 14:12:13</td>
      <td>BBC News (World)</td>
      <td>RT @BBCSport: GOAL! Uruguay 1-0 Russia. \n\nLu...</td>
      <td>-0.6588</td>
      <td>0.803</td>
      <td>0.197</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-06-25 14:26:37</td>
      <td>BBC News (World)</td>
      <td>RT @BBCSport: GOAL! Uruguay 2-0 Russia. \n\nTh...</td>
      <td>0.1759</td>
      <td>0.937</td>
      <td>0.000</td>
      <td>0.063</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-06-25 14:31:49</td>
      <td>BBC News (World)</td>
      <td>RT @BBCSport: GOAL!\n\nThat's the Mo Salah we ...</td>
      <td>0.5550</td>
      <td>0.860</td>
      <td>0.000</td>
      <td>0.140</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-06-25 14:46:00</td>
      <td>BBC News (World)</td>
      <td>RT @BBCSport: What a moment!\n\nEssam El-Hadar...</td>
      <td>0.2714</td>
      <td>0.738</td>
      <td>0.111</td>
      <td>0.151</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-06-25 14:54:34</td>
      <td>BBC News (World)</td>
      <td>RT @BBCSport: GOAL!\n\nAnother penalty for Sau...</td>
      <td>-0.1007</td>
      <td>0.764</td>
      <td>0.126</td>
      <td>0.111</td>
    </tr>
  </tbody>
</table>
</div>




```python
#extract dataframe for each seperate news agency
x=data_df[data_df['News Agency']=='BBC News (World)']
y=data_df[data_df['News Agency']=='CBS News']
z=data_df[data_df['News Agency']=='Fox News']
k=data_df[data_df['News Agency']=='The New York Times']
j=data_df[data_df['News Agency']=='CNN']
```


```python
#plot polarity scores for each agency
plt.figure(figsize=(12,8))
plt.scatter((np.arange(-len(x),0,1)),x['Compound Score'],label='BBC News',marker='o',color='blue',s=100,alpha=0.6)
plt.scatter((np.arange(-len(y),0,1)),y['Compound Score'],label='CBS News',marker='o',color='red',s=100,alpha=0.6)
plt.scatter((np.arange(-len(z),0,1)),z['Compound Score'],label='Fox News',marker='o',color='green',s=100,alpha=0.6)
plt.scatter((np.arange(-len(k),0,1)),k['Compound Score'],label='New York Times',marker='o',color='yellow',s=100,alpha=0.6)
plt.scatter((np.arange(-len(j),0,1)),j['Compound Score'],label='CNN',marker='o',color='maroon',s=100,alpha=0.6)

legend=plt.legend(bbox_to_anchor=(1,1),loc="upper left", title="News Agencies")

plt.title("Sentiment Analysis of Media Tweets (%s)" % time.strftime("%x"))
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.xticks([-100, -80, -60, -40, -20, 0], [100, 80, 60, 40, 20, 0])


# plt.savefig('output/news_sentiment_scatter.png',bbox_extra_artists=(legend,), bbox_inches='tight')
```


![png](output_5_0.png)



```python
#autolabel definitions from before
def autolabelpos(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,
                '+%.2f' % float(height),
                ha='center', va='bottom')

def autolabelneg(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., -1*height-0.015,
                '-%.2f' % float(height),
                ha='center', va='bottom')

#get mean compound score for each news agency's tweets
data_df_grouped=data_df.groupby(['News Agency']).mean()['Compound Score']

data_df_grouped
```




    News Agency
    BBC News (World)     -0.091120
    CBS News             -0.101242
    CNN                  -0.116257
    Fox News              0.051630
    The New York Times   -0.030223
    Name: Compound Score, dtype: float64




```python
#all mean scores from each news agency
comp_score_means = (data_df_grouped["BBC News (World)"], 
                    data_df_grouped["CBS News"], 
                    data_df_grouped["Fox News"], 
                    data_df_grouped["The New York Times"],
                    data_df_grouped["CNN"])


#bar plot of mean polarity (comp) scores
fig, ax = plt.subplots()
tick = np.arange(len(comp_score_means))  

bar1 = ax.bar(tick[0], comp_score_means[0], 1, color="blue")
bar2 = ax.bar(tick[1], comp_score_means[1], 1, color="red")
bar3 = ax.bar(tick[2], comp_score_means[2], 1, color="green")
bar4 = ax.bar(tick[3], comp_score_means[3], 1, color='yellow')
bar5 = ax.bar(tick[4], comp_score_means[4], 1, color='lightcoral')

#use labeling definition to get label shown
autolabelpos(bar1)
autolabelpos(bar2)
autolabelpos(bar3)
autolabelpos(bar4)
autolabelpos(bar5)

ax.set_title("Overall Media Sentiment based on Twitter (%s) " % (time.strftime("%x")))
ax.set_ylabel("Tweet Polarity")
ax.set_xticks(tick)
ax.set_xticklabels(("BBC News", "CBS News", "Fox News", "New York Times", "CNN"))

# plt.savefig("output/sentiment_bar.png",bbox_inches="tight")


```


![png](output_7_0.png)

