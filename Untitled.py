#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df= pd.read_csv('analysis_id(in).csv')


# In[6]:


df.isna().sum()


# In[10]:


df['prebid_win_count']= df['prebid_win_count'].fillna(0)


# In[11]:


df.isna().sum()


# In[32]:


win_rate=df.groupby('bidder').agg(total_bids=('request_count','sum'),
                                     total_win=('prebid_win_count','sum'))


# In[33]:


win_rate


# In[41]:


win_rate['win_rate']=win_rate.apply(lambda row: row['total_win']/row['total_bids'] if row['total_bids']>0 else 0, axis=1)*100


# In[42]:


win_rate.sort_values(by='win_rate', ascending=False)


# In[44]:


win_rate.sort_values(by='win_rate', ascending=False).head(2)


# Part1:
# It's a binary classifiacation problem the probablity will be between 0 and 1.Firstly we will handle missing values, handle categorical variables and then I will take the features as X and target variable as Y, split them into train and test datasets (test size can be take as 30%) and train the model with the losgistic Regression model, and try to see the results on Y test dataset after fitting it on X test dataset
# 
# The features I will select are:
# device _type: user device type can affect ad quality and compitiotion
# time_zone: some time zones might have higher ad value or higher compitiotion
# os_name: maybe we can include as we can correlate it with device type and it can affect auction behaviour
# ad_unit: different ad units will have different ad value and compitiotions or bids
# size: size can influence the cost of a add or its performance
# bid_range: shows the bid strategy and how much bidder is willing to spend
# request_count: more requests can help analyze more opportunities
# response_count: helps to show better system performance
# prebid_win_count: helps us to show who and how many times won
# win_count: If it is highly same as prebid_win_count it can cause multicollinearity but in our case we have many different values
# sum_bid: higher bid can show higher winning chances
# sum_time_to_respond: faster response time means high chances of winning
# min_bid, max_bid, avg_bid: useful to show bidding intensities
# sum_2nd_highest_bid: useful for how bids are close with top

# Part 2:
# bid_amount: Coefficient = 0.8 -> if 1 unit in bid amount increases the probablity of winning increases by that is higher bid amount has more chance of winning
# time_to_bid: Coefficient = -0.4 -> each 1 unit of delay reduces probability of by 0.4. Faster bids have compititive advantage and delays will decrease winning chances
# ad_unit_A: Coefficient = 0.3 (reference category: ad_unit_B) -> when the add is shown in Ad unit A instead of ad unit B the chances of winning is 0.3. that is Ad unit A performs better in auctions than in unit B.

# In[ ]:




