#!/usr/bin/env python
# coding: utf-8

# # Data Mining Twitter 
# ### github : https://github.com/pdeitel/IntroToPython/tree/master/examples/ch13 
# ### book   : http://localhost:8888/files/2241016309/Python%202/Python%20Book.pdf 
#             (only works in lab comp)

# In[2]:


get_ipython().system('pip install tweepy')


# In[3]:


import tweepy


# In[ ]:


API_KEY = "Lorem ipsum dolor sit amet, consectetur adipiscing elit"
API_SECRET = "Phasellus egestas accumsan felis vitae rhoncus"
ACCESS_TOKEN = "Fusce mollis eleifend odio sed rhoncus"
ACCESS_SECRET = "Phasellus ut urna dui"
BEARER_TOKEN = "Morbi maximus orci id nisl euismod, at maximus nulla gravida"

# Replace with real stuff by creating a twitter developers account (just google)


# In[5]:


client = tweepy.Client(bearer_token=BEARER_TOKEN)

username = "Jagadeeswar_Dev"

user = client.get_user(username=username, user_fields=["public_metrics"])


# In[6]:


if user.data:
    print("User ID:", user.data.id)
    print("Username:", user.data.username)
    print("Name:", user.data.name)
    print("Followers Count:", user.data.public_metrics["followers_count"])
    print("Location:", user.data.location)
    print("Bio:", user.data.description)
    print("Profile Image URL:", user.data.profile_image_url)
    print("Verified:", user.data.verified)
    print("Account Created At:", user.data.created_at)
    print("Protected Tweets:", user.data.protected)
else:
    print("User not found.")


# In[7]:


query = "Jagadeeswar Patro"

tweets = client.search_recent_tweets(query=query, max_results=15)


# In[16]:


for tweet in tweets.data:
    print(f"Tweet: {tweet.text}\n")


# In[9]:


import requests


# In[19]:


def get_user_info(username):
    url=f"https://api.twitter.com/2/users/by/username/{username}?user.fields=id, name, username, description, public_metrics"
    headers = {
        "Authorization" : f"Bearer {BEARER_TOKEN}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        user_data = response.json()["data"]
        print("User ID:", user_data["id"])
        print("User Name:", user_data["name"])
        print("Screen Name:", user_data["username"])
        print("Description:", user_data["description"])
        print("Followers Count:", user_data["public_metrics"]["followers_count"])
    else:
        print("Error:", response.status_code, response.text)


# In[20]:


get_user_info("Jagadeeswar_Dev")


# In[ ]:




