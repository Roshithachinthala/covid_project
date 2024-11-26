#!/usr/bin/env python
# coding: utf-8

# # COVID-19 PROJECT 
# 

# In[7]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import folium


df = pd.read_csv('case_time_series (2).csv')
  


# In[8]:


pip install folium


# In[9]:


import folium


# In[10]:


df


# # plotting the graph in sns pairplot 

# In[11]:


sns.pairplot(df)


# In[12]:


import plotly.graph_objects as go
import pandas as pd
 
# reading the database
data = pd.read_csv('case_time_series (2).csv')
 
 
plot = go.Figure(data=[go.Scatter(
    x=data['Date'],
    y=data['Daily Confirmed'],
    mode='markers',)
])
 
# Add dropdown
plot.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=["type", "bar"],
                    label="bar plot",
                    method="restyle"
                    
                ),
                dict(
                    args=["type", "line"],
                    label="lineplot",
                    method="restyle"
                ),
                
            ]),
        ),
    ]
)
 
plot.show()


# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('case_time_series (2).csv')

Y = data.iloc[61:,1].values
R = data.iloc[61:,3].values
D = data.iloc[61:,5].values
X = data.iloc[61:,0]

plt.figure(figsize=(25,8))

ax = plt.axes()
ax.grid(linewidth=0.4, color='#8f8f8f')

ax.set_facecolor("black")
ax.set_xlabel('\nDate',size=25,color='#4bb4f2')
ax.set_ylabel('Number of Confirmed Cases\n',
              size=25,color='#4bb4f2')

plt.xticks(rotation='vertical',size='20',color='red')
plt.yticks(size=20,color='red')
plt.tick_params(size=20,color='red')

for i,j in zip(X,Y):
    ax.annotate(str(j),xy=(i,j+100),color='white',size='13')

ax.annotate('Second Lockdown 15th April',
            xy=(15.2, 860),
            xytext=(19.9,500),
            color='white',
            size='25',
            arrowprops=dict(color='red',
                            linewidth=0.025))

plt.title("COVID-19 IN : Daily Confirmed\n",
         size=50,color='#28a9ff')                  #28a9ff

ax.plot(X,Y,
        color='#1F77B4',
        marker='*',
         linewidth=4,
         markersize=15,
        markeredgecolor='#28a9ff')


# In[14]:


data = pd.read_csv('district (1).csv')


data.head()
re=data.iloc[:15,5].values
de=data.iloc[:15,4].values
co=data.iloc[:15,3].values
x=list(data.iloc[:15,0])

plt.figure(figsize=(13,10))
ax=plt.axes()

ax.set_facecolor('black')
ax.grid(linewidth=0.4, color='#8f8f8f')


plt.xticks(rotation='vertical',size='20',color='black')#ticks of X

plt.yticks(size='20',color='black')


ax.set_xlabel('\nDistrict',size=25,color='#4bb4f2')    #17th may 2020
 
ax.invert_xaxis()  

ax.set_ylabel('No. of cases\n',size=25,color='#4bb4f2')


plt.tick_params(size=20,color='white')


ax.set_title('andhrapradesh District wise breakdown\n',size=50,color='#28a9ff')

plt.bar(x,co,label='re')
plt.bar(x,re,label='re',color='green')
plt.bar(x,de,label='re',color='red')

for i,j in zip(x,co):
	ax.annotate(str(int(j)),xy=(i,j+3),color='white',size='15')

plt.legend(['Confirmed','Recovered','Deceased'],
fontsize=20)


# In[15]:


data


# # plotting the graph in sunburst

# In[16]:


import plotly.express as px
import pandas as pd


DISTRICTS = ["westgodavari", "vizianagaram", "visakhapatnam", "srikakulam", "prakasam", "nellore",
        "kurnool", "krishna", "kadapa", "guntur","east godavari","chittor","ananthapur"]

ACTIVE= ["47", "100", "13", "134", "34","99",
        "187", "53", "3", "10", "41","7","19"]      
CONFIRMED = ["122", "177", "52", "417", "102","367" ,"611" ,
        "150", "66", "14", "75", "7","70"]         

RECOVERED = ["4", "0", "0", "8", "0","15",
        "19", "3", "0", "0", "1","1","1"]         


df = pd.DataFrame(dict(DISTRICTS=DISTRICTS, ACTIVE=ACTIVE, CONFIRMED=CONFIRMED, RECOVERED=RECOVERED))

fig = px.sunburst(df, path=['DISTRICTS','ACTIVE','CONFIRMED','RECOVERED'] )

fig.show()


# In[17]:


import pandas as pd
df= pd.read_csv('Covid cases in India (3).csv')  #states of covid cases


# In[18]:


df


# In[19]:


df['Total Cases']=df['Total Confirmed cases (Indian National)']+df['Total Confirmed cases ( Foreign National )']


# In[20]:


data


# In[21]:


total_cases_overall=df['Total Cases'].sum()
print('The total number of cases till now in India is ',total_cases_overall)


# In[22]:


df['Active Cases']=df['Total Cases']-(df['Death']+df['Cured'])


# In[23]:


df


# # heat map

# In[24]:


df.style.background_gradient(cmap='Reds')


# In[25]:


Total_Active_Cases=df.groupby('Name of State / UT')['Total Cases'].sum().sort_values(ascending=False).to_frame()


# In[26]:


Total_Active_Cases


# In[27]:


Total_Active_Cases.style.background_gradient(cmap='Reds')


# In[28]:


fig=plt.figure(figsize=(20,10),dpi=200)
axes=fig.add_axes([0,0,1,1])
axes.bar(df['Name of State / UT'],df['Total Cases'])
axes.set_title("Total Cases in India")
axes.set_xlabel("Name of State / UT")
axes.set_ylabel("Total Cases")

plt.show()

#plotly
fig=go.Figure()
fig.add_trace(go.Bar(x=df['Name of State / UT'],y=df['Total Cases']))
fig.update_layout(title='Total Cases in India',xaxis=dict(title='Name of State / UT'),yaxis=dict(title='Total Cases'))


# In[29]:


data= pd.read_csv('Covid cases in India (3).csv')  #st


# In[30]:


data


# # To be shown cases in world map

# In[31]:


Indian_Cord=pd.read_csv('Indian Coordinates.csv')


# In[32]:


Indian_Cord


# In[33]:


dd=pd.merge(Indian_Cord,df,on='Name of State / UT')


# In[34]:


dd


# In[35]:



map=folium.Map(location=[20,70],zoom_start=4,tiles='Stamenterrain')

for lat,long,value, name in zip(dd['Latitude'],dd['Longitude'],dd['Total Cases'],dd['Name of State / UT']):
    folium.CircleMarker([lat,long],radius=value*0.8,popup=('<strong>State</strong>: '+str(name).capitalize()+'<br>''<strong>Total Cases</strong>: ' + str(value)+ '<br>'),color='red',fill_color='red',fill_opacity=0.2).add_to(map)
    


# 

# In[36]:


map


# 

# In[ ]:




