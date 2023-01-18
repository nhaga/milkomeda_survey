import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS

import pandas as pd

file = 'dcSpark_Branding_Survey2023-01-16_05_26_14.csv'
st.sidebar.title('Milkomeda Survey Results')
data = pd.read_csv(file)
st.sidebar.subheader("16.01.2023")
st.sidebar.caption(f"Answers: {data.shape[0]}")

if st.sidebar.button('Get Winners'):
    st.sidebar.write('10 Random Winners')
    flds = ['Submission ID', 'Submission IP']
    flds = ['Submission ID', '(Optional) If you would like a chance to win 400 Ada , please write your Cardano wallet address ']
    sub = data[flds].dropna().sample(10)
    sub.columns = ['ID', 'Wallet']
    st.sidebar.dataframe(data=sub)

    @st.experimental_memo
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(sub)

    st.sidebar.download_button(
    "Press to Download",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv'
    )



else:
    pass



st.subheader('How did you get to know dcspark?')
raw = data[data.columns[0]].value_counts()
df = raw[raw > 2].sort_values()
fig = px.bar(df, y=df.index, x=df.values, orientation='h')
st.plotly_chart(fig, use_container_width=True)
st.text("Other answers...")
st.dataframe(data=raw[raw <= 2], use_container_width=True)





st.subheader('Questions (1 to 7)')
fig, ax = plt.subplots()
cols = [1,2,3,4,5,6, 19, 20, 22, 25]
colors = [
 '#c23728',
 '#e14b31',
 '#de6e56',
 '#e1a692',
 '#e2e2e2',
 '#a7d5ed',
 '#63bff0',
 '#22a7f0',
 '#1984c5']
subdata = {data.columns[col]: data[data.columns[col]].value_counts(normalize=True) for col in cols} 
fig = go.Figure()
df = pd.DataFrame(subdata)
colors = [
 '#c23728',
 '#e14b31',
 '#de6e56',
 '#e1a692',
 '#e2e2e2',
 '#a7d5ed',
 '#63bff0',
 '#22a7f0',
 '#1984c5']

for i, (idx, row) in enumerate(df.iterrows()):
    fig.add_trace(go.Bar(
        y=df.columns,
        x=row,
        name=idx,
        orientation='h',
        marker=dict(
            color=colors[i],
        )
    ))
annotations = []


    # labeling the bar net worth

averages = df.fillna(0).index @ df.fillna(0).values
for idx, val in enumerate(averages):
    annotations.append(dict(xref='x1', yref='y1',
                            y=idx, x=1.07,
                            text=f"{val:.2f}",
                            font=dict(family='Arial', size=10),
                            showarrow=False))
    
    

fig.update_layout(barmode='stack', plot_bgcolor='rgba(0,0,0,0)', annotations=annotations)

st.plotly_chart(fig, use_container_width=True)




# How often do you use dcspark's products or services?
label = data.columns[7]
st.subheader(label)
df = data[label].value_counts()
df = df[df > 1].sort_values()
fig = px.bar(df, y=df.index, x=df.values, orientation='h')
st.plotly_chart(fig, use_container_width=True)



# "Which products below are built by dcspark? Choose all that apply"
label = data.columns[8]
st.subheader(label)
raw = data[label].value_counts()
values = {}
for row in raw.index:
    words = row.split('\n')
    for word in words:
        values[word] = values[word] + raw[row] if word in values else raw[row]
df = pd.Series(values).sort_values()
fig = px.bar(df, y=df.index, x=df.values, orientation='h')
st.plotly_chart(fig, use_container_width=True)



# How would you rate the quality of dcsparks's products ? 
label = data.columns[9]
st.subheader(label)
df = data[label].value_counts()
df = df[df > 1].sort_values()
fig = px.bar(df, y=df.index, x=df.values, orientation='h')
st.plotly_chart(fig, use_container_width=True)


# Which blockchains does dcspark develop products for?
label = data.columns[10]
st.subheader(label)
raw = data[label].value_counts()
values = {}
for row in raw.index:
    words = row.split('\n')
    for word in words:
        values[word] = values[word] + raw[row] if word in values else raw[row]
df = pd.Series(values).sort_values()
fig = px.bar(df, y=df.index, x=df.values, orientation='h')
st.plotly_chart(fig, use_container_width=True)



# Which word or words come to your mind when you think of dcspark? 
label = data.columns[12]
st.subheader(label)
raw = data[label].value_counts()
values = {}
for row in raw.index:
    words = row.split('\n')
    for word in words:
        values[word] = values[word] + raw[row] if word in values else raw[row]
df = pd.Series(values).sort_values()
df = df[df > 1].sort_values()
fig = px.bar(df, y=df.index, x=df.values, orientation='h')
st.plotly_chart(fig, use_container_width=True)


# Wordcloud
comment_words = ''
for row in data[label]:
    tokens = row.split('\n')
    comment_words += " ".join(map(lambda x: x.lower(), tokens))+" "
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_word_length=0,
                stopwords = stopwords,
                ranks_only=True,
                collocations=False,  
                min_font_size = 10).generate(comment_words)   
fig, ax = plt.subplots(figsize=(5,8))            
ax.imshow(wordcloud)
ax.axis("off")
plt.tight_layout(pad = 0)
st.pyplot(fig)


# Agree / Disagree
st.subheader("Agree / Disagree")
fig, ax = plt.subplots()
cols = [13, 14, 15, 16]
colors = [
 '#f9f9f9',
'#e14b31',
   '#de6e56',     
 '#e2e2e2',
 '#63bff0',   
 '#1984c5', 
 ]
subdata = {data.columns[col]: data[data.columns[col]].value_counts(normalize=True) for col in cols} 
df = pd.DataFrame(subdata)
index = ['Do not know', 'Totally Disagree', 'Somewhat Disagree', 'Neutral', 'Somewhat Agree', 'Totally Agree' ]
df.columns = list(map(lambda x: x.split('>> ')[-1], df.columns))
df = df.loc[index]
fig = go.Figure()

colors = [
 '#dad6d6',
 '#c23728',
 '#e14b31',
 '#f1eeee',
 '#a7d5ed',
 '#63bff0',
 '#22a7f0',
 '#1984c5']

for i, (idx, row) in enumerate(df.iterrows()):
    fig.add_trace(go.Bar(
        y=df.columns,
        x=row,
        name=idx,
        orientation='h',
        marker=dict(
            color=colors[i],
        )
    ))
fig.update_layout(barmode='stack')
st.plotly_chart(fig, use_container_width=True)


# Rank what you think are the most important elements for a brand's success?
fig, ax = plt.subplots()
label = data.columns[18]
st.subheader(label)
colors = [
 '#1984c5',
 '#22a7f0',
 '#63bff0',
 '#e1a692',
 '#de6e56',
 '#e14b31',
 '#c23728'
 ]



val = {}
for row in data[label]:
    words = row.split('\n')
    for idx, word in enumerate(words, 1):
        word = word[3:].replace('\r', '')
        if not word in val:
            val[word] = {}
        try:
            val[word][idx] += 1
        except:
            val[word][idx] = 1

df = pd.DataFrame(val)
df = df[df>1]
fig = go.Figure()
for i, (idx, row) in enumerate(df.iterrows()):
    fig.add_trace(go.Bar(
        y=df.columns,
        x=row,
        name=idx,
        orientation='h',
        marker=dict(
            color=colors[i],
        )
    ))
fig.update_layout(barmode='stack')
st.plotly_chart(fig, use_container_width=True)


# What does dcspark make/provide? Choose all that apply
label = data.columns[24]
st.subheader(label)
df = data[label]
raw = df[~df.isna()].value_counts()

values = {}
for row in raw.index:
    words = row.split('\n')
    for word in words:
        values[word] = values[word] + raw[row] if word in values else raw[row]
subdata = pd.Series(values).sort_values()
df = data[label].value_counts()
fig = px.bar(subdata, y=subdata.index, x=subdata.values, orientation='h')
st.plotly_chart(fig, use_container_width=True)

# What is the single best thing you want to see dcspark continue doing?
words = "milkomeda, seba/sebastien/sebastian, interoperability/interoperable, cardano, dapps/dapp, content, education/educate, CIP/CIPs, wallet, flint, community, UI/UX"
keys = words.replace(' ', '').split(',')

label = data.columns[23]
st.subheader(label)
df = data[label]
df = df[~df.isna()]

raw = {}
comment_words = ''
for row in df:
    for key in keys:
        for word in key.split('/'):
            if word in row:
                raw[key] = raw[key] + 1 if key in raw else 1
                comment_words += f' {key}'  
subdata = pd.Series(raw).sort_values()
fig = px.bar(subdata, y=subdata.index, x=subdata.values, orientation='h')
st.plotly_chart(fig, use_container_width=True)

stopwords = set(STOPWORDS)

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_word_length=0,
                stopwords = stopwords,
                ranks_only=True,
                collocations=False,
                min_font_size = 10).generate(comment_words
                )
fig, ax = plt.subplots(figsize=(5,5))            
ax.imshow(wordcloud)
ax.axis("off")
plt.tight_layout(pad = 0)
st.pyplot(fig)

# (Optional) What is the most important improvement you want to see from dcspark?
label = data.columns[26]
st.subheader(label)
df = data[label]
df = df[~df.isna()]

raw = {}
comment_words = ''
for row in df:
    for key in keys:
        for word in key.split('/'):
            if word in row:
                raw[key] = raw[key] + 1 if key in raw else 1
                comment_words += f' {key}' 
subdata = pd.Series(raw).sort_values()
fig = px.bar(subdata, y=subdata.index, x=subdata.values, orientation='h')
st.plotly_chart(fig, use_container_width=True)