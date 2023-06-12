#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.text.all import*
path = untar_data(URLs.IMDB)


# In[2]:


files=get_text_files(path,folders=['train','test','unsup'])


# In[4]:


txt=files[0].open().read(); txt[:75]


# In[5]:


spacy=WordTokenizer()
toks=first(spacy([txt]))
print(coll_repr(toks,30))


# In[6]:


first(spacy(['The U.S. dollar list1.00.']))


# In[7]:


tkn=Tokenizer(spacy)
print(coll_repr(tkn(txt),31))


# In[8]:


defaults.text_proc_rules


# In[9]:


coll_repr(tkn('©   Fast.ai www.fast.ai/INDEX'), 31)


# In[12]:


txts = L(o.open(encoding='utf-8').read() for o in files[:2000])


# In[18]:


def subword(sz):
    sp=SubwordTokenizer(vocab_sz=sz)
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])


# In[19]:


subword(200)


# In[17]:


raw_text_path = "path/to/raw_text_file.txt"

with open(raw_text_path, 'w', encoding='utf-8') as f:
    for t in progress_bar(maps(*rules, items), total=len(items), leave=False):
        f.write(f'{t}\n')


# In[21]:


tokz=tkn(txt)
print(coll_repr(txt),31)


# In[22]:


toks200 = txts[:200].map(tkn)
toks200[0]


# In[23]:


num= Numericalize()
num.setup(toks200)
coll_repr(num.vocab,20)


# In[24]:


nums=num(toks)[:20]; nums


# In[25]:


' '.join(num.vocab[o] for o in nums)
     


# In[26]:


stream = "In this chapter, we will go back over the example of classifying movie reviews we studied in chapter 1 and dig deeper under the surface. First we will look at the processing steps necessary to convert text into numbers and how to customize it. By doing this, we'll have another example of the PreProcessor used in the data block API.\nThen we will study how we build a language model and train it for a while."
tokens = tkn(stream)
bs,seq_len = 6,15
d_tokens = np.array([tokens[i*seq_len:(i+1)*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))


# In[27]:


from IPython.display import HTML


# In[28]:


stream = "In this chapter, we will go back over the example of classifying movie reviews we studied in chapter 1 and dig deeper under the surface. First we will look at the processing steps necessary to convert text into numbers and how to customize it. By doing this, we'll have another example of the PreProcessor used in the data block API.\nThen we will study how we build a language model and train it for a while."
tokens = tkn(stream)
bs,seq_len = 6,15
d_tokens = np.array([tokens[i*seq_len:(i+1)*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))


# In[29]:


bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15:i*15+seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))


# In[30]:


#hide_input
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+seq_len:i*15+2*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))


# In[31]:


#hide_input
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+10:i*15+15] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))


# In[32]:


nums200 = toks200.map(num)


# In[33]:


dl = LMDataLoader(nums200)


# In[34]:


x,y = first(dl)
x.shape,y.shape


# In[35]:


' '.join(num.vocab[o] for o in x[0][:20])
     


# In[36]:


' '.join(num.vocab[o] for o in y[0][:20])


# In[37]:


get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])

dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)


# In[ ]:


#lmao

