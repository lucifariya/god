#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing [ NLP ]
# ### github : https://github.com/pdeitel/IntroToPython/tree/master/examples/ch12 
# ### book   : http://localhost:8888/files/2241016309/Python%202/Python%20Book.pdf 
#             (only works in lab comp)

# ## TextBlob:  
#   
# **TextBlob is an object-oriented NLP text-processing library that is built on the NLTK and
# pattern NLP libraries and simplifies many of their capabilities**  
#   
# **Installing the TextBlob Module:**  
# To install TextBlob, open your Anaconda Prompt (Windows), Terminal (macOS/Linux) or shell (Linux), then execute the following command:
# > conda install -c conda-forge textblob  
#   
# Windows users might need to run the Anaconda Prompt as an Administrator for proper software installation privileges. To do so, right-click Anaconda Prompt in the start menu and select More > Run as administrator.
# Once installation completes, execute the following command to download the NLTK corpora used by TextBlob:
# > ipython -m textblob.download_corpora  
#   
# These include:  
# * The Brown Corpus (created at Brown University4) for parts-of-speech tagging.  
# * Punkt for English sentence tokenization.  
# * WordNet for word definitions, synonyms and antonyms.  
# * Averaged Perceptron Tagger for parts-of-speech tagging.  
# * conll2000 for breaking text into components, like nouns, verbs, noun phrasesand more—known as chunking the text. The name conll2000 is from the conference that created the chunking data—Conference on Computational Natural Language Learning.  
# * Movie Reviews for sentiment analysis.  
#   
# ### **Creating a TextBlob :**  

# In[1]:


from textblob import TextBlob

text = 'Today is a beautiful day. Tomorrow looks like bad weather.'
blob = TextBlob(text)

blob


# TextBlobs—and, as you’ll see shortly, Sentences and Words—support string methods and can be compared with strings. They also provide methods for various NLP tasks. Sentences, Words and TextBlobs inherit from BaseBlob, so they have many common methods and properties. 
#   
# ### **Tokenizing Text into Sentences and Words :**  
# Natural language processing often requires tokenizing text before performing other NLP tasks. TextBlob provides convenient properties for accessing the sentences and words in TextBlobs. Let’s use the **sentence property** to get a list of **Sentence** objects:

# In[2]:


blob.sentences


# The **words property** returns a **WordList** object containing a list of **Word** objects, representing each word in the TextBlob with the punctuation removed:

# In[3]:


blob.words


# **Question:** Create a TextBlob with two sentences, then tokenize it into Sentences and Words, displaying all the tokens.

# In[4]:


ex = TextBlob('My old computer is slow. My new one is fast.')


# In[5]:


ex.sentences


# In[6]:


ex.words


# ### Parts-of-Speech Tagging :
#   
# **Parts-of-speech (POS) tagging** is the process of evaluating words based on their context
# to determine each word’s part of speech. There are eight primary English parts of speech—
# nouns, pronouns, verbs, adjectives, adverbs, prepositions, conjunctions and interjections
# (words that express emotion and that are typically followed by punctuation, like “Yes!” or
# “Ha!”). Within each category there are many subcategories.
# Some words have multiple meanings. For example, the words “set” and “run” have
# hundreds of meanings each! If you look at the dictionary.com definitions of the word
# “run,” you’ll see that it can be a verb, a noun, an adjective or a part of a verb phrase. An
# important use of POS tagging is determining a word’s meaning among its possibly many
# meanings. This is important for helping computers “understand” natural language.
# The **tags property** returns a list of tuples, each containing a word and a string representing its part-of-speech tag: 

# In[7]:


blob


# In[8]:


blob.tags


# In[9]:


blob.noun_phrases     #extracting nouns 


# ### Sentiment Analysis with TextBlob’s Default Sentiment Analyzer  
#   
# One of the most common and valuable NLP tasks is **sentiment analysis**, which determines
# whether text is positive, neutral or negative. For instance, companies might use this to
# determine whether people are speaking positively or negatively online about their products. Consider the positive word “good” and the negative word “bad.” Just because a sentence contains “good” or “bad” does not mean the sentence’s sentiment necessarily is
# positive or negative. For example, the sentence  
#   
# > The food is not good.  
#   
# clearly has negative sentiment. Similarly, the sentence  
#   
# > The movie was not bad.  
# 
# clearly has positive sentiment, though perhaps not as positive as something like  
#   
# > The movie was excellent!  
#   
# Sentiment analysis is a complex machine-learning problem. However, libraries like
# TextBlob have pretrained machine learning models for performing sentiment analysis.  
#   
# ### Getting the Sentiment of a TextBlob  
#   
# A TextBlob’s **sentiment property** returns a **Sentiment**  object indicating whether the text
# is positive or negative and whether it’s objective or subjective: 

# In[10]:


blob.sentiment


# In[11]:


get_ipython().run_line_magic('precision', '3')


# In[12]:


blob.sentiment.polarity


# In[13]:


blob.sentiment.subjectivity


# In[14]:


for sentence in blob.sentences:
    print(sentence.sentiment)


# In[15]:


from textblob import Sentence

Sentence('The food is not good.').sentiment


# In[16]:


Sentence('The movie was not bad.').sentiment


# In[17]:


Sentence('The movie was excellent!').sentiment


# ### Sentiment Analysis with the NaiveBayesAnalyzer:  
# By default, a TextBlob and the Sentences and Words you get from it determine sentiment
# using a PatternAnalyzer, which uses the same sentiment analysis techniques as in the Pattern library. The TextBlob library also comes with a **NaiveBayesAnalyzer** (module **textblob.sentiments**), which was trained on a database of movie reviews. Naive Bayes10 is a
# commonly used machine learning text-classification algorithm. The following uses the
# analyzer keyword argument to specify a TextBlob’s sentiment analyzer. Recall from earlier in this ongoing IPython session that text contains 'Today is a beautiful day.
# Tomorrow looks like bad weather.':

# In[18]:


from textblob.sentiments import NaiveBayesAnalyzer
    
blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())

blob


# In[19]:


blob.sentiment


# In[20]:


for sentence in blob.sentences:
    print(sentence.sentiment)


# [ Restart Kernel Here ]  
#   
# ###  Inflection: Pluralization and Singularization  
#   
# **Inflections** are different forms of the same words, such as singular and plural (like “person”
# and “people”) and different verb tenses (like “run” and “ran”). When you’re calculating
# word frequencies, you might first want to convert all inflected words to the same form for
# more accurate word frequencies. Words and WordLists each support converting words to
# their singular or plural forms. Let’s pluralize and singularize a couple of Word objects:

# In[1]:


from textblob import Word

index = Word('index')

index.pluralize()


# In[2]:


cacti = Word('cacti')
cacti.singularize()


# Pluralizing and singularizing are sophisticated tasks which, as you can see above, are
# not as simple as adding or removing an “s” or “es” at the end of a word.
# You can do the same with a WordList: 

# In[3]:


from textblob import TextBlob

animals = TextBlob('dog cat fish bird').words
animals.pluralize()    #“fish” is the same in both its singular and plural forms.


# ### Spell Checking and Correction  
# For natural language processing tasks, it’s important that the text be free of spelling errors.
# Software packages for writing and editing text, like Microsoft Word, Google Docs and
# others automatically check your spelling as you type and typically display a red line under
# misspelled words. Other tools enable you to manually invoke a spelling checker.  
#   
# You can check a Word’s spelling with its **spellcheck method**, which returns a list of
# tuples containing possible correct spellings and a confidence value. Let’s assume we meant
# to type the word “they” but we misspelled it as “theyr.” The spell checking results show
# two possible corrections with the word 'they' having the highest confidence value:

# In[4]:


word = Word('theyr')


# In[5]:


get_ipython().run_line_magic('precision', '2')


# In[6]:


word.spellcheck()


# Note that the word with the highest confidence value might not be the correct word for
# the given context.  
#   
# TextBlobs, Sentences and Words all have a **correct method** that you can call to correct spelling. Calling correct on a Word returns the correctly spelled word that has the
# highest confidence value (as returned by spellcheck):

# In[7]:


word.correct() # chooses word with the highest confidence value


# Calling correct on a TextBlob or Sentence checks the spelling of each word. For each
# incorrect word, correct replaces it with the correctly spelled one that has the highest confidence value: 

# In[8]:


sentence = TextBlob('Ths sentense has missplled wrds.')
sentence.correct()


# [ Restart Kernel ]  
#   
# ### Normalization: Stemming and Lemmatization :  
#   
# **Stemming** removes a prefix or suffix from a word leaving only a stem, which may or may
# not be a real word. **Lemmatization** is similar, but factors in the word’s part of speech and
# meaning and results in a real word.  
#   
# Stemming and lemmatization are **normalization** operations, in which you prepare
# words for analysis. For example, before calculating statistics on words in a body of text,
# you might convert all words to lowercase so that capitalized and lowercase words are not
# treated differently. Sometimes, you might want to use a word’s root to represent the word’s
# many forms. For example, in a given application, you might want to treat all of the following words as “program”: program, programs, programmer, programming and programmed (and perhaps U.K. English spellings, like programmes as well).
# Words and WordLists each support stemming and lemmatization via the methods
# **stem** and **lemmatize**. Let’s use both on a Word:

# In[1]:


from textblob import Word
    
word = Word('varieties')


# In[2]:


word.stem()


# In[3]:


word.lemmatize()


# [Restart Kernel]  
#   
# ### Word Frequencies  
#   
# Various techniques for detecting similarity between documents rely on word frequencies.
# As you’ll see here, TextBlob automatically counts word frequencies. First, let’s load the ebook for Shakespeare’s Romeo and Juliet into a TextBlob. To do so, we’ll use the **Path class** from the Python Standard Library’s **pathlib module**:

# In[1]:


from pathlib import Path
from textblob import TextBlob

blob = TextBlob(Path('RomeoAndJuliet.txt').read_text())


# When you read a file with Path’s **read_text
# method**, it closes the file immediately after it finishes reading the file.
# You can access the word frequencies through the TextBlob’s **word_counts dictionary**.
# Let’s get the counts of several words in the play:

# In[2]:


blob.word_counts['juliet']


# In[3]:


blob.word_counts['romeo']


# In[4]:


blob.word_counts['thou']


# If you already have tokenized a TextBlob into a WordList, you can count specific
# words in the list via the **count method**.

# In[5]:


blob.words.count('joy')


# In[ ]:


blob.noun_phrases.count('lady capulet')


# [ Restart Kernel ]  
#   
# ### Getting Definitions, Synonyms and Antonyms from WordNet  
#   
# **WordNet** is a word database created by Princeton University. The TextBlob library uses
# the NLTK library’s WordNet interface, enabling you to look up word definitions, and get
# synonyms and antonyms. For more information, check out the NLTK WordNet interface
# documentation at:  
#   
# > https://www.nltk.org/api/nltk.corpus.reader.html#modulenltk.corpus.reader.wordnet  
#   
# ### Getting Definitions  
#   
# First, let’s create a Word:

# In[1]:


from textblob import Word

happy = Word('happy')


# The Word class’s **definitions property** returns a list of all the word’s definitions in
# the WordNet database:

# In[2]:


happy.definitions


# In[3]:


name = Word('Aditya')
name.definitions


# The database does not necessarily contain every dictionary definition of a given word.
# There’s also a **define method** that enables you to pass a part of speech as an argument so
# you can get definitions matching only that part of speech.  
#   
# ### Getting Synonyms  
#   
# You can get a Word’s **synsets**—that is, its sets of synonyms—via the **synsets property**. The
# result is a list of Synset objects: 

# In[4]:


happy.synsets


# Each Synset represents a group of synonyms. In the notation happy.a.01:
# * happy is the original Word’s lemmatized form (in this case, it’s the same).  
# * a is the part of speech, which can be a for adjective, n for noun, v for verb, r for adverb or s for adjective satellite. Many adjective synsets in WordNet have satellite synsets that represent similar adjectives.  
# * 01 is a 0-based index number. Many words have multiple meanings, and this is the index number of the corresponding meaning in the WordNet database.  
#   
# There’s also a **get_synsets method** that enables you to pass a part of speech as an argument so you can get Synsets matching only that part of speech.  
#   
# You can iterate through the synsets list to find the original word’s synonyms. Each Synset has a **lemmas method** that returns a list of Lemma objects representing the synonyms. A Lemma’s name method returns the synonymous word as a string. In the following code, for each Synset in the synsets list, the nested for loop iterates through that Synset’s Lemmas (if any). Then we add the synonym to the set named synonyms. We used a set collection because it automatically eliminates any duplicates we add to it: 

# In[5]:


synonyms = set()

for synset in happy.synsets:
    for lemma in synset.lemmas():
        synonyms.add(lemma.name())
        
synonyms


# ### Getting Antonyms  
#   
# If the word represented by a Lemma has antonyms in the WordNet database, invoking the
# Lemma’s antonyms method returns a list of Lemmas representing the antonyms (or an empty
# list if there are no antonyms in the database). In snippet [4] you saw there were four Synsets for 'happy'. First, let’s get the Lemmas for the Synset at index 0 of the synsets list:

# In[6]:


lemmas = happy.synsets[0].lemmas()

lemmas


# > In this case, lemmas returned a list of one Lemma element. We can now check whether the database has any corresponding antonyms for that Lemma: 

# In[7]:


lemmas[0].antonyms()


# > The result is list of Lemmas representing the antonym(s). Here, we see that the one antonym for 'happy' in the database is 'unhappy'. 

# [ Restart Kernel ]  
#   
# ### Readability Assessment with Textatistic  
#   
# An interesting use of natural language processing is assessing text readability, which is
# affected by the vocabulary used, sentence structure, sentence length, topic and more.
# While writing this book, we used the paid tool Grammarly to help tune the writing and
# ensure the text’s readability for a wide audience.  
#   
# In this section, we’ll use the **Textatistic library** to assess readability.25 There are
# many formulas used in natural language processing to calculate readability. Textatistic uses
# five popular readability formulas—Flesch Reading Ease, Flesch-Kincaid, Gunning Fog,
# Simple Measure of Gobbledygook (SMOG) and Dale-Chall.  
#   
# ### Install Textatistic  
#   
# To install Textatistic, open your Anaconda Prompt (Windows), Terminal (macOS/
# Linux) or shell (Linux), then execute the following command:  
# > pip install textatistic  
#   
# Windows users might need to run the Anaconda Prompt as an Administrator for proper
# software installation privileges. To do so, right-click Anaconda Prompt in the start menu
# and select More > Run as administrator.   

# In[1]:


pip install textatistic


# ### Calculating Statistics and Readability Scores  
#   
# First, let’s load Romeo and Juliet into the text variable:  

# In[2]:


from pathlib import Path

text = Path('RomeoAndJuliet.txt').read_text()


# Calculating statistics and readability scores requires a **Textatistic** object that’s initialized with the text you want to assess:

# In[3]:


from textatistic import Textatistic

readability = Textatistic(text)


# Textatistic method **dict** returns a dictionary containing various statistics and the readability scores:

# In[4]:


get_ipython().run_line_magic('precision', '3')


# In[5]:


readability.dict()


# Each of the values in the dictionary is also accessible via a Textatistic property of
# the same name as the keys shown in the preceding output. The statistics produced include:  
# * char_count—The number of characters in the text.  
# * word_count—The number of words in the text.  
# * sent_count—The number of sentences in the text.  
# * sybl_count—The number of syllables in the text.  
# * notdalechall_count—A count of the words that are not on the Dale-Chall list, which is a list of words understood by 80% of 5th graders.27 The higher this number is compared to the total word count, the less readable the text is considered to be.  
# * polysyblword_count—The number of words with three or more syllables.  
# * flesch_score—The Flesch Reading Ease score, which can be mapped to a grade level. Scores over 90 are considered readable by 5th graders. Scores under 30 require a college degree. Ranges in between correspond to the other grade levels.  
# * fleschkincaid_score—The Flesch-Kincaid score, which corresponds to a specific grade level.  
# * gunningfog_score—The Gunning Fog index value, which corresponds to a specific grade level.  
# * smog_score—The Simple Measure of Gobbledygook (SMOG), which corresponds to the years of education required to understand text. This measure is considered particularly effective for healthcare materials.  
# * dalechall_score—The Dale-Chall score, which can be mapped to grade levels from 4 and below to college graduate (grade 16) and above. This score considered to be most reliable for a broad range of text types.  
#   
# You can learn about each of these readability scores produced here and several others at  
# > https://en.wikipedia.org/wiki/Readability  
#   
# The Textatistic documentation also shows the readability formulas used:  
# > http://www.erinhengel.com/software/textatistic/

# In[6]:


readability.word_count / readability.sent_count # sentence length


# In[7]:


readability.char_count / readability.word_count # word length


# In[8]:


readability.sybl_count / readability.word_count # syllables


# [ Restart Kernel ]  
#   
# ### Named Entity Recognition with spaCy  
#   
# NLP can determine what a text is about. A key aspect of this is **named entity recognition**, which attempts to locate and categorize items like dates, times, quantities, places, people, things, organizations and more. In this section, we’ll use the named entity recognition capabilities in the **spaCy NLP library** to analyze text.  
#   
# ### Install spaCy
#   
# To install spaCy, open your Anaconda Prompt (Windows), Terminal (macOS/Linux) or shell (Linux), then execute the following command:  

# In[4]:


get_ipython().run_line_magic('pip', 'install spacy')


# Windows users might need to run the Anaconda Prompt as an Administrator for proper software installation privileges. To do so, right-click Anaconda Prompt in the start menu and select **More > Run as administrator**.  
#   
# Once the install completes, you also need to execute the following command, so spaCy can download additional components it needs for processing English (en) text:  

# In[5]:


get_ipython().system('python -m spacy download en_core_web_sm')


# ### Loading the Language Model  
#   
# The first step in using spaCy is to load the language model representing the natural language of the text you’re analyzing. To do this, you’ll call the spacy module’s **load function**. Let’s load the English model that we downloaded above:   

# In[7]:


import spacy

nlp = spacy.load('en_core_web_sm')


# The spaCy documentation recommends the variable name nlp.  
#   
# ### Creating a spaCy Doc(())  
#   
# Next, you use the nlp object to create a spaCy **Doc** object representing the document to process. Here we used a sentence from the introduction to the World Wide Web in many of our books:  

# In[8]:


document = nlp('In 1994, Tim Berners-Lee founded the ' + 'World Wide Web Consortium (W3C), devoted to ' + 'developing web technologies')


# ### Getting the Named Entities  
#   
# The Doc object’s **ents property** returns a tuple of **Span** objects representing the named
# entities found in the Doc. Each Span has many properties.34 Let’s iterate through the Spans
# and display the text and label_ properties:

# In[9]:


for entity in document.ents:
    print(f'{entity.text}: {entity.label_}')


# Each Span’s **text property** returns the entity as a string, and the **label_ property** returns a string indicating the entity’s kind. Here, spaCy found three entities representing a DATE (1994), a PERSON (Tim Berners-Lee) and an ORG (organization; the World Wide Web Consortium). To learn more about spaCy, take a look at its Quickstart guide at  
# > https://spacy.io/usage/models#section-quickstart

# ### Similarity Detection with spaCy  
#   
# **Similarity detection** is the process of analyzing documents to determine how alike they
# are. One possible similarity detection technique is word frequency counting. For example,
# some people believe that the works of William Shakespeare actually might have been written by Sir Francis Bacon, Christopher Marlowe or others. Comparing the word frequencies of their works with those of Shakespeare can reveal writing-style similarities.
# Various machine-learning techniques we’ll discuss in later chapters can be used to
# study document similarity. However, as is often the case in Python, there are libraries such
# as spaCy and Gensim that can do this for you. Here, we’ll use spaCy’s similarity detection
# features to compare Doc objects representing Shakespeare’s Romeo and Juliet with Christopher Marlowe’s Edward the Second. You can download Edward the Second from Project
# Gutenberg as we did for Romeo and Juliet earlier in the chapter.

# In[11]:


from pathlib import Path
document1 = nlp(Path('RomeoAndJuliet.txt').read_text())
document2 = nlp(Path('RomeoAndJuliet.txt').read_text())


# ### Comparing the Books’ Similarity  
#   
# Finally, we use the Doc class’s **similarity method** to get a value from 0.0 (not similar) to
# 1.0 (identical) indicating how similar the documents are:

# In[12]:


document1.similarity(document2)


# In[ ]:




