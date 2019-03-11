----------------------------------------------------------------------------------------------------------------------------------------
Problem Description
----------------------------------------------------------------------------------------------------------------------------------------
Implement an Information Extraction application using NLP features
Project comprised of 4 stages:
Stage 1: Creation of at least 10 unique information templates with cumulative 40 attributes
Stage 2: Creation of corpus of at least 50,000 words
Stage 3: Implementation of NLP techniques to extract NLP features:
         Tokenization
         Lemmatization
         Part-Of-Speech Tagging
         Dependency Parsing
         Word Relations : Hypernyms, Hyponyms, Holonyms, Meronyms
Stage 4: Implementation of  a machine-learning, statistical, or heuristic based approach to extract filled information templates from the corpus

----------------------------------------------------------------------------------------------------------------------------------------
Proposed Solution
----------------------------------------------------------------------------------------------------------------------------------------
Selected Domain : Crime
Programming Language: Python 3.6
Open Source Libraries: NLTK, Spacy
Manual Creation of 10 templates with the required properties by exploring various authentic resources for crime reports
Using text scraping and manual exploration, collect the required corpus
Using open source libraries such as NLTK and Spacy, extract NLP features
Generate Heuristics for each template, the extracted NLP features and Named Entity Recognition perform Template Matching and Template Filling

----------------------------------------------------------------------------------------------------------------------------------------
How To run?
----------------------------------------------------------------------------------------------------------------------------------------
Unzip the folder and follow the steps below

Steps(in Windows command line)

1. To navigate to the destination folder
   cd DownloadsProjectFinalSubmission-ID91\source_code 

2. To execute the python file
   python ie_application.py

3. Follow the instructions on the console:
   Enter your sentence here::

4. After entering the sentence, select any of the following instructions:
   Press 1 :: Tokenize into sentence and words
   Press 2 :: Lemmatize the words to extract lemmas as features
   Press 3 :: POS Tagging of words
   Press 4 :: Perform dependency parsing or full-syntactic parsing
   Press 5 :: Extract hypernyms, hyponyms, meronyms AND holonyms
   Press 0 :: To continue to Task 4
   Enter your choice ::
