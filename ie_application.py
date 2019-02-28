import en_core_web_sm
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, Tree
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet


# Tokenize sentences
def tokenize_sentence(paragraph):
    sentence_list = sent_tokenize(paragraph)
    return sentence_list


# Tokenize words
def tokenize_words(sentence):
    #tokenizer = RegexpTokenizer('\w+')
    #token_list = tokenizer.tokenize(sentence)
    token_list = word_tokenize(sentence)
    return token_list


# Lemmatize the words to extract lemma as features
def lemmatize_words(sentences):
    lemma_list = []
    lemmatizer = WordNetLemmatizer()

    for words in sentences:
        for w in words:
            lemma = lemmatizer.lemmatize(w, 'v')
            lemma_list.append(lemma)

    return lemma_list


# Stem words to get root form
def stem_word(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)


# Part-of-Speech(POS) tag the words to extract POS tag features
def get_pos_tags(tokens):
    tags = pos_tag(tokens)
    return tags


# Perform dependency parsing or full-syntactic parsing to parse tree based patterns as features
def generate_parse_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [generate_parse_tree(child) for child in node.children])
    else:
        return node.orth_


# Using WordNet, extract hypernyms
def get_hypernyms(word):
    hypernyms = []

    for ss in wordnet.synsets(word):
        for hyper in ss.hypernyms():
            hypernyms.append(hyper.name().split('.')[0])

    return hypernyms


# Using WordNet, extract hyponyms
def get_hyponyms(word):
    hyponyms = []
    for ss in wordnet.synsets(word):
        for hypo in ss.hyponyms():
            hyponyms.append(hypo.name().split('.')[0])
    return hyponyms

# Using WordNet, extract meronyms
def get_meronyms(word):
    meronyms = []
    for ss in wordnet.synsets(word):
        for mero in ss.part_meronyms():
               meronyms.append(mero.name().split('.')[0])
    return meronyms


# Using WordNet, extract holonyms
def get_holonyms(word):
    holonyms = []
    for ss in wordnet.synsets(word):
        for holo in ss.part_holonyms():
            holonyms.append(holo.name().split('.')[0])
    return holonyms


# Using WordNet, extract synonyms
def get_synonyms(word):
    synonym = []
    for ss in wordnet.synsets(word):
        for l in ss.lemmas():
            synonym.append(l.name())
    synonym_set = set(synonym)
    synonym = list(synonym_set)
    return synonym


# Print the following code
def print_on_console():
    print("Press 1 :: Tokenize the entered sentence/s into words")
    print("Press 2 :: Lemmatize the words to extract lemmas as features")
    print("Press 3 :: POS Tagging of words")
    print("Press 4 :: Perform dependency parsing or full-syntactic parsing")
    print("Press 5 :: Extract hypernyms, hyponyms, meronyms AND holonyms")
    print("Press 0 :: To continue to Task 4")

    choice = input("Enter your choice :: ")
    return choice


# Get all related words for all template
def get_all_related_template_words(all_templates):
    all_words = []
    lemmatizer = WordNetLemmatizer()
    for template in all_templates:
        template = lemmatizer.lemmatize(template, 'v')
        template_words = []
        template_words.extend(get_synonyms(template))
        template_words.extend(get_hypernyms(template))
        template_words.extend(get_hyponyms(template))
        template_words.extend(get_meronyms(template))
        template_words.extend(get_holonyms(template))
        all_words.append((template, template_words))
    return all_words


# Match a sentence to its corresponding template
def template_matching(roots):
    matched_templates = []
    lemmatizer = WordNetLemmatizer()
    for root in roots:
        root = lemmatizer.lemmatize(root, 'v')
        for possible_words in all_possible_words:
            if root in possible_words[1]:
                matched_templates.append(possible_words[0])
    return matched_templates


#Relation Extraction Methods
def get_spans(doc):
    spans = list(doc.ents) + list(doc.noun_chunks)
    for span in spans:
        span.merge()


# Name-Name relation
def extract_culprit_victim_relation(doc):
    get_spans(doc)
    relations = []
    for person in filter(lambda x: x.ent_type_ == 'PERSON' or x.ent_type_ == 'ORG', doc):
        if person.dep_ in ('attr', 'nsubj'):
            object = [w for w in person.head.rights if w.dep_ == 'dobj']
            if object:
                object = object[0]
                if (object.pos_ == 'NOUN') or (object.pos_ == 'PROPN' and object.ent_type_== 'PERSON') or (object.pos_ == 'NUM') or (object.pos_ == 'DET'):
                    relations.append((person, object))
                else:
                    relations.append((person, ' '))
            else:
                relations.append((person, ' '))

        elif person.dep_ in ('attr', 'nsubjpass'):
            object = [w for word in person.head.rights for w in word.children if w.dep_ == 'pobj']
            if object:
                object = object[0]
                if (object.pos_ == 'NOUN') or (object.pos_ == 'PROPN' and object.dep_ == 'PERSON') or (
                        object.pos_ == 'NUM') or (object.pos_ == 'DET'):
                    relations.append((object, person))
                else:
                    relations.append((' ', person))
            else:
                relations.append((' ',person))

        elif person.dep_ == 'pobj':
            object = [w for w in person.head.head.lefts if w.dep_ == 'nsubjpass']
            if object:
                object = object[0]
                relations.append((person, object))
            elif person.head.dep_ == 'agent':
                relations.append((person, ' '))
            elif person.head.dep_ == 'prep':
                relations.append((' ', person))

        elif person.dep_ == 'dobj':
            object = [w for w in person.head.lefts if w.dep_ == 'nsubj']
            if object:
                object = object[0]
                relations.append((object, person))
            elif person.head.dep_ == 'ROOT':
                relations.append((' ', person))
            elif person.head.dep_ == 'VERB':
                relations.append((' ', person))
        elif person.ent_type_ == "PERSON" or person.ent_type_ == "ORG":
            relations.append((person, ' '))
    return relations


# extract date event relation
def extract_date_relation(doc):
    get_spans(doc)
    relations = []
    for date in filter(lambda x: x.ent_type_ == 'DATE', doc):
        if date.dep_ == 'nsubj':
            object = [w for w in date.head.rights if w.dep_ == 'dobj']
            if object:
                relations.append((date))
        elif (date.dep_ == 'pobj' or date.dep_ == 'dobj') and date.head.dep_ == 'prep':
            object = [w for w in date.head.head.children if (w.dep_ == 'nsubj') or (w.dep_ == 'nsubjpass')]
            if object:
                object = object[0]
                relations.append((date))
    return relations


# extract location event relation
def extract_location_relation(doc):
    get_spans(doc)
    relations = []
    for location in filter(lambda x: (x.ent_type_ == 'GPE') or (x.ent_type_ == 'LOC'), doc):
        if location.dep_ == 'dobj' and location.head.dep_ == 'prep':
            object = [w for w in location.head.head.children if (w.dep_ == 'nsubj') or (w.dep_ == 'nsubjpass')]
            if object:
                object = object[0]
                if object.pos_ == 'PROPN':
                    relations.append((location))
        elif location.dep_ == 'pobj' and location.head.dep_ == 'prep':
            object = [w for w in location.head.head.children if (w.dep_ == 'agent') or(w.dep_ == 'nsubj') or (w.dep_ == 'nsubjpass')]
            if object:
                relations.append((location))
    return relations


# extract weapon event relation
def extract_weapon_relation(doc):
    get_spans(doc)
    relations = []
    for weapon in filter(lambda x: ((x.dep_ == 'pobj') or (x.dep_ == 'dobj'))
                                   and (x.head.dep_ != 'ROOT')
                                   and (x.head.dep_ != 'VERB')
                                   and(x.ent_type_ != 'DATE')
                                   and (x.ent_type_ != 'GPE')
                                   and ((x.head.dep_ == 'advcl') or (x.head.dep_ == 'prep'))
                                   and (x.ent_type_ != 'LOC'), doc):
        subject = [w for w in weapon.head.head.children if (w.dep_ == 'nsubj') or (w.dep_ == 'nsubjpass')]
        if subject:
            subject = subject[0]
            relations.append((weapon))
    return relations


# extract money event relation
def extract_money_relations(doc):
    get_spans(doc)
    relations = []
    for money in filter(lambda x: x.ent_type_ == 'MONEY', doc):
        if money.dep_ in ('attr', 'dobj'):
            subject = [w for w in money.head.lefts if w.dep_ == 'nsubj']
            if subject:
                subject = subject[0]
                relations.append((subject, money))
        elif money.dep_ == 'pobj' and money.head.dep_ == 'prep':
            relations.append((money.head.head, money))
    return relations


# extract robbed item event relation
def extract_rob_item_relations(doc):
    get_spans(doc)
    relations = []
    for item in filter(lambda x: (x.pos_ == 'NOUN' and x.head.pos_ == 'VERB') or x.ent_type_ == 'MONEY', doc):
        if item.dep_ in ('attr', 'dobj'):
            if item.head.dep_ == 'conj':
                subject = [w for w in item.head.lefts if w.dep_ == 'nsubj']
                if subject:
                    subject = subject[0]
                    relations.append((subject, item))
                else:
                    relations.append(('', item))
            else:
                relations.append(('', item))
        elif item.dep_ == 'pobj' and item.head.dep_ == 'prep':
            relations.append((item.head.head, item))
        else:
            relations.append((' ', item))
    return relations


# extract criminal crime event relation
def extract_criminal_crime_relation(doc):
    get_spans(doc)
    relations = []
    for person in filter(lambda x: x.ent_type_ == 'PERSON' or x.ent_type_ == 'ORG', doc):
        if person.dep_ == 'nsubj':
            object = [w for w in person.head.rights if w.dep_ == 'pobj']
            if object:
                relations.append((person, object))
            else:
                relations.append((person, person.head))

        if person.dep_ == 'nsubjpass':
            object = [w for w in person.head.lefts if w.dep_ == 'dobj']
            if object:
                relations.append((person, object))
            else:
                relations.append((person, person.head))

        elif (person.dep_ == 'pobj' or person.dep_ == 'dobj') and person.head.dep_ == 'prep':
            object = [w for w in person.head.head.children if (w.dep_ == 'nsubj') or (w.dep_ == 'nsubjpass')]
            if object:
                object = object[0]
                relations.append((person, object))
            else:
                relations.append((person, ' '))

    return relations


# extract attack damage relation
def extract_attack_damage_relation(doc):
    get_spans(doc)
    relations = []
    for person in filter(lambda x: (x.pos_ == 'PROPN' and x.ent_type_ != 'DATE')
                                   or x.pos_ == 'DET'
                                   or x.ent_type_ == 'PEROSN'
                                   or x.ent_type_ == 'ORG', doc):
        print(person)
        if person.dep_ in ('attr', 'nsubj'):
            object = [w for w in person.head.rights if w.dep_ == 'dobj']
            if object:
                object = object[0]
                relations.append((person, object))
            else:
                relations.append((person, ' '))

        elif person.dep_ in ('attr', 'nsubjpass'):
            object = [w for word in person.head.rights for w in word.children if w.dep_ == 'pobj']
            if object:
                object = object[0]
                relations.append((person, object))
            else:
                relations.append((person, ' '))

        elif person.dep_ == 'pobj':
            object = [w for w in person.head.head.lefts if w.dep_ == 'nsubjpass']
            if object:
                object = object[0]
                relations.append((object, person))
            elif (person.head.pos_ == 'PROPN' or person.head.pos_ == 'ADP') and person.head.head.dep_ == 'ROOT':
                relations.append((' ', person))
            elif person.head.dep_ == 'agent':
                relations.append((person, ' '))
            elif person.head.dep_ == 'prep':
                relations.append((' ', person))
            else:
                relations.append((' ', person))

        elif person.dep_ == 'dobj':
            object = [w for w in person.head.lefts if w.dep_ == 'nsubj']
            if object:
                object = object[0]
                relations.append((object, person))
            elif (person.head.pos_ == 'PROPN' or person.head.pos_=='ADP') and person.head.head == 'ROOT':
                relations.append((' ', person))
            elif person.head.dep_ == 'ROOT':
                relations.append((' ', person))
            elif person.head.dep_ == 'VERB':
                relations.append((' ', person))
            else:
                relations.append((' ', person))
        elif person.pos_ == "PROPN":
            relations.append((person, ' '))

    return relations


# extract date event relation
def extract_duration_relation(doc):
    get_spans(doc)
    relations = []
    for date in filter(lambda x: x.dep_ == 'pobj' or x.dep_ == 'dobj', doc):
        if date.pos_ == 'NUM':
            relations.append(('',date))

    return relations


input_sentence = input("Enter your sentence here:: ")
print("\nSentence :: ", input_sentence)

nlp = en_core_web_sm.load()
doc_nlp = nlp((str)(input_sentence))

sentences = []
words = []
lemmas = []
pos_tags = []
all_hypernyms = []
all_hyponyms = []
all_meronyms = []
all_holonyms = []
choice = print_on_console()

while choice != '0':
    if choice == '1':
        sentences = tokenize_sentence(input_sentence)
        for sentence in sentences:
            words.append(tokenize_words(sentence))
        print("Tokenization :: ", words)
    elif choice == '2':
        lemmas = lemmatize_words(words)
        print("Lemmatization :: ", lemmas)
    elif choice == '3':
        pos_tags = get_pos_tags(lemmas)
        print("POS Tagging :: ", pos_tags)
    elif choice == '4':
        nltk_tree = [generate_parse_tree(sent.root).pretty_print() for sent in doc_nlp.sents]
    elif choice == '5':
        for lemma in lemmas:
            hypernyms = []
            hypernyms = get_hypernyms(lemma)
            all_hypernyms.append((lemma, hypernyms))

            hyponyms = []
            hyponyms = get_hyponyms(lemma)
            all_hyponyms.append((lemma, hyponyms))

            meronyms = []
            meronyms = get_meronyms(lemma)
            all_meronyms.append((lemma, meronyms))

            holonyms = []
            holonyms = get_holonyms(lemma)
            all_holonyms.append((lemma, holonyms))

        print("Hypernyms :: ", all_hypernyms)
        print("Hyponyms :: ", all_hyponyms)
        print("Meronyms :: ", all_meronyms)
        print("Holonyms :: ", all_holonyms)

    choice = input("\nEnter your choice :: ")

# Task 4 - Template Extraction and Filling
templates_dict = {'murder': {'Date': '12/11/2018', 'Location': 'USA', 'Culprit': '', 'Victim': '',
                             'Murder_Weapon': ''},
                  'kidnap':  {'Date': '12/11/2018', 'Location': 'USA', 'Culprit': '', 'Victim': '',
                              'Ransom': ''},
                  'rob': {'Date': '12/11/2018', 'Location': 'USA', 'Culprit': '', 'Victim': '',
                              'Stolen_Item': 'Cash'},
                  'bail': {'Date': '12/11/2018', 'Location': 'USA', 'Criminal_Name': ' ', 'Crime': '',
                           'Bail_Amount': ''},
                  'lawsuit': {'Date': '12/11/2018', 'Location': 'USA', 'From': '', 'To': '', 'Status': 'Pending',
                             'Amount': ' '},
                  'attack': {'Date': '12/11/2018', 'Location': 'USA', 'Organization': '', 'Damage': '',
                             'Attack_Weapon': ''},
                  'punish': {'Date': '12/11/2018', 'Location': 'USA', 'Criminal_Name': '', 'Crime': '',
                             'Duration': ''},
                  'hack': {'Date': '12/11/2018', 'Location': 'USA', 'Victim': '', 'Damage': ''},
                  'arson': {'Date': '12/11/2018', 'Location': 'USA', 'Culprit': '', 'Victim': ''},
                  'bribe': {'Date': '12/11/2018', 'Location': 'USA', 'Criminal_Name': ''}
                  }

template_names = []
for key in templates_dict:
    template_names.append(key)

all_possible_words = get_all_related_template_words(template_names)

roots = []
for tree in doc_nlp.sents:
    roots.append(tree.root.orth_)
for word_tag in doc_nlp:
    if word_tag.pos_ == 'NOUN' or word_tag.pos_ == 'VERB':
        roots.append(word_tag.text)

matched_templates = template_matching(roots)
matched_set = set(matched_templates)
matched_templates = list(matched_set)

if len(matched_templates) == 0:
    print(" No matching template found. Please enter sentences related to the domain.")

for template in matched_templates:
    rel = []

    if template == 'murder':
        print("murder {Date: ___ , Location: ___ , Culprit: ___ , Victim: ___ , Murder_Weapon: ___ }")

        rel = extract_date_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Date'] = r1.text

        rel = extract_location_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Location'] = r1.text

        rel = extract_culprit_victim_relation(doc_nlp)
        for r1, r2 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Culprit'] = r1.text
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Victim'] = r2.text

        relations = extract_weapon_relation(doc_nlp)
        for r1 in relations:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Murder_Weapon'] = r1.text

        print(template,templates_dict[template], "\n")

    if template == 'kidnap':
        print("kidnap {Date: ___ , Location: ___ , Culprit: ___ , Victim: ___ , Ransom: ___ }")

        rel = extract_date_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Date'] = r1.text

        rel = extract_location_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ':
                templates_dict[template]['Location'] = r1.text

        rel = extract_culprit_victim_relation(doc_nlp)
        for r1, r2 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Culprit'] = r1.text
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Victim'] = r2.text

        rel = extract_money_relations(doc_nlp)
        for r1, r2 in rel:
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Ransom'] = r2.text

        print(template, templates_dict[template])

    if template == 'bail':
        print("bail {Date: ___ , Location: ___ , Culprit: ___ , Crime: ___ , Bail_Amount: ___ }")
        rel = extract_date_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Date'] = r1.text

        rel = extract_location_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Location'] = r1.text

        rel = extract_criminal_crime_relation(doc_nlp)
        for r1, r2 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Criminal_Name'] = r1.text
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Crime'] = r2.text

        rel = extract_money_relations(doc_nlp)
        for r1, r2 in rel:
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Bail_Amount'] = r2.text

        print(template, templates_dict[template])

    if template == 'attack':
        print("attack {Date: ___ , Location: ___ , Organization: ___ , Damage: ___ , Attack_Weapon: ___ }")

        rel = extract_date_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Date'] = r1.text

        rel = extract_location_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Location'] = r1.text

        rel = extract_attack_damage_relation(doc_nlp)
        for r1, r2 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Organization'] = r1.text
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Damage'] = r2.text

        relations = extract_weapon_relation(doc_nlp)
        for r1 in relations:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Attack_Weapon'] = r1.text

        print(template,templates_dict[template], "\n")

    if template == 'rob':
        print("rob {Date: ___ , Location: ___ , Culprit: ___ , Victim: ___ , Stolen_Item: ___ }")

        rel = extract_date_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Date'] = r1.text

        rel = extract_location_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Location'] = r1.text

        rel = extract_culprit_victim_relation(doc_nlp)
        for r1, r2 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Culprit'] = r1.text
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Victim'] = r2.text

        rel = extract_rob_item_relations(doc_nlp)
        for r1, r2 in rel:
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Stolen_Item'] = r2.text

        print(template, templates_dict[template])

    if template == 'lawsuit':
        print("lawsuit {Date: ___ , Location: ___ , From: ___ , To: ___ , Status: ___, Amount: ___ }")

        rel = extract_date_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Date'] = r1.text

        rel = extract_location_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Location'] = r1.text

        rel = extract_culprit_victim_relation(doc_nlp)
        for r1, r2 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['From'] = r1.text
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['To'] = r2.text

        relations = extract_money_relations(doc_nlp)
        for r1, r2 in relations:
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Amount'] = r2.text

        print(template, templates_dict[template])

    if template == 'punish':
        print("punish {Date: ___ , Location: ___ , Criminal_Name: ___ , Crime: ___ , Duration: ___}")

        rel = extract_date_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Date'] = r1.text

        rel = extract_location_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Location'] = r1.text

        rel = extract_criminal_crime_relation(doc_nlp)
        for r1, r2 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Criminal_Name'] = r1.text
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Crime'] = r2.text

        rel = extract_duration_relation(doc_nlp)
        for r1, r2 in rel:
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Duration'] = r2.text

        print(template, templates_dict[template])

    if template == 'hack':
        print("hack {Date: ___ , Location: ___ ,  Victim: ___ , Damage: ___ }")

        rel = extract_date_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Date'] = r1.text

        rel = extract_location_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Location'] = r1.text

        rel = extract_culprit_victim_relation(doc_nlp)
        for r1, r2 in rel:
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Victim'] = r2.text

        rel = extract_rob_item_relations(doc_nlp)
        for r1, r2 in rel:
            if str(r2) != ' ' or len(r1) > 1:
                templates_dict[template]['Damage'] = r2.text

        print(template, templates_dict[template])

    if template == 'bribe':
        print("bribe {Date: ___ , Location: ___ , Culprit: ___}")

        rel = extract_date_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Date'] = r1.text

        rel = extract_location_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Location'] = r1.text

        rel = extract_culprit_victim_relation(doc_nlp)
        for r1, r2 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Criminal_Name'] = r1.text

        print(template, templates_dict[template])

    if template == 'arson':
        print("rob {Date: ___ , Location: ___ , Culprit: ___ , Victim: ___ }")

        rel = extract_date_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Date'] = r1.text

        rel = extract_location_relation(doc_nlp)
        for r1 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Location'] = r1.text

        rel = extract_culprit_victim_relation(doc_nlp)
        for r1, r2 in rel:
            if str(r1) != ' ' or len(r1) > 1:
                templates_dict[template]['Culprit'] = r1.text
            if str(r2) != ' ' or len(r2) > 1:
                templates_dict[template]['Victim'] = r2.text

        print(template, templates_dict[template])
