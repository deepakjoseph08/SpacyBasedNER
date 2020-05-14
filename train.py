import spacy
import random
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score

TRAIN_DATA =[('Currently Working as Sr Software Engineer in Virtusa Technologies India Private Limited Hyderabad, From Sep 2015 to till now.', {'entities': [(45, 87, 'Company')]}), ('Worked as Sr Software Engineer in Honeywell Technology Solutions Hyderabad on payroll of Mindteck (India) Limited Bangalore, From March 2015 to till now.', {'entities': [(34, 74, 'Company')]}), ('Worked as Software Engineer in Mobilerays Hyderabad from Oct 2010 to March 2015.', {'entities': [(31, 51, 'Company')]}), ('Post-Graduation: Masters of Computer Applications from Gayatri Vidya Parishad  College for PG Courses affiliated to Andhra University with 67.99%  marks in the year 2013', {'entities': [(33, 49, 'Company')]}), ('Working as a PHP programmer in Complitsol (www.complitsol.com) from Dec 2013 to till now', {'entities': [(31, 41, 'Company')]})]

TEST_DATA = [('Currently Working as Sr Software Engineer in Virtusa Technologies India Private Limited Hyderabad, From Sep 2015 to till now.', {'entities': [(45, 87, 'Company')]}), ('Worked as Sr Software Engineer in Honeywell Technology Solutions Hyderabad on payroll of Mindteck (India) Limited Bangalore, From March 2015 to till now.', {'entities': [(34, 74, 'Company')]}), ('Worked as Software Engineer in Mobilerays Hyderabad from Oct 2010 to March 2015.', {'entities': [(31, 51, 'Company')]}), ('Post-Graduation: Masters of Computer Applications from Gayatri Vidya Parishad  College for PG Courses affiliated to Andhra University with 67.99%  marks in the year 2013', {'entities': [(33, 49, 'Company')]}), ('Working as a PHP programmer in Complitsol (www.complitsol.com) from Dec 2013 to till now', {'entities': [(31, 41, 'Company')]})]



def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


prdnlp = train_spacy(TRAIN_DATA, 20)

# Save our trained Model
modelfile = input("Enter your Model Name: ")
prdnlp.to_disk(modelfile)


tp=0
tr=0
tf=0

ta=0
c=0    

for text,annot in TEST_DATA:

#         f=open("resume"+str(c)+".txt","w")
    doc_to_test=prdnlp(text)
#         print("doc_to_test")
#         print(doc_to_test)
    d={}
    for ent in doc_to_test.ents:
        d[ent.label_]=[]
    for ent in doc_to_test.ents:
        d[ent.label_].append(ent.text)
#             print("ent.label {} ==> ent.text {}".format(ent.label,ent.text))

#         for i in set(d.keys()):

#             f.write("\n\n")
#             f.write(i +":"+"\n")
#             for j in set(d[i]):
#                 f.write(j.replace('\n','')+"\n")
    d={}
#         print("printing doc ents")
#         print(doc_to_test.ents)
    
    for ent in doc_to_test.ents:
        d[ent.label_]=[0,0,0,0,0,0]
    for ent in doc_to_test.ents:
        doc_gold_text= prdnlp.make_doc(text)
        gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
        y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
        y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
        # print(text,y_pred)
        if(d[ent.label_][0]==0):
            print("For Entity "+ent.label_+"\n")   
            #f.write(classification_report(y_true, y_pred)+"\n")
            (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
            a=accuracy_score(y_true,y_pred)
            d[ent.label_][0]=1
            d[ent.label_][1]+=p
            d[ent.label_][2]+=r
            d[ent.label_][3]+=f
            d[ent.label_][4]+=a
            d[ent.label_][5]+=1
    c+=1
print(d)
for i in d:
    print("\n For Entity "+i+"\n")
    print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
    print("Precision : "+str(d[i][1]/d[i][5]))
    print("Recall : "+str(d[i][2]/d[i][5]))
    print("F-score : "+str(d[i][3]/d[i][5]))
