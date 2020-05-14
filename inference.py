import spacy
import os

print(os.getcwd())
prdnlp = spacy.load("Company")

a =[]
import textract
text = str(textract.process('/home/deepak/deepak_old_system/Documents/Documents_New/MintMesh/Datasets/all_resumes/Gaurav Bhadan.pdf', method='pdfminer'))


for i in text.split("\n"):
	a.append(i.replace("	"," "))

a = " ".join(a)

print("ENTITIES\n\n\n")

doc = prdnlp(a)
for ent in doc.ents:

    print(ent.text, ent.start_char, ent.end_char, ent.label_)

