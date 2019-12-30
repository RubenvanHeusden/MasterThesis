import loremipsum
import random
import csv



num_entries = 15

sample_text = loremipsum.generate(num_entries, loremipsum.ParagraphLength.MEDIUM).split("\n\n")

classes = [random.randint(1, 10) if random.random()<0.5 else "None" for _ in range(num_entries)]

header_line = ['document', 'class']




with open('dbase/dummy_dataset.csv','w', newline='') as csv_file:
    csvwriter = csv.writer(csv_file, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(header_line)
    for i in range(num_entries):
        csvwriter.writerow([sample_text[i]]+[classes[i]])