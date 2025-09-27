import xml.etree.ElementTree as ET
import csv

# SemEval 2016 Task 5 dataset is only available in XML format -> preprocess to TSV
def preprocess_xml_to_tsv(xml_path, out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    with open(out_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerow(['id', 'text', 'labels', 'labels_phrases'])

        index = 0
        for review in root.findall('.//Review'):
            for sentence in review.findall('.//sentence'):
                text_element = sentence.find('text')
                if text_element is None:
                    continue
                text = text_element.text
                opinions = sentence.find('Opinions')
                labels = []
                labels_phrases = []

                if opinions is not None and len(opinions) > 0:
                    for opinion in opinions.findall('Opinion'):
                        target = opinion.attrib.get('target', 'NULL')
                        category = opinion.attrib.get('category', '')
                        polarity = opinion.attrib.get('polarity', '')
                        label = f'({category}, {polarity.upper()})'
                        label_phrase = f'({category}, {polarity.upper()}, \"{target}\")'
                        labels.append(label)
                        labels_phrases.append(label_phrase)
                else:
                    continue

                writer.writerow([
                    index,
                    text,
                    str(labels),
                    str(labels_phrases)
                ])
                index += 1

# https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools

xml_path_test = r'D:\Uni\Masterarbeit Code\jakob_finetuning\data\rest-16-spanish\test.xml'
out_path_test = r'D:\Uni\Masterarbeit Code\jakob_finetuning\data\rest-16-spanish\test.tsv'

xml_path_train = r'D:\Uni\Masterarbeit Code\jakob_finetuning\data\rest-16-spanish\train.xml'
out_path_train = r'D:\Uni\Masterarbeit Code\jakob_finetuning\data\rest-16-spanish\train.tsv'

preprocess_xml_to_tsv(xml_path_test, out_path_test)
preprocess_xml_to_tsv(xml_path_train, out_path_train)