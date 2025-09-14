import pandas as pd
import deepl
import os
import re
import ast

"""
Translate the whole rest-16 dataset (train and test) to German using DeepL API.
https://github.com/DeepLcom/deepl-python/tree/main
"""
class DeepLTranslator:
    def __init__(self, api_key):
        self.client = deepl.DeepLClient(api_key)
    
    def check_usage(self):
        usage = self.client.get_usage()
        print(f"Current usage: {usage.character.count} / {usage.character.limit} characters")
        
        if usage.character.limit_reached:
            raise Exception("DeepL quota already exceeded!")
        
        remaining = usage.character.limit - usage.character.count
        print(f"Remaining: {remaining} characters")

    def translate_text(self, text):
        if not text or text.strip() == "":
            return None
        try:
            result = self.client.translate_text(text, target_lang="DE", source_lang="EN")
            return result.text
        except deepl.QuotaExceededException:
            print("QUOTA EXCEEDED! Translation stopped.")
            raise
        except Exception as e:
            print(f"Error translating: {e}")
            return None
    
    def translate_phrase_in_labels(self, labels_phrases_str):
        """
        Translate phrases within the labels_phrases structure
        Input: "['(RESTAURANT#GENERAL, NEGATIVE, \"\"place\"\")']"
        Output: "['(RESTAURANT#GENERAL, NEGATIVE, \"\"Ort\"\")']"
        """        
        if not labels_phrases_str or labels_phrases_str.strip() == "":
            return labels_phrases_str
            
        try:
            labels_list = ast.literal_eval(labels_phrases_str)
            translated_labels = []
            
            for label_tuple_str in labels_list:
                phrase_match = re.search(r'"([^"]*?)"', label_tuple_str)
                if phrase_match:
                    original_phrase = phrase_match.group(1)

                    if original_phrase and original_phrase != "NULL" and original_phrase.strip() != "":
                        translated_phrase = self.translate_text(original_phrase)
                        
                        if translated_phrase:
                            # Replace the original phrase with translated one
                            new_label_tuple_str = label_tuple_str.replace(f'"{original_phrase}"', f'"{translated_phrase}"')
                            translated_labels.append(new_label_tuple_str)
                        else:
                            # Replace with NULL if translation failed
                            new_label_tuple_str = label_tuple_str.replace(f'"{original_phrase}"', '"NULL"')
                            translated_labels.append(new_label_tuple_str)
                            print(f"Translation failed for phrase '{original_phrase}', using NULL")
                    else:
                        # Keep NULL or empty phrases as-is
                        translated_labels.append(label_tuple_str)
                else:
                    # No phrase found, keep original structure
                    translated_labels.append(label_tuple_str)
        
            result = str(translated_labels)
            return result
            
        except Exception as e:
            print(f"Error translating labels_phrases: {e}")
            return labels_phrases_str
    
    def translate_batch(self, texts):
        translated = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            print(f"Translating text {i+1}/{len(texts)}: {text}...")
            try:
                translated_text = self.translate_text(text)
                if translated_text is not None:
                    translated.append(translated_text)
                    valid_indices.append(i)
                else:
                    print(f"Skipping row {i+1} - translation failed")
            except deepl.QuotaExceededException:
                print(f"Stopped at {i+1}/{len(texts)} due to quota limit")
                break
                
        return translated, valid_indices
    
    def translate_phrases_batch(self, labels_phrases_list):
        translated_labels = []
        
        for i, labels_phrases in enumerate(labels_phrases_list):
            print(f"Translating phrases {i+1}/{len(labels_phrases_list)}: {labels_phrases}...")
            try:
                translated_label = self.translate_phrase_in_labels(labels_phrases)
                translated_labels.append(translated_label)
            except deepl.QuotaExceededException:
                print(f"Stopped at phrase translation {i+1} due to quota limit")
                break
        
        return translated_labels

def translate_dataset_file(input_file, output_file, translator):
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    print(f"Loaded {len(df)} rows")
    
    total_chars = sum(len(str(text)) for text in df['text'])
    print(f"Estimated characters to translate: {total_chars}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        print("Translating text column...")
        translated_texts, valid_indices = translator.translate_batch(df['text'].tolist())
        
        # Create dataframe with only successfully translated rows
        df_translated = df.iloc[valid_indices].copy()
        df_translated['text'] = translated_texts
        df_translated['original_text'] = df.iloc[valid_indices]['text'].tolist()
        
        print("Translating phrases in labels_phrases column...")
        labels_phrases_list = df_translated['labels_phrases'].tolist()
        translated_labels_phrases = translator.translate_phrases_batch(labels_phrases_list)
        df_translated['labels_phrases'] = translated_labels_phrases
        
        df_translated = df_translated.reset_index(drop=True)
        
        df_translated.to_csv(output_file, sep='\t', index=False)
        print(f"Saved to: {output_file}")
        print(f"Successfully translated: {len(df_translated)}/{len(df)} rows")
        
        skipped = len(df) - len(df_translated)
        if skipped > 0:
            print(f"Skipped {skipped} rows due to translation failures")
        
    except deepl.QuotaExceededException:
        print("Cannot continue - DeepL quota exceeded")
        return False
    
    return True

def main():
    API_KEY = input("Enter your DeepL API key: ").strip()
    translator = DeepLTranslator(API_KEY)
    
    try:
        translator.check_usage()
    except Exception as e:
        print(f"Error checking usage: {e}")
        return
    
    base_dir = "D:/Uni/Masterarbeit Code/jakob_finetuning/data"
    input_dir = os.path.join(base_dir, "rest-16")
    output_dir = os.path.join(base_dir, "rest-16-german")

    for input_file, output_file in [("train.tsv", "train.tsv"), ("test.tsv", "test.tsv")]:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)
        
        if os.path.exists(input_path):
            print(f"\nTranslating {input_file}...")
            success = translate_dataset_file(input_path, output_path, translator)
            
            if not success:
                print("Stopping due to quota limit")
                break
    
    print("Translation process finished!")

if __name__ == "__main__":
    main()