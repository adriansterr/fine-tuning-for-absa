import re
import torch

from transformers import StoppingCriteria

REGEX_ASPECTS_ACD = r'\[([^\]]+)\]'
POLARITIES = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
ASPECTS = [
    "AMBIENCE#GENERAL", "DRINKS#PRICES", "DRINKS#QUALITY", "DRINKS#STYLE_OPTIONS",
    "FOOD#PRICES", "FOOD#QUALITY", "FOOD#STYLE_OPTIONS", "LOCATION#GENERAL",
    "SERVICE#GENERAL", "RESTAURANT#GENERAL", "RESTAURANT#PRICES", "RESTAURANT#MISCELLANEOUS"
]

# Adapted from https://github.com/JakobFehle/Fine-Tuning-LLMs-for-ABSA/blob/main/src/utils/evaluation.py

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

def extractAspects(output, task):
    # Validate Output
    if output.count('(') != output.count(')'):
        return []
        
    if task == 'acd':
        pattern_asp = re.compile(REGEX_ASPECTS_ACD)
        matches = pattern_asp.findall(output)
        
        return matches[0].split(', ') if matches else []
        
    elif task == 'acsa':
        aspect_pattern = "|".join(re.escape(a) for a in ASPECTS)
        polarity_pattern = "|".join(POLARITIES)
        # Matches: "<any or none symbol> ASPECT <any symbol> POLARITY <any or none symbol>"
        # For example: (FOOD#QUALITY, POSITIVE), (FOOD#QUALITY: POSITIVE), (FOOD#QUALITY POSITIVE), FOOD#QUALITY: POSITIVE, FOOD#QUALITY POSITIVE, -FOOD#QUALITY POSITIVE
        REGEX_HARDCODED = re.compile(
            rf"\W?\s*({aspect_pattern})\s*\W+\s*({polarity_pattern})\W?",
            re.IGNORECASE
        )
        
        matches = REGEX_HARDCODED.findall(output)
        
        results = []
        for m in matches:
            aspect = m[0]
            polarity = m[1]
            if aspect and polarity:
                results.append([aspect.upper(), polarity.upper()])
        return results
    
def convertLabels(labels, task, label_space):
    false_predictions = []
    conv_l = []
    label_space = sorted(set(lab.split(':')[0] for lab in label_space)) if task == 'acd' else label_space
    for sample in labels:
        conv_s = []
        for pair in sample:
            if task != 'acd':
                pair_str = ':'.join(label.replace('"', '').replace("'", "") for label in pair[:2])
            else:
                pair_str = pair
                
            if pair_str in label_space or task == 'e2e' or task == 'e2e-e':
                conv_s.append(':'.join([pair_str, pair[2]]) if task == 'tasd' else pair_str)
            else:
                false_predictions.append(pair_str)
        conv_l.append(conv_s)

    return conv_l, false_predictions

def subset_recall(pred_labels, gold_labels):
    correct = 0
    for pred, gold in zip(pred_labels, gold_labels):
        # All gold labels must be in pred (ignore extra predictions)
        if all(label in pred for label in gold):
            correct += 1
    return correct / len(gold_labels) if gold_labels else 0.0

def calculateMetrics(predictions, ground_truths):
    tp, fp, fn = 0, 0, 0
    
    for pred, gold in zip(predictions, ground_truths):
        _, gold_copy = pred[:], gold[:]  # Work with copies to avoid modifying original lists
        
        # Calculate True Positives
        for label in pred:
            if label in gold_copy:
                tp += 1
                gold_copy.remove(label)  # Remove the matched label from the gold list
            else:
                fp += 1  # False Positive: label in pred but not in gold
        
        # Remaining items in gold are False Negatives
        fn += len(gold_copy)

    # Precision, recall, F1, and accuracy calculations
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0
    support = tp + fp + fn
    
    return {'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), 'accuracy': round(accuracy, 4), 'support': support}


def createResults(pred_labels, gold_labels, label_space, task):
    
    if task == 'acd':
        # Calculate Micro Metrics
        micro_asp = calculateMetrics(pred_labels, gold_labels)
        label_space_grouped = sorted(list(set([lab.split(':')[0] for lab in label_space])))
        
        # Metrics by Aspects disregarding Polarities
        metrics_asp = []
        for i in range(len(label_space_grouped)):
            pred_labels_subset = [[label for label in pred if label == label_space_grouped[i]] for pred in pred_labels]
            gold_labels_subset = [[label for label in gold if label == label_space_grouped[i]] for gold in gold_labels]
            metrics_asp.append({'aspect': label_space_grouped[i], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})
    
        # micro_asp = calculateMetrics([[label for label in pred] for pred in pred_labels], [[label for label in gold] for gold in gold_labels], task)
        
        macro_asp = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'accuracy': "",
                 'support': micro_asp['support']}
        
        result = {metr['aspect']:metr['metrics'] for metr in metrics_asp}
        result['Micro-AVG'] = micro_asp
        result['Macro-AVG'] = macro_asp

        return result, None, None, None, None
    
    elif task == 'acsa':
        
        # Calculate Micro Metrics; Overall performance (all pairs pooled)
        micro_asp_pol = calculateMetrics(pred_labels, gold_labels)
    
        label_space = list(set([label for labels in gold_labels for label in labels]))
        label_space.sort()

        # Group Aspect labels together ignoring their polarity
        label_space_grouped = [[label for label in label_space if label.split(':')[0] ==  aspect] for aspect in sorted(list(set([lab.split(':')[0] for lab in label_space])))]
        
        # Metrics by Aspects disregarding Polarities
        metrics_asp = []
        for i in range(len(label_space_grouped)):
            pred_labels_subset = [[label.split(':')[0] for label in pred if label == label_space_grouped[i][0].split(':')[0]] for pred in pred_labels]
            gold_labels_subset = [[label.split(':')[0] for label in gold if label == label_space_grouped[i][0].split(':')[0]] for gold in gold_labels]
            metrics_asp.append({'aspect': label_space_grouped[i][0].split(':')[0], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})
    
        micro_asp = calculateMetrics([[label.split(':')[0] for label in pred] for pred in pred_labels], [[label.split(':')[0] for label in gold] for gold in gold_labels])
        
        macro_asp = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_asp]) / (len(label_space_grouped)), 4), 
                 'accuracy': "",
                 'support': micro_asp['support']}
    
        # Metrics by Aspects but Polarities have to match
        metrics_asp_pol = []
        for i in range(len(label_space_grouped)):
            pred_labels_subset = [[label for label in pred if label in label_space_grouped[i]] for pred in pred_labels]
            gold_labels_subset = [[label for label in gold if label in label_space_grouped[i]] for gold in gold_labels]
            metrics_asp_pol.append({'aspect': label_space_grouped[i][0].split(':')[0], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})
    
        macro_asp_pol = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_asp_pol]) / (len(label_space_grouped)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_asp_pol]) / (len(label_space_grouped)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_asp_pol]) / (len(label_space_grouped)), 4), 
                 'accuracy': "",
                 'support': micro_asp_pol['support']}
    
        # Metrics by Aspect-Polarity-Pairs (Classifier Class-Labels)
        metrics_pairs = []
        for i in range(len(label_space)):
            pred_labels_subset = [[label for label in pred if label == label_space[i]] for pred in pred_labels]
            gold_labels_subset = [[label for label in gold if label == label_space[i]] for gold in gold_labels]
            metrics_pairs.append({'aspect': label_space[i], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})
        
        macro_pairs = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_pairs]) / (len(label_space)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_pairs]) / (len(label_space)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_pairs]) / (len(label_space)), 4), 
                 'accuracy': "",
                 'support': micro_asp_pol['support']}

        # Metrics by Polarities
        metrics_pol = []
        
        for i in range(len(POLARITIES)):
            pred_labels_subset = [[label for label in pred if (isinstance(label, str) and label.split(':')[1] == POLARITIES[i])] for pred in pred_labels]
            gold_labels_subset = [[label for label in gold if (isinstance(label, str) and label.split(':')[1] == POLARITIES[i])] for gold in gold_labels]
            metrics_pol.append({'polarity': POLARITIES[i], 'metrics': calculateMetrics(pred_labels_subset, gold_labels_subset)})

        macro_pol = {'precision': round(sum([metrics['metrics']['precision'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'recall': round(sum([metrics['metrics']['recall'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'f1': round(sum([metrics['metrics']['f1'] for metrics in metrics_pol]) / (len(POLARITIES)), 4), 
                 'accuracy': "",
                 'support': micro_asp_pol['support']}
            
        result_asp = {metr['aspect']:metr['metrics'] for metr in metrics_asp}
        result_asp['Micro-AVG'] = micro_asp
        result_asp['Macro-AVG'] = macro_asp
    
        result_asp_pol = {metr['aspect']:metr['metrics'] for metr in metrics_asp_pol}
        result_asp_pol['Micro-AVG'] = micro_asp_pol
        result_asp_pol['Macro-AVG'] = macro_asp_pol
    
        result_pairs = {metr['aspect']:metr['metrics'] for metr in metrics_pairs}
        result_pairs['Micro-AVG'] = micro_asp_pol
        result_pairs['Macro-AVG'] = macro_pairs

        result_pol = {metr['polarity']:metr['metrics'] for metr in metrics_pol}
        result_pol['Micro-AVG'] = micro_asp_pol
        result_pol['Macro-AVG'] = macro_pol

        result_subset_recall = subset_recall(pred_labels, gold_labels)

        return result_asp, result_asp_pol, result_pairs, result_pol, None, result_subset_recall