from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score, precision_score, recall_score

# Load model
clf = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")

df = pd.read_csv("sample.csv")
df["label"] = (df["target"] >= 0.5).astype(int)

batch_size = 100
preds = []

for i in range(0, len(df), batch_size):
    batch_texts = df["comment_text"].iloc[i:i+batch_size].tolist()
    batch_preds = clf(batch_texts, truncation=True, top_k=None)
    
    for pred in batch_preds:
        # Get the score for hate label
        toxic_score = next(x['score'] for x in pred if x['label'] == 'hate')
        preds.append(1 if toxic_score > 0.5 else 0)
    
    if (i + batch_size) % 1000 == 0:
        print(f"Processed {i + batch_size} comments...")

df['pred'] = preds
print(f"\nTotal comments processed: {len(df)}")

# Overall Performance Metrics
accuracy = accuracy_score(df["label"], df['pred'])
precision, recall, f1, _ = precision_recall_fscore_support(df["label"], df['pred'], average="binary", zero_division=0)

print("\nOverall Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Identity columns
identity_columns = [
    'male', 'female', 'transgender', 'other_gender',
    'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation',
    'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion',
    'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity',
    'physical_disability', 'intellectual_or_learning_disability', 
    'psychiatric_or_mental_illness', 'other_disability'
]

print("\n" + "="*60)
print("Identity Bias Analysis")
print("="*60)

bias_results = []

for identity in identity_columns:
    # Filter for comments mentioning this identity (value > 0)
    identity_comments = df[df[identity] > 0]
    
    if len(identity_comments) == 0:
        continue
    
    # Calculate metrics for THIS identity subset
    identity_f1 = f1_score(identity_comments['label'], identity_comments['pred'], zero_division=0)
    identity_precision = precision_score(identity_comments['label'], identity_comments['pred'], zero_division=0)
    identity_recall = recall_score(identity_comments['label'], identity_comments['pred'], zero_division=0)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        identity_comments['label'], 
        identity_comments['pred'],
        labels=[0, 1]
    ).ravel()
    
    # Calculate FPR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    bias_results.append({
        'Identity': identity,
        'Count': len(identity_comments),
        'F1': identity_f1,
        'Precision': identity_precision,
        'Recall': identity_recall,
        'False_Positive_Rate': fpr,
    })

    print(f"""
        {identity.upper()}
        Comments: {len(identity_comments)}
        F1 Score: {identity_f1:.4f}
        Precision: {identity_precision:.4f}
        Recall: {identity_recall:.4f}
        False Positive Rate (on non-toxic): {fpr:.4f}""")

bias_df = pd.DataFrame(bias_results)
bias_df.to_csv("dynabench_bias_by_identity_10000.csv", index=False)
