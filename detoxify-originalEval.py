import pandas as pd
import numpy as np
from detoxify import Detoxify
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

df = pd.read_csv('sample.csv')
print("Dataset Shape:", df.shape)

# initializing Detoxify Original model
model = Detoxify("original")
predictions = []
toxicity_scores = []
for idx, comment in enumerate(df['comment_text'].values):
    try:
        result = model.predict(comment)
        toxicity_scores.append(result['toxicity'])
        predictions.append(1 if result['toxicity'] > 0.5 else 0)  # Binary: 1 = toxic, 0 = non-toxic
    except:
        # Handling errors with problematic comments
        predictions.append(0)
        toxicity_scores.append(0)
    if (idx + 1) % 1000 == 0:
        print(f"Processed {idx + 1} comments...")

df['detoxify_toxicity_score'] = toxicity_scores
df['detoxify_prediction'] = predictions
print(f"\nTotal comments processed: {len(df)}")
df['human_label']=(df['target'] >= 0.5).astype(int)

# Overall Performance Metrics
print("\nOverall performance of Detoxify Original Model")
f1_overall = f1_score(df['human_label'], df['detoxify_prediction'])
precision_overall = precision_score(df['human_label'], df['detoxify_prediction'])
recall_overall = recall_score(df['human_label'], df['detoxify_prediction'])

print(f"""
F1 Score:  {f1_overall:.4f}
Precision: {precision_overall:.4f}
Recall:    {recall_overall:.4f}
Classification Report:{classification_report(df['human_label'], df['detoxify_prediction'], target_names=['Non-Toxic', 'Toxic'])}""")

# Analyzing Identity-Related Comments
print("="*60)
print("Identity related comments Analysis")
identity_columns = ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 
                    'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian', 
                    'intellectual_or_learning_disability', 'jewish', 'latino', 'male', 
                    'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity', 
                    'other_religion', 'other_sexual_orientation', 'physical_disability', 
                    'psychiatric_or_mental_illness', 'transgender', 'white']

bias_results = []
for identity in identity_columns:
    identity_mask = df[identity] > 0 #Getting comments that mention this identity
    if identity_mask.sum() > 0:  # Only if there are comments with this identity
        identity_subset = df[identity_mask]
        f1=f1_score(identity_subset['human_label'], identity_subset['detoxify_prediction'], zero_division=0)
        precision=precision_score(identity_subset['human_label'], identity_subset['detoxify_prediction'], zero_division=0)
        recall=recall_score(identity_subset['human_label'], identity_subset['detoxify_prediction'], zero_division=0)
        
        # False positive rate on non-toxic comments for this identity
        non_toxic_identity = identity_subset[identity_subset['human_label'] == 0]
        if len(non_toxic_identity) > 0:
            false_positive_rate = (non_toxic_identity['detoxify_prediction'] == 1).sum() / len(non_toxic_identity)
        else:
            false_positive_rate = 0
        bias_results.append({
            'Identity': identity,
            'Count': identity_mask.sum(),
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'False_Positive_Rate': false_positive_rate
        })
        print(f"""
        {identity.upper()}
        Comments: {identity_mask.sum()}
        F1 Score: {f1:.4f}
        Precision: {precision:.4f}
        Recall: {recall:.4f}
        False Positive Rate (on non-toxic): {false_positive_rate:.4f}""")

bias_df = pd.DataFrame(bias_results)

# identifying FALSE POSITIVES on Identity Comments
print("\nFALSE POSITIVES: Harmless Comments FLagged as Toxic")

#comments that are non-toxic (human) but flagged as toxic (model) AND mention identities
has_identity = df[identity_columns].sum(axis=1) > 0
false_positives = df[(df['human_label'] == 0) & (df['detoxify_prediction'] == 1) & has_identity]
print(f"\nTotal false positives on identity comments: {len(false_positives)}")
print(f"Out of {has_identity.sum()} total identity-related comments")
print(f"False positive rate on identity comments: {len(false_positives)/has_identity.sum() * 100:.2f}%")

print("\nTop 5 Examples of False Positives (harmless comments flagged as toxic):")
for idx, (_, row) in enumerate(false_positives.head(5).iterrows()):
    print(f"\n{idx+1}. Score: {row['detoxify_toxicity_score']:.4f} | Pred: Toxic | Human: Not Toxic")
    print(f"   Comment: {row['comment_text'][:120]}...")
    # showing which identities were mentioned
    mentioned = [id_col for id_col in identity_columns if row[id_col] > 0]
    if mentioned:
        print(f"   Identities mentioned: {', '.join(mentioned)}")
# saving results
bias_df.to_csv('detoxify_bias_by_identity.csv', index=False)
false_positives.to_csv('detoxify_false_positives_detailed.csv', index=False)