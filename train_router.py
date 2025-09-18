import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

print("Starting model training process...")

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('intent_dataset.csv')
queries = df['query'].tolist()
intents = df['intent'].tolist()

# 2. Encode text labels into numbers
print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(intents)

# The encoder learns the mapping, e.g., 'casual' -> 0, 'doc_qna' -> 1
# We can see the classes it has learned
print(f"Encoded classes: {label_encoder.classes_}")

# 3. Generate sentence embeddings
print("Generating sentence embeddings... (This may take a moment)")
# We use a popular, lightweight model perfect for this task
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
X_embeddings = embedding_model.encode(queries, show_progress_bar=True)

print(f"Embeddings generated with shape: {X_embeddings.shape}")

# 4. Train the Logistic Regression model
print("Training the classification model...")
classifier = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
classifier.fit(X_embeddings, y_encoded)

print("Model training complete.")

# 5. Save the trained model and the label encoder
print("Saving model and label encoder...")
joblib.dump(classifier, 'intent_classifier.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

print("✅ Training process finished successfully. Artifacts saved.")
print("   - intent_classifier.joblib (The trained model)")
print("   - label_encoder.joblib (The label mapping)")