from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd
df = pd.read_csv("ss.csv").fillna('')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
X = tokenizer(df['comment_text'].tolist(), padding=True, truncation=True, max_length=10, return_tensors='tf')
y = df.iloc[:, 2:].values
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X['input_ids'], y, epochs=1, batch_size=8)
input_text = input("Enter a comment: ")
pred = model.predict(tokenizer([input_text], padding=True, truncation=True, max_length=100, return_tensors='tf')['input_ids'])[0]
print("Predictions:", pred)