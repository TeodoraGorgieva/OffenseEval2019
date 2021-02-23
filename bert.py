import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Model
from transformers import TFBertModel, BertTokenizer


def tokenize(data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    input_ids, att_mask = [], []
    for tweet in data:
        tokenized = tokenizer.encode_plus(tweet, add_special_tokens=True, max_length=50, padding='max_length',
                                          truncation=True)
        input_ids.append(tokenized['input_ids'])
        att_mask.append(tokenized['attention_mask'])
    return np.array(input_ids), np.array(att_mask)


def create_model(text, labels):
    x_train, x_test_tmp, y_train, y_test_tmp = train_test_split(text, labels, test_size=0.2, shuffle=True,
                                                                random_state=200)
    x_val, x_test, y_val, y_test = train_test_split(x_test_tmp, y_test_tmp, test_size=0.5, shuffle=True,
                                                    random_state=200)

    num_labels = len(set(labels))
    model = TFBertModel.from_pretrained('bert-base-cased', num_labels=num_labels)

    train_input_ids, train_att_mask = tokenize(x_train)
    test_input_ids, test_att_mask = tokenize(x_test)
    val_input_ids, val_att_mask = tokenize(x_val)

    input_ids = Input(shape=(50,), name='input_token', dtype='int32')
    att_masks = Input(shape=(50,), name='masked_token', dtype='int32')
    bert_in = model(input_ids, attention_mask=att_masks)
    bert_out = Dense(64, activation='relu')(bert_in[1])
    bert_out = Dense(16, activation='relu')(bert_out)
    bert_out = Flatten()(bert_out)
    bert_out = Dense(1, activation='softmax')(bert_out)
    model = Model(inputs=[input_ids, att_masks], outputs=bert_out)

    model.summary()
    if num_labels == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

    model.fit([train_input_ids, train_att_mask], np.array(y_train), batch_size=32, epochs=3, verbose=1,
              validation_data=([val_input_ids, val_att_mask], np.array(y_val)))

    predicted_classes = model.predict([test_input_ids, test_att_mask], verbose=1)
    print(f'Accuracy with Bert Model: {accuracy_score(y_test, predicted_classes)}\n'
          f'Precision score with Bert Model: {precision_score(y_test, predicted_classes, average="micro")}\n'
          f'Recall score with Bert Model: {recall_score(y_test, predicted_classes, average="micro")}\n'
          f'F1 score with Bert Model: {f1_score(y_test, predicted_classes, average="micro")}')


def create_samples(file, task):
    data = pd.read_csv(file, ',')

    positive = data[data[f'subtask_{task.lower()}'] == 1]
    negative = data[data[f'subtask_{task.lower()}'] == 0]
    neutral = data[data[f'subtask_{task.lower()}'] == 2]

    if task == 'C':
        length = min(len(positive), len(negative), len(neutral))
    else:
        length = min(len(positive), len(negative))

    data = positive.sample(length, random_state=200)
    data = pd.concat([data, negative.sample(length, random_state=200)], ignore_index=True)

    if task == 'C':
        data = pd.concat([data, neutral.sample(length, random_state=200)], ignore_index=True)

    text = data['tweet'].values.astype('U')
    labels = list(data[f'subtask_{task.lower()}'])
    return text, labels


if __name__ == '__main__':
    file = f'data/trainA.csv'
    tweet, labels = create_samples(file, 'A')
    create_model(tweet, labels)

    file = f'data/trainB.csv'
    tweet, labels = create_samples(file, 'B')
    create_model(tweet, labels)

    file = f'data/trainC.csv'
    tweet, labels = create_samples(file, 'C')
    create_model(tweet, labels)
