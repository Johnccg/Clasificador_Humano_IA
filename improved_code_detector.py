#!/usr/bin/env python3
"""
Improved C-CNN for detecting AI-generated vs human-written code.
Supports style features, character-level branch, focal loss, and attention.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Tree-sitter imports
from tree_sitter import Language, Parser
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_c_sharp as tree_sitter_csharp

# ------------------------------
# Configuration
# ------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MAX_WORDS = 10000
MAX_LEN = 500               # for syntax sequences
MAX_CHAR_LEN = 2000         # for raw character sequences
BATCH_SIZE = 64
EPOCHS = 10
EMBEDDING_DIM = 256
NUM_FILTERS = 128
FILTER_SIZES = [3, 4, 5]
DROPOUT_RATE = 0.4
L2_STRENGTH = 0.0005

# Language mapping
LANGUAGES = {
    'c': Language(tree_sitter_c.language()),
    'cpp': Language(tree_sitter_cpp.language()),
    'C#': Language(tree_sitter_csharp.language())
}
LANG_INV = {0: 'cpp', 1: 'C#', 2: 'c'}

# ------------------------------
# Feature extraction functions
# ------------------------------
def get_syntactic_sequence(code: str, lang: str) -> str:
    """Parse code and return a string of node types (preorder)."""
    if not code or not isinstance(code, str) or not code.strip():
        return ""
    try:
        parser = Parser(LANGUAGES[lang])
        tree = parser.parse(bytes(code, "utf8"))
        seq = []
        def walk(node):
            seq.append(node.type)
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return " ".join(seq)
    except Exception:
        return ""

def extract_style_features(code: str) -> np.ndarray:
    """Extract 10 statistical style features."""
    if not isinstance(code, str) or len(code) == 0:
        return np.zeros(10, dtype=np.float32)
    lines = code.split('\n')
    num_lines = len(lines)
    num_chars = len(code)
    if num_lines == 0:
        num_lines = 1
    features = []
    # 1. code length
    features.append(num_chars)
    # 2. number of lines
    features.append(num_lines)
    # 3. average line length
    features.append(num_chars / max(1, num_lines))
    # 4. brace balance (increase of { vs })
    open_braces = code.count('{')
    close_braces = code.count('}')
    features.append((open_braces - close_braces) / (max(1, open_braces + close_braces)) + 0.5)
    # 5. semicolon density (per line)
    features.append(code.count(';') / max(1, num_lines))
    # 6. space density
    features.append(code.count(' ') / max(1, num_chars))
    # 7. newline density
    features.append(code.count('\n') / max(1, num_chars))
    # 8. character entropy
    unique_ratio = len(set(code)) / max(1, num_chars)
    features.append(unique_ratio)
    # 9. digit density
    digits = sum(c.isdigit() for c in code)
    features.append(digits / max(1, num_chars))
    # 10. uppercase ratio
    upper = sum(c.isupper() for c in code)
    features.append(upper / max(1, num_chars))
    return np.array(features, dtype=np.float32)

def char_sequence(code: str, max_len: int) -> str:
    """Return raw code truncated to max_len chars for character-level CNN."""
    if not isinstance(code, str):
        return ""
    return code[:max_len]

# ------------------------------
# Data loading and preprocessing
# ------------------------------
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # Keep only needed columns (assume same structure as original)
    if 'task_url' in df.columns:
        df = df.drop(['task_url', 'task_name', 'task_description', 'set'], axis=1, errors='ignore')
    # Drop rows with missing code
    df = df.dropna(subset=['code'])
    # Encode categorical columns
    df['language_name'] = df['language_name'].astype('category').cat.codes
    df['target'] = df['target'].map({'Ai_generated': 0, 'Human_written': 1})
    # Map language codes
    df['lang'] = df['language_name'].map(LANG_INV)
    return df

def preprocess_data(df, tokenizer=None, fit_tokenizer=True, char_to_idx=None):
    # Generate syntactic sequences
    df['syn_seq'] = df.apply(lambda r: get_syntactic_sequence(r['code'], r['lang']), axis=1)
    # Generate style features
    style_features = np.stack(df['code'].apply(extract_style_features).values)
    # Generate raw character sequences (optional)
    df['char_seq'] = df['code'].apply(lambda x: char_sequence(x, MAX_CHAR_LEN))

    # Tokenize syntactic sequences
    if fit_tokenizer:
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
        tokenizer.fit_on_texts(df['syn_seq'].astype(str).tolist())
    seqs = tokenizer.texts_to_sequences(df['syn_seq'].astype(str).tolist())
    X_syn = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')

    # Character level: build or reuse char_to_idx
    if fit_tokenizer:
        # Build character vocabulary from training data
        char_vocab = set()
        for s in df['char_seq']:
            char_vocab.update(s)
        char_to_idx = {ch: i+1 for i, ch in enumerate(sorted(char_vocab))}  # 0 for padding
    # Convert each string to list of indices
    char_sequences = []
    for text in df['char_seq']:
        ids = [char_to_idx.get(ch, 0) for ch in text[:MAX_CHAR_LEN]]
        char_sequences.append(ids)
    X_char = pad_sequences(char_sequences, maxlen=MAX_CHAR_LEN, padding='post', truncating='post', value=0)

    y = df['target'].values
    return X_syn, X_char, style_features, y, tokenizer, char_to_idx

# ------------------------------
# Model building
# ------------------------------
def build_model(vocab_size, char_vocab_size, num_style_features,
                use_char=True, use_attention=False, use_focal=False):
    # Inputs
    syn_input = layers.Input(shape=(MAX_LEN,), name='syn_input')
    style_input = layers.Input(shape=(num_style_features,), name='style_input')
    inputs = [syn_input, style_input]
    
    # Syntactic branch (embedding + convs)
    emb = layers.Embedding(vocab_size, EMBEDDING_DIM, mask_zero=False)(syn_input)
    conv_outputs = []
    for fsz in FILTER_SIZES:
        conv = layers.Conv1D(NUM_FILTERS, fsz, padding='valid', use_bias=False)(emb)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        if use_attention:
            # Simple attention: average over time with learned weights
            score = layers.Dense(1, activation='tanh')(conv)
            score = layers.Flatten()(score)
            weight = layers.Activation('softmax')(score)
            weighted = layers.Dot(axes=1)([conv, weight])
            conv_outputs.append(weighted)
        else:
            pool = layers.GlobalMaxPooling1D()(conv)
            conv_outputs.append(pool)
    if not use_attention:
        concat = layers.Concatenate()(conv_outputs)
    else:
        concat = layers.Concatenate()(conv_outputs)  # attention outputs are already vectors
    syn_features = layers.Dropout(DROPOUT_RATE)(concat)
    
    # Character branch (optional)
    if use_char:
        char_input = layers.Input(shape=(MAX_CHAR_LEN,), name='char_input')
        inputs.append(char_input)
        char_emb = layers.Embedding(char_vocab_size, 64, mask_zero=False)(char_input)
        char_conv = layers.Conv1D(64, 5, activation='relu', padding='same')(char_emb)
        char_pool = layers.GlobalMaxPooling1D()(char_conv)
        char_drop = layers.Dropout(0.3)(char_pool)
        # Merge all
        merged = layers.Concatenate()([syn_features, style_input, char_drop])
    else:
        merged = layers.Concatenate()([syn_features, style_input])
    
    # Dense layers
    dense = layers.Dense(64, activation='relu',
                         kernel_regularizer=regularizers.l2(L2_STRENGTH))(merged)
    dense = layers.BatchNormalization()(dense)
    dense = layers.Dropout(0.3)(dense)
    output = layers.Dense(2, activation='softmax', name='output')(dense)
    
    model = models.Model(inputs=inputs, outputs=output)
    # Loss
    if use_focal:
        def focal_loss(gamma=2., alpha=0.25):
            def focal_loss_fixed(y_true, y_pred):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
                ce = -y_true * tf.math.log(y_pred)
                weight = y_true * tf.pow(1 - y_pred, gamma) * alpha + \
                         (1 - y_true) * tf.pow(y_pred, gamma) * (1 - alpha)
                return tf.reduce_mean(weight * ce)
            return focal_loss_fixed
        loss = focal_loss()
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# ------------------------------
# Training and evaluation
# ------------------------------
def train_model(model, X_syn, X_char, style_features, y_train,
                X_syn_val, X_char_val, style_val, y_val,
                use_char, checkpoint_dir='./checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'improved_cnn_best.h5')
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    # Prepare inputs
    train_inputs = [X_syn, style_features]
    val_inputs = [X_syn_val, style_val]
    if use_char:
        train_inputs.append(X_char)
        val_inputs.append(X_char_val)
    
    history = model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    # Load best weights
    model.load_weights(checkpoint_path)
    return history

def evaluate(model, X_syn, X_char, style_features, y_test, use_char):
    test_inputs = [X_syn, style_features]
    if use_char:
        test_inputs.append(X_char)
    loss, acc = model.evaluate(test_inputs, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    y_pred = np.argmax(model.predict(test_inputs), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ai_generated', 'Human_written']))
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ai_generated', 'Human_written'],
                yticklabels=['Ai_generated', 'Human_written'])
    plt.title('Confusion Matrix - Improved C-CNN')
    plt.xlabel('Prediction')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_improved.png', dpi=150)
    plt.show()

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='./H-AIRosettaMP_C_Cpp_Csharp.csv',
                        help='Path to input CSV')
    parser.add_argument('--use_char', action='store_true', help='Use character-level branch')
    parser.add_argument('--use_attention', action='store_true', help='Use attention in conv branch')
    parser.add_argument('--use_focal', action='store_true', help='Use focal loss')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    
    # Load and split data
    df = load_data(args.csv)
    # Split train/test (75/25)
    df_train = df.sample(frac=0.75, random_state=SEED)
    df_test = df.drop(df_train.index)
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    # Preprocess training (fit tokenizer)
    X_syn_train, X_char_train, style_train, y_train, tokenizer, char_to_idx = preprocess_data(df_train, fit_tokenizer=True)
    # Preprocess test (use same tokenizer and char_to_idx)
    X_syn_test, X_char_test, style_test, y_test, _, _ = preprocess_data(df_test, tokenizer=tokenizer, fit_tokenizer=False, char_to_idx=char_to_idx)
    
    # Validation split from training (20% of train)
    X_syn_train, X_syn_val, style_train, style_val, y_train, y_val, X_char_train, X_char_val = train_test_split(
        X_syn_train, style_train, y_train, X_char_train, test_size=0.2, random_state=SEED, stratify=y_train
    )
    
    # Build model
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    char_vocab_size = len(char_to_idx) + 1 if args.use_char else 0
    num_style = style_train.shape[1]
    model = build_model(vocab_size, char_vocab_size, num_style,
                        use_char=args.use_char,
                        use_attention=args.use_attention,
                        use_focal=args.use_focal)
    model.summary()
    
    # Train
    history = train_model(model, X_syn_train, X_char_train, style_train, y_train,
                          X_syn_val, X_char_val, style_val, y_val,
                          use_char=args.use_char)
    
    # Evaluate on test
    evaluate(model, X_syn_test, X_char_test, style_test, y_test, use_char=args.use_char)

if __name__ == '__main__':
    main()