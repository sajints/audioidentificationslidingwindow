import sqlite3
import json

def create_database(db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_name TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def store_embedding(audio_name, embedding, db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Convert numpy array to list
    embedding_list = embedding.tolist() 
    embedding_json = json.dumps(embedding_list) # Store embedding as JSON string
    cursor.execute('''
        INSERT INTO embeddings (audio_name, embedding)
        VALUES (?, ?)
    ''', (audio_name, embedding_json))
    conn.commit()
    conn.close()

def load_embeddings(id, db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT embedding FROM embeddings WHERE id = ?', (id,))
    rows = cursor.fetchall()
    conn.close()

    if len(rows) == 0:
        return None

    # Assuming one embedding per id, so take the first row, first column
    embedding_json_str = rows[0][0]
    embedding = json.loads(embedding_json_str)
    return embedding