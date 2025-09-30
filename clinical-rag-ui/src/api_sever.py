from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import json
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for React app

class ClinicalRAGDB:
    def __init__(self, db_path: str = "clinical_rag.db"):
        self.db_path = db_path
    
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    
    def search_patients(self, query: str, limit: int = 10):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, dob 
            FROM patients 
            WHERE name LIKE ? 
            ORDER BY name 
            LIMIT ?
        ''', (f'%{query}%', limit))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def search_documents(self, query: str, patient_id=None, k: int = 8, doc_types=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Escape the FTS query
        escaped_query = escape_fts_query(query)
        print(f"🔍 Original query: '{query}' -> Escaped: '{escaped_query}'")
        
        sql = '''
            SELECT d.id, d.patient_id, d.encounter_id, d.doc_type, d.section, 
                   d.date, d.title, d.content, d.author, d.department,
                   snippet(documents_fts, 2, '<mark>', '</mark>', '...', 100) as snippet,
                   bm25(documents_fts) as score
            FROM documents_fts fts
            JOIN documents d ON d.rowid = fts.rowid
            WHERE documents_fts MATCH ?
        '''
        
        params = [escaped_query]
        
        if patient_id:
            sql += ' AND d.patient_id = ?'
            params.append(patient_id)
        
        if doc_types:
            placeholders = ','.join(['?' for _ in doc_types])
            sql += f' AND d.doc_type IN ({placeholders})'
            params.extend(doc_types)
        
        sql += ' ORDER BY bm25(documents_fts) DESC LIMIT ?'
        params.append(k)
        
        print(f"🗄️ Executing SQL: {sql}")
        print(f"📋 With params: {params}")
        
        try:
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                # Clean up snippet
                if result['snippet']:
                    result['snippet'] = result['snippet'].replace('<mark>', '').replace('</mark>', '')
                else:
                    result['snippet'] = result['content'][:200] + '...'
                
                # Make score positive
                result['score'] = abs(float(result['score'])) if result['score'] else 0.0
                
                results.append(result)
            
            print(f"✅ Found {len(results)} documents")
            conn.close()
            return results
            
        except Exception as e:
            print(f"❌ SQL Error: {e}")
            conn.close()
            return []

# Initialize database connection
db = ClinicalRAGDB()

def escape_fts_query(query):
    """Escape special characters in FTS queries"""
    # Remove or escape problematic characters
    query = re.sub(r'[\'\"*(){}[\]]', ' ', query)  # Remove quotes and special chars
    query = re.sub(r'\s+', ' ', query)  # Normalize whitespace
    query = query.strip()
    
    # Split into words and quote each one for exact matching
    words = query.split()
    if len(words) > 1:
        # For multi-word queries, use OR between words
        escaped = ' OR '.join(f'"{word}"' for word in words if len(word) > 2)
    else:
        escaped = f'"{words[0]}"' if words and len(words[0]) > 2 else query
    
    return escaped if escaped else query

@app.route('/api/patients/search', methods=['POST'])
def search_patients():
    data = request.json
    query = data.get('query', '')
    
    try:
        results = db.search_patients(query)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/search', methods=['POST'])
def search_documents():
    data = request.json
    query = data.get('query', '')
    patient_id = data.get('patient_id')
    k = data.get('k', 8)
    doc_types = data.get('doc_types', [])
    
    try:
        results = db.search_documents(query, patient_id, k, doc_types if doc_types else None)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Clinical RAG API server...")
    print("Available endpoints:")
    print("  POST /api/patients/search - Search patients")
    print("  POST /api/documents/search - Search documents")
    print("  GET /api/health - Health check")
    
    app.run(debug=True, host='localhost', port=5200)