
from flask import Flask, render_template_string, request
import sqlite3
import json

app = Flask(__name__)

@app.route('/')
def index():
    conn = sqlite3.connect('myeloma_reports_de.sqlite')

    # Get table info
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    html = '''
    <h1>Simple Database Browser</h1>
    <h2>Tables:</h2>
    <ul>
    {% for table in tables %}
        <li><a href="/table/{{ table[0] }}">{{ table[0] }}</a></li>
    {% endfor %}
    </ul>
    '''

    conn.close()
    return render_template_string(html, tables=tables)

@app.route('/table/<table_name>')
def show_table(table_name):
    conn = sqlite3.connect('myeloma_reports_de.sqlite')

    # Get column info
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()

    # Get sample data (first 10 rows)
    rows = conn.execute(f"SELECT * FROM {table_name} LIMIT 10").fetchall()

    html = '''
    <h1>Table: {{ table_name }}</h1>
    <a href="/">Back to tables</a>
    <h2>Columns:</h2>
    <ul>
    {% for col in columns %}
        <li>{{ col[1] }} ({{ col[2] }})</li>
    {% endfor %}
    </ul>
    <h2>Sample Data:</h2>
    <table border="1">
    <tr>
    {% for col in columns %}
        <th>{{ col[1] }}</th>
    {% endfor %}
    </tr>
    {% for row in rows %}
    <tr>
    {% for cell in row %}
        <td>{{ cell if cell is not none else 'NULL' }}</td>
    {% endfor %}
    </tr>
    {% endfor %}
    </table>
    '''

    conn.close()
    return render_template_string(html, table_name=table_name, columns=columns, rows=rows)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8522, debug=True)
