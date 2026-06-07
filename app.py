from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# --- Dados ---

dados_filmes = {
    'titulo': [
        'Superbad', 'Duro de Matar', 'Invocação', '17 outra vez',
        'Gente Grande', 'John Wick', 'O Chamado', 'Ela é demais pra mim',
        'Anjos da Lei', 'Mad Max'
    ],
    'duracao': [113, 132, 112, 116, 102, 101, 115, 123, 109, 120],
    'piadas':  [  1,   0,   0,   1,   1,   0,   0,   1,   1,   0],
    'acao':    [  0,   1,   0,   0,   0,   1,   0,   0,   1,   1],
    'terror':  [  0,   0,   1,   0,   0,   0,   1,   0,   0,   0],
    'romance': [  0,   0,   0,   1,   0,   0,   0,   1,   0,   0],
    'genero': [
        'comedia', 'acao', 'terror', 'romance',
        'comedia', 'acao', 'terror', 'romance',
        'comedia', 'acao'
    ]
}

df = pd.DataFrame(dados_filmes)

features_genero  = ['piadas', 'acao', 'terror', 'romance']   # features de gênero
features_duracao = ['duracao']                                # feature de duração
features_todas   = ['duracao', 'piadas', 'acao', 'terror', 'romance']

scaler = MinMaxScaler()
scaler.fit(df[features_todas])


def knn_ponderado(df_base, novo_scaled, peso_genero: float, n_vizinhos: int = 3):
    """
    KNN manual com pesos separados para gênero e duração.
    peso_genero: 0.0 = só duração, 1.0 = só gênero, 0.5 = equilibrado.
    peso_duracao = 1 - peso_genero.
    """
    peso_duracao = 1.0 - peso_genero

    X_base = scaler.transform(df_base[features_todas])

    # Índices das colunas no array escalado
    # features_todas = ['duracao', 'piadas', 'acao', 'terror', 'romance']
    idx_duracao = [0]
    idx_genero  = [1, 2, 3, 4]

    # Distância euclidiana ponderada
    diff = X_base - novo_scaled                          # (n, 5)
    dist_dur = np.sqrt(np.sum(diff[:, idx_duracao] ** 2, axis=1))
    dist_gen = np.sqrt(np.sum(diff[:, idx_genero]  ** 2, axis=1))

    distancias = peso_duracao * dist_dur + peso_genero * dist_gen

    n = min(n_vizinhos, len(df_base))
    indices = np.argsort(distancias)[:n]
    return indices, distancias[indices]


# --- Rotas ---

@app.route('/recomendar', methods=['POST'])
def recomendar():
    body = request.get_json()
    duracao     = body.get('duracao', 120)
    piadas      = body.get('piadas', 0)
    acao        = body.get('acao', 0)
    terror      = body.get('terror', 0)
    romance     = body.get('romance', 0)
    # 0 = só duração importa | 1 = só gênero importa | 0.5 = equilibrado
    peso_genero = float(body.get('peso_genero', 0.5))

    novo = pd.DataFrame([[duracao, piadas, acao, terror, romance]], columns=features_todas)
    novo_scaled = scaler.transform(novo)[0]

    # 1. Encontrar gênero dominante pelos k vizinhos mais próximos
    indices, _ = knn_ponderado(df, novo_scaled, peso_genero, n_vizinhos=3)
    generos_vizinhos = df['genero'].iloc[indices].tolist()
    genero = max(set(generos_vizinhos), key=generos_vizinhos.count)  # moda

    # 2. Dentro do gênero escolhido, buscar o filme mais próximo
    df_filtrado = df[df['genero'] == genero].reset_index(drop=True)
    indices_f, distancias_f = knn_ponderado(df_filtrado, novo_scaled, peso_genero, n_vizinhos=1)
    filme = df_filtrado['titulo'].iloc[indices_f[0]]

    return jsonify({
        'genero': genero,
        'filme': filme,
        'peso_genero': peso_genero,
        'vizinhos': generos_vizinhos
    })


@app.route('/filmes', methods=['GET'])
def listar_filmes():
    return jsonify(df.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
