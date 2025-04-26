# dog-v-cat

Este projeto utiliza aprendizado de máquina para classificar imagens de cães e gatos. O modelo foi treinado utilizando a arquitetura **MobileNetV2** e implementado com a biblioteca **TensorFlow**.
O dataset inicial ficou muito pequeno, por isso, tive que usar um código de WebScraping para aumentar em praticamente 416% o dataset de raças de cachorro.

## Como iniciar
```bash
cd gui
python gui.py
```

## Funcionalidades Principais

1. **Pré-processamento de Dados**:
   - Redimensionamento de imagens para 300x300 ou 224x224 pixels, depende do modelo em questão.
   - Normalização dos valores dos pixels.

2. **Divisão do Dataset**:
   - Separação dos dados em conjuntos de treinamento e teste.

3. **Modelo de Classificação**:
   - Utilização do modelo **MobileNetV2** como base.
   - Adição de camadas densas para ajuste fino e classificação binária.

4. **Treinamento do Modelo**:
   - Uso de callbacks como **EarlyStopping**, **ReduceLROnPlateau** e **ModelCheckpoint** para otimizar o treinamento.

5. **Avaliação e Visualização**:
   - Geração de gráficos de acurácia e perda.
   - Exibição de matriz de confusão e cálculo de métricas como **Recall**.

6. **Predição em Imagens Reais**:
   - Classificação de novas imagens utilizando o modelo treinado.

## Estrutura do Código

- **Importação de Bibliotecas**: Importa bibliotecas essenciais como TensorFlow, OpenCV, Matplotlib, entre outras.
- **Carregamento de Dados**: Lê imagens do dataset e mapeia rótulos para valores inteiros.
- **Treinamento**: Realiza o treinamento do modelo com o conjunto de dados de treinamento.
- **Avaliação**: Avalia o modelo com o conjunto de teste e exibe métricas de desempenho.
- **Teste Real**: Classifica imagens externas fornecidas pelo usuário.

## Resultados do modelo diferenciador dog-v-cat

- **Acurácia de Treinamento**: 99.58%
- **Acurácia de Validação**: 98.96%
- **Recall**: 99.2%

## Observações
- Certifique-se de que o dataset está organizado nos diretórios corretos antes de executar o código.
- O modelo treinado será salvo no arquivo `dog_or_cat.keras`.
- Resumo dos notebooks:
    - **main.ipynb**:
        - **Objetivo**: Classificar imagens como "cão" ou "gato".
        - **Pipeline**:
        1. **Pré-processamento**:
            - Redimensionamento de imagens para 300x300 pixels.
            - Normalização dos valores dos pixels.
        2. **Modelo**:
            - Baseado na arquitetura **MobileNetV2**.
            - Adição de camadas densas para classificação binária.
        3. **Treinamento**:
            - Utiliza callbacks como **EarlyStopping**, **ReduceLROnPlateau**, e **ModelCheckpoint**.
            - Salva o modelo no arquivo `dog_or_cat.keras`.
        4. **Avaliação**:
            - Exibe métricas como acurácia, matriz de confusão e **Recall**.
        5. **Predição**:
            - Classifica novas imagens como "cão" ou "gato".
            - Dependendo do resultado, utiliza os modelos de raças de cães ou gatos (`dogbreed` ou `catbreed`).
            - ![Matriz de confusão](images/imagens%20para%20o%20relatório/dog_cat_conf_matrix.png)

    - **dogbreed_v3_reduced_enhanced.ipynb**:
        - **Objetivo**: Classificar imagens de 24 raças de cães.
        - **Pipeline**:
        1. **Pré-processamento**:
            - Criação de um dataframe com imagens e classes.
            - Divisão dos dados em treino (80%) e teste (20%).
        2. **Modelo**:
            - Baseado na arquitetura **EfficientNetV2B3**.
            - Adição de camadas densas para classificação.
            - Configuração do modelo com otimizador Adam e função de perda `categorical_crossentropy`.
        3. **Treinamento**:
            - Utiliza callbacks como **EarlyStopping**, **ReduceLROnPlateau**, e **ModelCheckpoint**.
            - Salva o modelo no arquivo `model_24_webscraped_classes.h5`.
        4. **Resultados**:
            - Exibe métricas como acurácia e perda durante o treinamento.
            - Avalia o modelo no conjunto de teste.
            - ![Matriz de confusão](images/imagens%20para%20o%20relatório/dog_conf_matrix.png)

    - **catbreed_v5.ipynb**:
        - **Objetivo**: Classificar imagens de 24 raças de gatos.
        - **Pipeline**:
            1. **Pré-processamento**:
            - Criação de um dataframe com imagens e classes.
            - Divisão dos dados em treino (80%) e teste (20%).
            - Codificação one-hot para as classes.
            2. **Modelo**:
            - Baseado na arquitetura **EfficientNetV2B3**.
            - Adição de camadas densas para classificação.
            - Configuração do modelo com otimizador Adam e função de perda `categorical_crossentropy`.
            3. **Treinamento**:
            - Utiliza callbacks como **EarlyStopping**, **ReduceLROnPlateau**, e **ModelCheckpoint**.
            - Salva o modelo no arquivo `catbreed_model_v5.h5`.
            4. **Resultados**:
            - Exibe métricas como acurácia e perda durante o treinamento.
            - Avalia o modelo no conjunto de teste.
            - ![Matriz de confusão](images/imagens%20para%20o%20relatório/cat_conf_matrix.png)

## Exemplos de Uso dentro do main.ipynb

```python
import model_usage as mu

# Classificar uma imagem
mu.dog_cat_breed_classifier('image.jpg')
```

Este projeto demonstra como utilizar redes neurais convolucionais para resolver problemas de classificação de imagens de forma eficiente.
