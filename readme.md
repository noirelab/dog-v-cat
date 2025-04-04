# Dog/Cat Classifier
Este é projeto final do LAMIA, sendo uma aplicação de classificação de imagens para distinguir gatos e cachorros utilizando técnicas de transferência de aprendizado.

## 1. Organização dos Dados
Foi usado o dataset ["Cat and Dog"](https://www.kaggle.com/datasets/tongpython/cat-and-dog), vindo do site Kaggle.
Os dados são organizados em pastas, onde cada subpasta representa uma classe (por exemplo, "cats" e "dogs"). Tanto o conjunto de treinamento quanto o de teste seguem essa estrutura, permitindo que os rótulos sejam inferidos automaticamente.

![Dataset](/images/dataset.png)

## 2. Pré-processamento e Carregamento
As imagens são carregadas e pré-processadas para garantir que todas tenham o mesmo tamanho (256x256 pixels) e sejam normalizadas (valores dos pixels ajustados para a faixa [0,1]). Esse pré-processamento é realizado utilizando bibliotecas como OpenCV e funções do TensorFlow, garantindo a consistência dos dados de entrada para o modelo.

## 3. Modelo de Transfer Learning
O modelo utiliza o MobileNetV2 pré-treinado com pesos do ImageNet para extração de características visuais. As camadas do MobileNetV2 são congeladas para aproveitar o conhecimento prévio. Em seguida, uma nova cabeça de classificação é adicionada, composta por camadas de GlobalAveragePooling2D, Dense, e DropOut para reduzir o overfitting. A saída final é composta por uma única unidade com ativação sigmoid, adequada para problemas de classificação binária.

## 4. Treinamento do Modelo
O treinamento é realizado utilizando técnicas como Early Stopping e Checkpoints. O modelo é avaliado durante o treinamento com base no desempenho do conjunto de validação, interrompendo o treinamento caso não haja melhorias na perda de validação por um número definido de épocas. Isso assegura um treinamento eficiente e evita overfitting.

## 5. Avaliação e Métricas
Após o treinamento, o desempenho do modelo é avaliado com métricas como acurácia, perda, recall e uma matriz de confusão. Essas métricas permitem identificar a eficácia do modelo na distinção entre as classes. Gráficos ilustram a evolução da acurácia e da perda durante o treinamento, fornecendo uma visão clara do comportamento do modelo.
O modelo chegou nos seguintes resultados:
- Acurácia de Treinamento: 99.56%
- Acurácia de validação: 98.71%
- Recall: 99.30%

![Matriz Confusão](/images/confusion_matrix.png)
