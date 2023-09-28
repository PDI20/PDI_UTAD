## Introdução

Uma empresa de tecnologia procura o desenvolvimento de um algoritmo que faça a deteção de matrículas de veículos e extraia os seus caracteres, especificamente, de matrículas portuguesas, para gestão de entradas e saídas das instalações da empresa.
O algoritmo deve ser constituído por quatro módulos:
-	**Deteção da matrícula**: módulo responsável pela deteção das matrículas. Neste módulo serão recolhidas imagens de veículos com matrícula portuguesa, com as quais será criado um dataset utilizado para treinar um modelo de deep learning. Depois do treino, os pesos (um ficheiro do tipo .pt, por exemplo, “best.pt”), gerados durante o treino, serão utilizados para efetuar inferências sobre imagens nunca antes vistas. Por fim, guardar as imagens com as anotações feitas durante as inferências e os respetivos ficheiros (do tipo .txt) que contêm as coordenadas das bounding boxes, de forma a ser utilizado no próximo módulo. 
-	**Recorte da imagem com base nas coordenadas da bounding box**: módulo responsável por pelo recorte da área onde foi detetada a matrícula. Com os valores presentes em cada ficheiro, calcular as coordenadas do canto superior esquerdo e o canto inferior direito da bouding box. Por fim, utilizar uma biblioteca que permita processar digitalmente imagens (OpenCV) para efetuar o recorte da área da bounding box e guardar as novas imagens que apenas contêm a matrícula.
-	 **Pipeline de processamento digital de imagem**: módulo responsável por extrair os caracteres presentes nas imagens recortadas, obtidas no módulo anterior. De forma a atingir este objetivo existem várias abordagens que podem ser utilizadas:
o	OCR (Optical Character Recognition) - processo que converte imagens com texto em texto capaz de ser compreendido por computadores.
o	Pipeline de processamento digital de imagem - segmentação e extração dos caracteres utilizando técnicas de processamento de imagem.
-	**Análise de texto e correção de erros**: módulo responsável pela análise do texto extraído, de todas as imagens, e correção de potenciais erros cometidos pela abordagem utilizado no módulo anterior.



## Conteúdo

- [Recursos](#software-a-utilizar)
- [Módulo 1 - Deteção da matrícula](#módulo-1---deteção-da-matrícula)
  - [Construção do dataset](#construção-do-dataset)
  - [Preparação do dataset (Roboflow)](#preparar-dataset)
    - [Criar projeto de deteção de objetos](#criar-projeto-roboflow)
    - [Efetuar upload das imagens recolhidas](#upload-imagens)
    - [Anotar as imagens](#anotar-imagens)
    - [Aplicar aumentações (opcional)](#aumentar-imagens)
    - [Efetuar download do dataset no formato desejado](#download-dataset)
  - [Treino do dataset](#treino-dataset)
    - [Resultados do treino](#)
  - [Inferir sobre novas imagens](#inferências)
- [Módulo 2 - Recorte da imagem com base nas coordenadas da bounding boxes](#módulo-2)
  - [Organização do ficheiro com as bounding boxes](#ficheiro-bb)
  - [Obter os caminhos dos ficheiros](#caminhos-bb)
  - [Abrir os ficheiros e ler os seus conteúdos](#ler-ficheiro)
    - [Reverter a normalização das coordenadas geradas](#desnormalização)
  - [Calcular as coordenadas dos ponto superior esquerdo e do ponto inferior direito](#calculo-pontos)
  - [Efetuar o recorte da imagem com base nas coordenadas calculadas](#recorte-imagens)
- [Módulo 3 - Pipeline de processamento digital da imagem](#módulo-3)
  - [Pré-processamento das imagens recortadas](#processar-imagens)
  - [Abordagem 1 - Utilização de uma biblioteca OCR (Optical Character recognition)](#ocr)
    - [Instalar a biblioteca PaddleOCR](#instalar-ocr)
    - [Carregar o modelo responsável pelo reconhecimento de texto](#modelo-ocr)
    - [Obter os caminhos das imagens recortadas](#imagens-ocr)
    - [Aplicar o OCR sobre as imagens](#aplicar-ocr)
    - [Guardar os resultados](#guardar-ocr)
  - [Abordagem 2 - Aplicação do método de Otsu](#otsu)
    - [Obter o caminho das imagens recortadas](#caminhos-otsu)
    - [Aplicar o algoritmo de Otsu](#aplicar-otsu)
    - [Verificar o número de píxeis pretos](#pixeis-otsu)
    - [Calcular contours da imagem binarizada](#contours-otsu)
    - [Com base nos contours extrair os caracteres](#extrair-otsu)
    - [Classificar os caracteres extraídos](#classificar-otsu)
  - [Abordagem 3 - Utilização da biblioteca Grounding Dino (deteção de caracteres) e Segment Anything Model (segmentação de caracteres)](#gd-sam)
    - [Instalar bibliotecas Grounding Dino e Segmente Anythin Model (SAM)](#instalar-gd-sam)
    - [Obter os caminhos das imagens recortadas](#caminhos-gd-sam)
    - [Aplicar do Grounding Dino sobre as imagens](#aplicar-gd)
    - [Guardar as imagens geradas, com as bounding boxes](#imagens-gd)
    - [Aplicar o SAM sobre as imagens geradas pelo Grounding Dino](#aplicar-sam)
    - [Obter as máscaras geradas](#mascaras-sam)
    - [Inverter as cores das máscaras](#inverter-cor-sam)
    - [Calcular contours da imagem binarizada](#contours-gd-sam)
    - [Com base nos contours extrair os caracteres](#extrair-gd-sam)
    - [Classificar os caracteres extraídos](#classificar-gd-sam)
- [Módulo 4 - Análise de texto e correção de erros](#módulo-4)
  - [Formato das matrículas portuguesas](#formato-matriculas)
  - [Erros nos resultados obtidos pelo OCR e classificação de caracteres](#erros-resultados)
  - [Correção de erros](#correcao-erros)
  - [Guardar os resultados](#guardar-resultadoss)
  - [Comparar resultados com as matrículas](#comparar-resultados)


## Recursos

Para o desenvolvimento do protocolo serão utilizados os seguintes recursos:

- [Google Colab](https://colab.research.google.com/): desenvolvimento do código e treino dos modelos;
- [Google Drive](https://www.google.com/drive/): guardar os datasets e os resultados do treino;
- [Roboflow](https://roboflow.com/): preparar os dados para o dataset;
- [YOLO (You Only Look Once)](https://github.com/ultralytics): treinar um modelo utilizando o dataset desenvolvido e inferir sobre novas imagens;
- [Otsu](https://en.wikipedia.org/wiki/Otsu%27s_method): segmentar imagens (abordagem 2);
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/README_en.md): deteção automática de texto presente em imagens (abordagem 1);
- [Grounding Dino](https://github.com/IDEA-Research/GroundingDINO): deteção automática de objetos, com base num prompt (abordagem 3);
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything): segmentação automática de objetos (abordagem 3).


## Módulo 1 - Deteção da matrícula

## Construção do dataset

### Preparação do dataset (Roboflow)

### Criar projeto de deteção de objetos

### Efetuar upload das imagens recolhidas

### Anotar as imagens

### Aplicar aumentações (opcional)

### Efetuar download do dataset no formato desejado

## Treino do dataset

### Resultados do treino

## Inferir sobre novas imagens

## Módulo 2 - Recorte da imagem com base nas coordenadas da bounding boxes

## Organização do ficheiro com as bounding boxes

## Obter os caminhos dos ficheiros

## Abrir os ficheiros e ler os seus conteúdos

### Reverter a normalização das coordenadas geradas

## Calcular as coordenadas dos ponto superior esquerdo e do ponto inferior direito

## Efetuar o recorte da imagem com base nas coordenadas calculadas

## Módulo 3 - Pipeline de processamento digital da imagem

## Pré-processamento das imagens recortadas

## Abordagem 1 - Utilização de uma biblioteca OCR (Optical Character recognition)

### Instalar a biblioteca PaddleOCR

### Carregar o modelo responsável pelo reconhecimento de texto

### Obter os caminhos das imagens recortadas
  
### Aplicar o OCR sobre as imagens

### Guardar os resultados

## Abordagem 2 - Aplicação do método de Otsu

### Obter o caminho das imagens recortadas

### Aplicar o algoritmo de Otsu

### Verificar o número de píxeis pretos 

### Calcular contours da imagem binarizada

### Com base nos contours extrair os caracteres

### Com base nos contours extrair os caracteres

### Classificar os caracteres extraídos

## Abordagem 3 - Utilização da biblioteca Grounding Dino (deteção de caracteres) e Segment Anything Model (segmentação de caracteres)

### Instalar bibliotecas Grounding Dino e Segmente Anythin Model (SAM)

### Obter os caminhos das imagens recortadas

### Aplicar do Grounding Dino sobre as imagens

### Guardar as imagens geradas, com as bounding boxes

### Aplicar o SAM sobre as imagens geradas pelo Grounding Dino

### Obter as máscaras geradas

### Inverter as cores das máscaras

### Calcular contours da imagem binarizada

### Com base nos contours extrair os caracteres

### Classificar os caracteres extraídos

## Módulo 4 - Análise de texto e correção de erros

### Formato das matrículas portuguesas

### Erros nos resultados obtidos pelo OCR e classificação de caracteres

### Correção de erros

### Guardar os resultados

### Comparar resultados com as matrículas