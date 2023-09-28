# Introdução

Uma empresa de tecnologia procura o desenvolvimento de um algoritmo que faça a deteção de matrículas de veículos e extraia os seus caracteres, especificamente, de matrículas portuguesas, para gestão de entradas e saídas das instalações da empresa.
O algoritmo deve ser constituído por quatro módulos:
-	**Deteção da matrícula**: módulo responsável pela deteção das matrículas. Neste módulo serão recolhidas imagens de veículos com matrícula portuguesa, com as quais será criado um dataset utilizado para treinar um modelo de deep learning. Depois do treino, os pesos (um ficheiro do tipo .pt, por exemplo, “best.pt”), gerados durante o treino, serão utilizados para efetuar inferências sobre imagens nunca antes vistas. Por fim, guardar as imagens com as anotações feitas durante as inferências e os respetivos ficheiros (do tipo .txt) que contêm as coordenadas das bounding boxes, de forma a ser utilizado no próximo módulo. 
-	**Recorte da imagem com base nas coordenadas da bounding box**: módulo responsável por pelo recorte da área onde foi detetada a matrícula. Com os valores presentes em cada ficheiro, calcular as coordenadas do canto superior esquerdo e o canto inferior direito da bouding box. Por fim, utilizar uma biblioteca que permita processar digitalmente imagens (OpenCV) para efetuar o recorte da área da bounding box e guardar as novas imagens que apenas contêm a matrícula.
-	 **Pipeline de processamento digital de imagem**: módulo responsável por extrair os caracteres presentes nas imagens recortadas, obtidas no módulo anterior. De forma a atingir este objetivo existem várias abordagens que podem ser utilizadas:
o	OCR (Optical Character Recognition) - processo que converte imagens com texto em texto capaz de ser compreendido por computadores.
o	Pipeline de processamento digital de imagem - segmentação e extração dos caracteres utilizando técnicas de processamento de imagem.
-	**Análise de texto e correção de erros**: módulo responsável pela análise do texto extraído, de todas as imagens, e correção de potenciais erros cometidos pela abordagem utilizado no módulo anterior.



# Conteúdo

- [Recursos](#software-a-utilizar)
- [Módulo 1 - Deteção da matrícula](#módulo-1---deteção-da-matrícula)
  - [Construção do dataset](#construção-do-dataset)
  - [Preparação do dataset (Roboflow)](#preparação-do-dataset-(roboflow))
    - [Criar projeto de deteção de objetos](#criar-projeto-de-deteção-de-objetos)
    - [Efetuar upload das imagens recolhidas](#efetuar-upload-das-imagens-recolhidas)
    - [Anotar as imagens](#anotar-as-imagens)
    - [Aplicar aumentações (opcional)](#aplicar-aumentações-(opcional))
    - [Efetuar download do dataset no formato desejado](#efetuar-download-do-dataset-no-formato-desejado)
  - [Treino do dataset](#treino-do-dataset)
    - [Resultados do treino](#resultados-do-treino)
  - [Inferir sobre novas imagens](#inferir-sobre-novas-imagens)
- [Módulo 2 - Recorte da imagem com base nas coordenadas da bounding boxes](#módulo-2---Recorte-da-imagem-com-base-nas-coordenadas-da-bounding-boxes)
  - [Organização dos ficheiros com as bounding boxes](#organização-dos-ficheiros-com-as-bounding-boxes)
  - [Obter os caminhos dos ficheiros](#obter-os-caminhos-dos-ficheiros)
  - [Abrir os ficheiros e ler os seus conteúdos](#abrir-os-ficheiros-e-ler-os-seus-conteúdos)
    - [Reverter a normalização das coordenadas geradas](#reverter-a-normalização-das-coordenadas-geradas)
  - [Calcular as coordenadas dos ponto superior esquerdo e do ponto inferior direito](#calcular-as-coordenadas-dos-ponto-superior-esquerdo-e-do-ponto-inferior-direito)
  - [Efetuar o recorte da imagem com base nas coordenadas calculadas](#efetuar-o-recorte-da-imagem-com-base-nas-coordenadas-calculadas)
- [Módulo 3 - Pipeline de processamento digital da imagem](#módulo-3---pipeline-de-processamento-digital-da-imagem)
  - [Pré-processamento das imagens recortadas](#pré---processamento-das-imagens-recortadas)
  - [Abordagem 1 - Utilização de uma biblioteca OCR (Optical Character recognition)](#abordagem-1---Utilização-de-uma-biblioteca-OCR-(Optical-Character-recognition))
    - [Instalar a biblioteca PaddleOCR](#instalar-a-biblioteca-PaddleOCR)
    - [Carregar o modelo responsável pelo reconhecimento de texto](#carregar-o-modelo-responsável-pelo-reconhecimento-de-texto)
    - [Obter os caminhos das imagens recortadas](#obter-os-caminhos-das-imagens-recortadas)
    - [Aplicar o OCR sobre as imagens](#aplicar-o-OCR-sobre-as-imagens)
    - [Guardar os resultados](#guardar-os-resultados)
  - [Abordagem 2 - Aplicação do método de Otsu](#abordagem-2---aplicação-do-método-de-otsu)
    - [Obter o caminho das imagens recortadas](#obter-o-caminho-das-imagens-recortadas)
    - [Aplicar o algoritmo de Otsu](#aplicar-o-algoritmo-de-Otsu)
    - [Verificar o número de píxeis pretos](#verificar-o-número-de-píxeis-pretos)
    - [Calcular contours da imagem binarizada](#calcular-contours-da-imagem-binarizada)
    - [Com base nos contours extrair os caracteres](#com-base-nos-contours-extrair-os-caracteres)
    - [Classificar os caracteres extraídos](#classificar-os-caracteres-extraídos)
  - [Abordagem 3 - Utilização da biblioteca Grounding Dino (deteção de caracteres) e Segment Anything Model (segmentação de caracteres)](#abordagem-3---utilização-da-biblioteca-grounding-dino-(deteção-de-caracteres)-e-segment-anything-model-(segmentação-de-caracteres))
    - [Instalar bibliotecas Grounding Dino e Segment Anythin Model (SAM)](#instalar-bibliotecas-grounding-dino-e-segment-anythin-model-(SAM))
    - [Obter os caminhos das imagens recortadas](#obter-os-caminhos-das-imagens-recortadas)
    - [Aplicar do Grounding Dino sobre as imagens](#aplicar-do-grounding-dino-sobre-as-imagens)
    - [Guardar as imagens geradas](#guardar-as-imagens-geradas)
    - [Aplicar o SAM sobre as imagens geradas pelo Grounding Dino](#aplicar-o-sam-sobre-as-imagens-geradas-pelo-grounding-dino)
    - [Obter as máscaras geradas](#obter-as-máscaras-geradas)
    - [Inverter as cores das máscaras](#inverter-as-cores-das-máscaras)
    - [Calcular contours da imagem binarizada](#calcular-contours-da-imagem-binarizada)
    - [Com base nos contours extrair os caracteres](#com-base-nos-contours-extrair-os-caracteres)
    - [Classificar os caracteres extraídos](#classificar-os-caracteres-extraídos)
- [Módulo 4 - Análise de texto e correção de erros](#módulo-4---Análise-de-texto-e-correção-de-erros)
  - [Formato das matrículas portuguesas](#formato-das-matrículas-portuguesas)
  - [Erros nos resultados obtidos pelo OCR e classificação de caracteres](#erros-nos-resultados-obtidos-pelo-ocr-e-classificação-de-caracteres)
  - [Correção de erros](#correção-de-erros)
  - [Guardar os resultados](#guardar-os-resultados)
  - [Comparar resultados com as matrículas](#comparar-resultados-com-as-matrículas)


# Recursos

Para o desenvolvimento do protocolo serão utilizados os seguintes recursos:

- [Google Colab](https://colab.research.google.com/): desenvolvimento do código e treino dos modelos;
- [Google Drive](https://www.google.com/drive/): guardar os datasets e os resultados do treino;
- [Roboflow](https://roboflow.com/): preparar os dados para o dataset;
- [YOLO (You Only Look Once)](https://github.com/ultralytics): treinar um modelo utilizando o dataset desenvolvido e inferir sobre novas imagens;
- [Otsu](https://en.wikipedia.org/wiki/Otsu%27s_method): segmentar imagens (abordagem 2);
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/README_en.md): deteção automática de texto presente em imagens (abordagem 1);
- [Grounding Dino](https://github.com/IDEA-Research/GroundingDINO): deteção automática de objetos, com base num prompt (abordagem 3);
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything): segmentação automática de objetos (abordagem 3).


# Módulo 1 - Deteção da matrícula

## Construção do dataset

O dataset deve ser constituído por imagens diurnas (tiradas manualmente, obtidas da Internet, vídeo, etc..) de veículos (carros, carrinhas, camiões, motas...) com matrícula portuguesa frontais e traseiras sem muita angulação (vertical ou horizontal) em que os caracteres das da matrícula sejam bem visíveis. A condição de iluminação das imagens também deve ser variada. Quantas mais imagens melhor, embora o tempo de anotação seja maior (sugestão: 200 a 500 imagens). 

<div align="center">

| **Imagens adequadas para o treino** |
|:----:|
| **Imagens não adequadas para o treino** | 

</div>

### Preparação do dataset (Roboflow)

Necessário ter uma conta Roboflow.

Colocar video

### Criar projeto de deteção de objetos

Colocar video

### Efetuar upload das imagens recolhidas

Colocar video

### Anotar as imagens

Colocar video

### Aplicar aumentações (opcional)

As aumentações são opcionais, uma vez que os resultados do treino podem ser os pretendidos mesmo sem as efetuar.
São úteis quando os dados são poucos.

Caso sejam necessárias, as seguintes são as mais pertinentes:

- Flip
- Crop
- Rotation
- Brightness
- Noise

Colocar video

### Efetuar download do dataset no formato desejado

Efetuar o download do dataset no formato do modelo a utilizar no treino e colocar no Google Drive

Colocar video

## Treino do dataset

Parâmetros a ter em conta:

- epochs -> número de iterações do treino
- batch -> o número de imagens utilizadas em cada iteração - este número deve ter em conta o tamanho da imagem e a memória de vídeo da placa gráfica
- img -> tamanho da imagem utilizado como input
- patience -> o número de “epochs” necessárias em que não existe melhoria da loss de validação para parar o treino automaticamente
- name -> o nome da diretoria onde são guardados os resultados do treino

CODIGO

### Resultados do treino

Colocar imagens do treino (metricas)

## Inferir sobre novas imagens

A confiança mínima utilizada deve ser de 0.65.
Utilizar o parâmetro "--save-txt" para guardar ficheiros com as coordenadas das bounding boxes detetadas. Caso não seja detetada nenhuma matrícula, o ficheiro não é gerado.

Colocar imagem do ficheiro txt e da inferencia

# Módulo 2 - Recorte da imagem com base nas coordenadas da bounding boxes

## Organização do ficheiro com as bounding boxes

Cada ficheiro tem pelo menos uma linha de texto constituído por cinco valores:

- class -> não é relevante para o problema em questão;
- x_center -> valor de x, do centro da bounding box;
- y_center -> valor de y, do centro da bounding box;
- width -> largura da bounding box;
- height -> altura da bounding box.

## Obter os caminhos dos ficheiros

CODIGO

## Abrir os ficheiros e ler os seus conteúdos

CODIGO

### Reverter a normalização das coordenadas geradas

CODIGO

## Calcular as coordenadas dos ponto superior esquerdo e do ponto inferior direito

CODIGO

## Efetuar o recorte da imagem com base nas coordenadas calculadas

CODIGO

# Módulo 3 - Pipeline de processamento digital da imagem

## Pré-processamento das imagens recortadas

Pré-processamento efetuado para maximizar os resultados quando aplicadas as diferentes abordagens.

CODIGO

## Abordagem 1 - Utilização de uma biblioteca OCR (Optical Character recognition)

### Instalar a biblioteca PaddleOCR

CODIGO

### Carregar o modelo responsável pelo reconhecimento de texto

CODIGO

### Obter os caminhos das imagens recortadas

CODIGO
  
### Aplicar o OCR sobre as imagens

CODIGO

### Guardar os resultados

CODIGO

Apresentar resultados

## Abordagem 2 - Aplicação do método de Otsu

### Obter o caminho das imagens recortadas

CODIGO

### Aplicar o algoritmo de Otsu

CODIGO

### Verificar o número de píxeis pretos

CODIGO

### Calcular contours da imagem binarizada

CODIGO

### Com base nos contours extrair os caracteres

CODIGO

### Com base nos contours extrair os caracteres

CODIGo

### Classificar os caracteres extraídos

CODIGO

## Abordagem 3 - Utilização da biblioteca Grounding Dino (deteção de caracteres) e Segment Anything Model (segmentação de caracteres)

### Instalar bibliotecas Grounding Dino e Segment Anythin Model (SAM)

CODIGO

### Obter os caminhos das imagens recortadas

CODIGO

### Aplicar do Grounding Dino sobre as imagens

CODIGO

### Guardar as imagens geradas

CODIGO

### Aplicar o SAM sobre as imagens geradas pelo Grounding Dino

CODIGO

### Obter as máscaras geradas

CODIGO

### Inverter as cores das máscaras

CODIGO

### Calcular contours da imagem binarizada

CODIGO

### Com base nos contours extrair os caracteres

CODIGO

### Classificar os caracteres extraídos

CODIGO

# Módulo 4 - Análise de texto e correção de erros

### Formato das matrículas portuguesas

Tabela com
Imagens das diferentes matriculas

### Erros nos resultados obtidos pelo OCR e classificação de caracteres

Exemplos de erros

### Correção de erros

Como a correcao pode ser efetuada

### Comparar resultados com as matrículas