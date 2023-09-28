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

- [Software a Utilizar](#software-a-utilizar)
- [Módulo 1](#módulo-1)
- [Dataset](#dataset)
- [Treino do Dataset](#treino-do-dataset)
- [Inferências](#inferências)
- [Módulo 2](#módulo-2)
- [Módulo 3](#módulo-3)
- [Módulo 4](#módulo-4)

## Recursos

Para o desenvolvimento do protocolo serão utilizados os seguintes recursos:

- [Google Colab] (https://colab.research.google.com/): desenvolvimento do código e treino dos modelos;
**Google Drive**: guardar os datasets e os resultados do treino;
**Roboflow**: preparar os dados para o dataset;
**YOLO (You Only Look Once)**: treinar um modelo utilizando o dataset desenvolvido e inferir sobre novas imagens;
**Otsu**: segmentar imagens (abordagem 2);
**PaddleOCR**: deteção automática de texto presente em imagens (abordagem 1);
**Grounding Dino**: deteção automática de objetos, com base num prompt (abordagem 3);
**Segment Anything Model**: segmentação automática de objetos (abordagem 3).


## Módulo 1
Este é o primeiro módulo do projeto, onde abordamos ...

## Dataset
O conjunto de dados utilizado neste projeto é...

## Treino do Dataset
Nesta seção, descrevemos o processo de treinamento do modelo usando o conjunto de dados mencionado acima.

## Inferências
Aqui, discutimos os resultados das inferências feitas pelo modelo treinado.

## Módulo 2
Este é o segundo módulo do projeto, onde exploramos...

## Módulo 3
No terceiro módulo, abordamos...

## Módulo 4
O quarto módulo é dedicado a...

