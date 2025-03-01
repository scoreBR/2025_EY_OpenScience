# 2025_EY_OpenScience
Este script desafia o desenvolvedor a prever o índice de ilhas de calor utilizando uma ML de regressão, que obtém informações diretamente do satélite Sentinel A-2, provida pela Microsoft.

## Training_data_uhi_index_2025
Este arquivo nos foi entregue pela EY para modelo e treinamento da ML que produziriamos, apresentando dados coletados em Nova York e Bronx, nos Estados Unidos, incluindo o horário em que foi captado, coordenadas do ponto e o índice de calor do local. 

Os participantes estão estritamente proibidos de usar valores de Longitude e Latitude como recursos na construção de seus modelos de aprendizado de máquina. Envios que utilizem valores de longitude e latitude como características do modelo serão desqualificados. Esses valores devem ser utilizados apenas para compreensão dos atributos e características dos locais.

A incorporação de dados de latitude e longitude em suas formas brutas ou por meio de qualquer forma de manipulação (incluindo multiplicação, incorporação ou conversão em coordenadas polares) como recursos preditivos em seu modelo é estritamente proibida, pois pode comprometer a adaptabilidade de seu modelo em diversos cenários. Esta proibição estende-se ao cálculo da distância a um ponto de referência e à sua utilização como feição, o que é essencialmente uma transformação das coordenadas geográficas originais numa nova forma de feição. As inscrições que incluam esses tipos de recursos serão consideradas não conformes e serão desqualificadas.

## UHI_ML.py



## Submission.csv
O arquivo de Submission será entregue para a EY com as previsões feitas pela ML e calculada por eles para avaliar a precisão do script.
