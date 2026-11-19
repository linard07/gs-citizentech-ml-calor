# Dados — GS CitizenTech (IA & ML) — calor extremo

## Fonte
- **Instituição:** INMET (Instituto Nacional de Meteorologia)
- **Base:** Banco de Dados Históricos (dados horários de estações automáticas)
- **Link:** https://portal.inmet.gov.br/dadoshistoricos
- **Arquivos usados:** `https://portal.inmet.gov.br/uploads/dadoshistoricos/AAAA.zip` (um por ano, de 2013 a 2025)
- **Estação:** A701, São Paulo / Mirante de Santana
- **Período:** 2013 a 2025
- **Natureza:** dados reais e públicos. Nenhum dado simulado.

## Variáveis utilizadas (horárias, agregadas por dia)
Temperatura do ar e temperatura máxima da hora, umidade relativa, pressão atmosférica, velocidade do vento, precipitação e radiação global (esta última com falhas em parte do período).

## Resultado do processamento
- 113.945 registros horários (de 31/12/2012 22h a 31/12/2025 20h, horário de Brasília).
- 4.711 dias utilizáveis após a limpeza (dias com pelo menos 20 horas válidas e com o dia seguinte observado).
- Treino (2013–2022): 3.633 dias, 41 com calor extremo (1,1%). Teste (2023–2025): 1.078 dias, 18 com calor extremo (1,7%).

## Arquivos
- Os dados brutos (zips do INMET) não estão no repositório porque são grandes; o notebook baixa tudo automaticamente (com novas tentativas se o servidor do INMET derrubar a conexão).
- Ao rodar o notebook, ele gera `dados/inmet_a701_diario_processado.csv`, a tabela diária final usada no modelo, e salva as figuras em `figuras/`.

## Como reproduzir
1. `pip install -r ../requirements.txt`
2. Abra `../GS_CitizenTech_ML.ipynb` (no Google Colab ou no Jupyter) e rode todas as células.

## Tratamento
- Valores `-9999` tratados como ausentes.
- Hora UTC convertida para horário de Brasília.
- Dias com menos de 20 horas válidas descartados; valores faltantes dos atributos preenchidos com a **mediana do treino** (sem usar dados do teste).
- Alvo: `calor_extremo_amanha = 1` se a temperatura máxima do dia seguinte for ≥ 35 °C (o mesmo nível de *Atenção* para calor extremo da API em C# e do circuito ESP32 da CitizenTech).

## Limitações
- Uma única estação representa a cidade toda.
- Calor extremo é raro (apenas 59 dias no período), então as métricas do teste (18 eventos) são sensíveis: acertar ou errar um único dia muda o recall em mais de 5 pontos percentuais.
