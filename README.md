# ProjetoLibras
Projeto desenvolvido na Mentoria usando Deep Learning

# Reconhecimento de LIBRAS com I.A

Algoritmo de visão computacional para reconhecimento de sinais do alfabeto de Libras utilizando webcam em tempo real.

## Pré-requisitos

Antes de executar o projeto, é necessário  possuir:

* Python 3.11 (recomendado)
* VS Code (recomendado)
* Webcam funcional

## Modelo de Machine Learning

O modelo utilizado neste projeto foi treinado utilizando o **Teachable Machine**, uma plataforma que permite criar modelos de aprendizado de máquina de forma simplificada e sem necessidade de programação avançada.

Após o treinamento com imagens de gestos de Libras, o modelo foi exportado no formato **Keras (`keras_model.h5`)**, integrado com o ecossistema TensorFlow. Esse arquivo contém tanto a arquitetura da rede neural quanto os pesos ajustados durante o treinamento.

No projeto, o modelo é carregado diretamente em Python utilizando a biblioteca Keras/TensorFlow e é responsável por realizar a classificação em tempo real dos gestos capturados pela webcam.

Além disso, é utilizado o arquivo `labels.txt`, que contém o mapeamento das classes aprendidas pelo modelo, permitindo a interpretação das previsões em formato legível.


## Instalação

### 1. Baixe ou clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
```

ou faça o download do arquivo `.zip` e extraia a pasta do projeto.

### 2. Abra o projeto no VS Code

Abra a pasta do projeto no VS Code e inicie um terminal na raiz do repositório.

### 3. Crie o ambiente virtual

No terminal iniciado, execute:
```bash
python -m venv .venv
```

### 4. Ative o ambiente virtual


No terminal do Windows execute:

```bash
.venv\Scripts\activate
```

Após a ativação  do ambiente virtual `.venv`, estará visível no terminal em destaque: `(.venv)`.

### 5. Instale as dependências

```bash
pip install -r requirements.txt
```

Aguarde a conclusão da instalação de todas as bibliotecas necessárias.



## Executando o Projeto

Com o ambiente virtual ativado, execute no terminal:

```bash
python main.py
```

A câmera será aberta automaticamente e o sistema iniciará o reconhecimento dos sinais.


## Solução de Problemas

### Erro ao ativar o ambiente virtual no PowerShell

Caso apareça uma mensagem relacionada à política de execução de scripts, execute o projeto utilizando o Prompt de Comando (CMD) ou altere a política de execução do PowerShell.

### Dependências não encontradas

Certifique-se de que o ambiente virtual está ativado antes de executar:

```bash
pip install -r requirements.txt
```

## Autores

- Jennifer Aguiar
- Lucas de Carvalho
- Samile Barreto
- Saul Ramos
- Victor Almeida
