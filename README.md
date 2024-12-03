
# **Stock Price Predictor API**

## **Descrição do Projeto**
Esta API utiliza um modelo LSTM (Long Short-Term Memory) para prever preços de ações com base em dados históricos. O modelo foi treinado com dados obtidos da biblioteca `yfinance`.

A API foi desenvolvida com **FastAPI** e está configurada para rodar em um container **Docker**. Os endpoints foram testados utilizando o **Insomnia**.

---

## **Requisitos**
- Python 3.9+
- Docker
- Insomnia ou outra ferramenta para testar APIs (opcional)
- Dependências listadas no `requirements.txt`

---

## **Instalação e Configuração**

### **1. Clonar o Repositório**
Clone o repositório para sua máquina local:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### **2. Construir e Rodar com Docker**

#### **Construir a Imagem Docker**
No diretório do projeto, execute:

```bash
docker build -t stock-price-api .
```

#### **Rodar o Container**
Execute o container com:

```bash
docker run -d -p 8888:8888 --name stock-api-container stock-price-api
```

A API estará acessível em: `http://localhost:8888`.

---

## **Endpoints da API**

### **1. `/predict/` (POST)**
**Descrição:** Recebe dados históricos de preços de ações e retorna o próximo preço previsto.

**Exemplo de Requisição:**
```json
POST /predict/
Content-Type: application/json

{
    "historical_data": [150.0, 152.3, 153.2, 155.4, 157.8, 159.1, 158.7, 160.0, 161.2, 162.5, 163.0, 162.8, 163.4, 164.0, 165.3, 166.0, 166.7, 167.5, 168.0, 169.2, 170.0, 171.5, 172.0, 173.3, 174.1, 175.0, 176.5, 177.0, 177.8, 178.2, 179.0, 179.7, 180.3, 181.0, 181.8, 182.5, 183.0, 184.0, 184.5, 185.3, 186.0, 186.7, 187.5, 188.2, 189.0, 189.8, 190.3, 191.0, 192.5, 193.0, 193.8, 194.2, 195.0, 195.7, 196.3, 197.0, 198.5, 199.0, 199.8, 200.2]
}
```

**Resposta:**
```json
{
	"predicted_price": 200.7896139358945,
	"response_time_seconds": 0.3302285671234131
}
```

### **2. `/monitor/` (GET)**
**Descrição:** Retorna as métricas de avaliação do modelo.

**Exemplo de Resposta:**
```json
{
	"model_name": "LSTM Stock Price Predictor",
	"metrics": {
		"mae": 2.860363558403351,
		"rmse": 4.163185170829687,
		"mape": 2.5867933838407695
	},
	"status": "API is running"
}
```

---

## **Testando com Insomnia**
1. Abra o Insomnia.
2. Crie um novo workspace para este projeto.
3. Adicione as seguintes requisições:
   - **POST** para `/predict/`
   - **GET** para `/monitor/`
4. Use o exemplo de payload fornecido acima para testar.