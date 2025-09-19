# API de Predicción de Glucosa con FastAPI

Este proyecto implementa una API REST utilizando FastAPI para servir un modelo de Machine Learning pre-entrenado que predice los niveles de glucosa en sangre. La API está diseñada para ser robusta, segura y fácil de integrar con otros servicios como n8n y ManyChat.

## Características

- **Endpoint de Predicción**: Recibe datos de un paciente y devuelve una predicción de glucosa.
- **Seguridad**: Protegido mediante una clave de API (API Key) en las cabeceras.
- **Validación de Datos**: Utiliza Pydantic para validar los datos de entrada y garantizar su integridad.
- **Manejo de Datos Faltantes**: El pipeline de `scikit-learn` está preparado para imputar valores faltantes automáticamente.
- **Documentación Automática**: Documentación interactiva de la API (Swagger UI) disponible en `/docs`.
- **Configuración Flexible**: La ruta del modelo y la clave de API se gestionan a través de variables de entorno.
- **Endpoint de Salud**: Un endpoint `/health` para verificar el estado del servicio.

## Tecnologías Utilizadas

- **Framework**: FastAPI
- **Servidor ASGI**: Uvicorn
- **Machine Learning**: Scikit-learn, Pandas, Numpy
- **Validación de Datos**: Pydantic
- **Carga del Modelo**: Joblib
- **Variables de Entorno**: python-dotenv

## Instalación y Configuración

Sigue estos pasos para poner en marcha el proyecto en tu entorno local.

### 1. Prerrequisitos

- Python 3.8 o superior.
- `pip` y `venv` para la gestión de paquetes y entornos virtuales.

### 2. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio
```


### 3. Crear y Activar un Entorno Virtual

Es una buena práctica aislar las dependencias del proyecto.

```bash
# Para Windows
python -m venv env
.\env\Scripts\activate

# Para macOS/Linux
python3 -m venv env
source env/bin/activate
```

### 4. Instalar Dependencias

Instala todas las librerías necesarias desde el archivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 5. Configurar Variables de Entorno

La configuración de la aplicación se gestiona a través de un archivo `.env`.

1.  Crea una copia del archivo de ejemplo:
    ```bash
    # En Windows
    copy .env.example .env
    
    # En macOS/Linux
    cp .env.example .env
    ```

2.  Abre el archivo `.env` y edita las variables:
    ```dotenv
    # Ruta al modelo de Machine Learning
    MODEL_PATH=RF_GlucosaMujeres.joblib

    # Clave secreta para proteger el endpoint de predicción
    API_KEY="TU_CLAVE_SECRETA_AQUI"
    ```
    - **`MODEL_PATH`**: Asegúrate de que tu modelo (`.joblib`) se encuentre en la ruta especificada.
    - **`API_KEY`**: Genera una clave secreta y segura.

## Ejecutar la Aplicación

Para iniciar la API, utiliza Uvicorn:

```bash
uvicorn main:app --reload
```

- `main`: El archivo `main.py`.
- `app`: El objeto `FastAPI` creado en `main.py`.
- `--reload`: El servidor se reiniciará automáticamente cada vez que modifiques el código.

Una vez que el servidor esté en marcha, verás un mensaje como este:
`INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`

Puedes acceder a la documentación interactiva en http://127.0.0.1:8000/docs.

##  Uso de la API

### Endpoint de Salud (`GET /health`)

Verifica si la API está funcionando correctamente.

- **URL**: `/health`
- **Método**: `GET`
- **Respuesta Exitosa (200 OK)**:
  ```json
