# Despliegue — Dashboard Saber 11 Córdoba

Contenedor Docker con el tablero Dash que sirve los modelos predictivos del proyecto.

## Estructura

- `Dockerfile` — define la imagen del contenedor (Python 3.11 slim + dependencias del dashboard).
- `.dockerignore` (en la raíz del repo) — excluye archivos pesados o innecesarios al construir la imagen (notebooks, mlruns, presentaciones, etc.).

## Construir la imagen

Desde la raíz del repositorio:

```bash
docker build -f despliegue/Dockerfile -t saber11-dash .
```

## Correr el contenedor localmente

```bash
docker run -d -p 8050:8050 --name saber11-dash saber11-dash
```

Abrir el navegador en `http://localhost:8050`.

Para ver los logs:

```bash
docker logs -f saber11-dash
```

Para detenerlo y eliminarlo:

```bash
docker stop saber11-dash
docker rm saber11-dash
```

## Desplegar en AWS EC2

1. Crear una instancia EC2 (Ubuntu 22.04, t2.micro o t3.small, free tier).
2. Configurar el security group para permitir tráfico entrante en el puerto 8050 (TCP) desde 0.0.0.0/0.
3. SSH a la instancia.
4. Instalar Docker:
   ```bash
   sudo apt update
   sudo apt install -y docker.io
   sudo systemctl enable --now docker
   sudo usermod -aG docker ubuntu
   ```
   Cerrar y volver a abrir la sesión SSH para que el usuario tome el grupo.
5. Clonar el repositorio:
   ```bash
   git clone https://github.com/abrahambohorquez/Proyecto2ACTM.git
   cd Proyecto2ACTM
   ```
6. Construir y correr el contenedor:
   ```bash
   docker build -f despliegue/Dockerfile -t saber11-dash .
   docker run -d -p 8050:8050 --restart unless-stopped --name saber11-dash saber11-dash
   ```
7. El dashboard queda accesible en `http://<EC2_PUBLIC_IP>:8050`.
