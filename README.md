# Configuración inicial para usar el código

1. Configuración del Almacenamiento (Google Drive)
Acceso al directorio: Confirme que posee acceso a la carpeta compartida del proyecto denominada CEIA_VpC2_TP. Se requieren permisos de Editor para generar y guardar nuevos resultados (como el renderizado de la Etapa 06), o de Lector si únicamente se desea ejecutar la evaluación de métricas.

Creación de acceso directo (Paso Crítico): Ingrese a su cuenta de Google Drive y diríjase a la sección "Compartido conmigo" (Shared with me). Haga clic derecho sobre la carpeta CEIA_VpC2_TP y seleccione "Agregar acceso directo a Drive", eligiendo la raíz de "Mi unidad" como destino.

Nota: Este procedimiento sincroniza el directorio, permitiendo que el entorno virtual de Colab localice los archivos bajo la ruta estandarizada /content/drive/MyDrive/CEIA_VpC2_TP/... independientemente del usuario que inicie la sesión.

2. Inicialización y Ejecución en Colab
Habilitación del montaje: Al abrir el cuaderno en Google Colab, diríjase a la primera celda de código (sección 1. Configuración del Entorno, Google Drive y Reproducibilidad) y verifique que la variable de persistencia se encuentre activa:

Python
MOUNT_DRIVE = True
Autorización de lectura: Al ejecutar esta primera celda, el entorno solicitará autorización estándar para acceder a su cuenta de Drive. Una vez otorgados los permisos, el cuaderno quedará enlazado a la carpeta del proyecto.

Flujo de evaluación: A partir de este punto, el cuaderno puede ejecutarse de manera secuencial. El sistema accederá automáticamente a los modelos almacenados (best.pt) y al video de entrada, procesando las métricas (Etapa 04) y el posprocesamiento y anotación del video (Etapas 05 y 06) sin requerir modificaciones adicionales en el código.




# ⚽ Detección de Jugadores, Pelota y Árbitros en Fútbol
## Visión por Computadora II – CEIA – FIUBA
### **Grupo 11**

Integrantes:
a2539 | Agustín Ross
a2510 | Ayelén Calo
a2529 | María Teresa Mallaupoma León
a2406 | Andrés Pablo García
---

## 📌 1. Descripción del Proyecto
Este proyecto implementa un sistema de **visión por computadora y deep learning** para la detección automática de los cuatro elementos fundamentales en un partido de fútbol:
1. **Pelota (`ball`)**
2. **Arqueros (`goalkeeper`)**
3. **Jugadores de campo (`player`)**
4. **Árbitros (`referee`)**

El flujo abarca desde la descarga, curación y análisis exploratorio del dataset (EDA), el fine-tuning y la comparación de 3 variantes de YOLOv8 orientadas a mejorar la detección de la pelota (~5 px), la evaluación formal (P, R, F1, $mAP@50$, $mAP@50-95$, matriz de confusión y latencia), la calibración de umbrales de posprocesamiento (Confidence Threshold y NMS), hasta la anotación de un video de partido con cajas y un HUD con el conteo por clase.

**Resultado:** el modelo elegido (YOLOv8s a 1280 px, FP16) eleva el AP50 de la pelota de 0,51 a 0,70 respecto del baseline y procesa 43 FPS en una GPU T4, lo que lo haría viable para tiempo real.

---

## 🗺️ 2. Arquitectura del Sistema (Diagrama de Etapas)

```
====================================================================================================
FASE A: ENTRENAMIENTO Y EVALUACIÓN (Google Colab / Local con GPU)
====================================================================================================
[01. Dataset]
  • Kaggle: oussamamoussa/foot-ball-dataset (Formato YOLO)
  • 4 clases: ball, goalkeeper, player, referee
  • 53.445 imágenes totales (Train: 46.511 | Valid: 4.757 | Test: 2.177)
       │
       ▼
[02. Curación, EDA y Preprocesamiento]
  • Limpieza: Detección y eliminación de 2.620 imágenes sin etiqueta (fondos vacíos)
  • Submuestreo al 50% con semilla fija (seed=42): 25.412 imágenes (22.174 / 2.220 / 1.018)
  • Conteo y diagnóstico de desbalance: player >> resto (>85% jugadores)
  • Análisis dimensional: Pelota ≈ 5 px de lado (mediana a 640x640)
  • Data Augmentation: fliplr=0.5, flipud=0.0 (desactivado), variaciones HSV, mosaic
       │
       ▼
[03. Modelo Principal: Fine-tuning YOLO (3 Modelos - 15 Épocas - save_period=5)]
  • Transfer Learning desde MS COCO
  • Modelo 1 (Baseline): imgsz 640 (yolov8s.pt)
  • Modelo 2 (Alta Resolución): imgsz 1280 (yolov8s.pt, pelota pasa de 5 px a ~10 px)
  • Modelo 3 (Objetos Pequeños): imgsz 640 con Cabeza P2 (stride 4, yolov8s-p2.yaml)
  • Persistencia: Checkpoints cada 5 épocas (epoch5.pt, epoch10.pt, epoch15.pt, best.pt)
  • Resiliencia: Reanudación de entrenamiento con resume=True ante desconexiones
       │
       ▼
[04. Evaluación (validación y test)]
  • Métricas: P, R, F1, mAP@50, mAP@75 y mAP@50-95 por clase para los 3 modelos
  • Matriz de confusión: pelota vs. fondo y arquero vs. jugador
  • Latencia por frame (batch=1, FP32 y FP16)
  • Modelo elegido: M2 (1280) en FP16
       │
       ▼
[05. Posprocesamiento]
  • Barrido de Confidence Threshold y NMS IoU en validación (máximo F1 macro)
  • Elegido: conf = 0.30, iou = 0.5, confirmado en test
       │
       ▼
[06. Aplicación en Video]
  • Clip DFL Bundesliga (1080p, 25 FPS) anotado cuadro a cuadro con cajas, clases, scores
    y HUD con conteo por clase y FPS de procesamiento
```
```


