# COMO USAR:

Este cuaderno está hecho para usar en COLAB.

Para trabajos en grupo la forma más práctica es usar un Drive. (AGUS -> Les di acceso a todos a mi Drive)

Para que el cuaderno funcione tal cual está en las sesiones de Colab de tus compañeros, solo deben hacer un pequeño "truco" con los accesos directos de Google Drive.

Este es el paso a paso que deben seguir:

LISTO -> Compartir tu carpeta: Ve a tu Google Drive, haz clic derecho en tu carpeta principal CEIA_VpC2_TP y compártela con las cuentas de Google de tus compañeros. Dales permiso de Editor si quieres que el video que ellos procesen en la Etapa 06 se guarde directamente en esa carpeta, o de Lector si solo van a evaluar métricas.

Crear el acceso directo (El paso clave para ellos): Tus compañeros deben entrar a su propio Google Drive, ir a la pestaña "Compartido conmigo" (Shared with me), hacer clic derecho sobre la carpeta CEIA_VpC2_TP y elegir "Agregar acceso directo a Drive" -> "Mi unidad".

Ejecutar el cuaderno: Cuando ellos abran el Colab y corran la celda de configuración con MOUNT_DRIVE = True, Colab montará sus respectivos Drives. Al tener el acceso directo en la raíz de su "Mi unidad", la ruta /content/drive/MyDrive/CEIA_VpC2_TP/... funcionará mágicamente como si la carpeta fuera de ellos, leyendo tus pesos entrenados y el video original sin cambiar ni una línea de código.





# ⚽ Detección de Jugadores, Pelota y Árbitros en Fútbol
## Visión por Computadora II – CEIA – FIUBA
### **Grupo 11**
---

## 📌 1. Descripción del Proyecto
Este proyecto implementa un sistema integral de **visión por computadora y deep learning** para la detección y seguimiento automático de los cuatro elementos fundamentales en un partido de fútbol:
1. **Pelota (`ball`)**
2. **Arqueros (`goalkeeper`)**
3. **Jugadores de campo (`player`)**
4. **Árbitros (`referee`)**

El flujo abarca desde la descarga, curación y análisis exploratorio del dataset (EDA), el fine-tuning y comparación de 3 modelos de detección en tiempo real (familia YOLO), la evaluación formal con métricas ($mAP@50$, $mAP@50-95$, matriz de confusión arquero vs. jugador), la calibración de umbrales de posprocesamiento (Confidence Threshold y NMS), hasta la inferencia sobre secuencias de video con anotaciones y HUD en tiempo real.

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
  • Submuestreo al 50% con semilla fija (seed=42) para dinamismo computacional (~25.400 imágenes)
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
[04. Evaluación]
  • Métricas: mAP@75, mAP@50 y mAP@50-95 por clase para los 3 modelos. Para validación y para testeo.
  • Matriz de confusión: Arquero vs. Jugador (separación de roles morfológicamente similares)
  • Selección del modelo ganador
       │
       ▼
[05. Posprocesamiento]
  • Confidence Threshold calibrado (conf = 0.35)
  • Non-Maximum Suppression (iou = 0.60) para preservar cajas en disputas cuerpo a cuerpo
       │
       ▼
[06. Aplicación en Video]
  • Video de partido anotado cuadro a cuadro con cajas, clases, scores y HUD estadístico en vivo
```

---


