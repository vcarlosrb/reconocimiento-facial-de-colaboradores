# Reconocimiento Facial de Colaboradores
### Proyecto Final de Técnicas Avanzadas de Data Mining y Sistemas Inteligentes

Este proyecto permite el reconocimiento de rostros en tiempo real mediante el uso de una camara web.

La solución continene los siguientes pasos principales:
  - Detección de rostro
  - Reconocimiento de rostro
  
Para el paso de detección de rostro nos apoyamos de la funcion de **OpenCV "Haar Cascade"**, posteriormente obtenemos el bounding box y ajustamos la imágen para que pueda ingresar al modelo. Para el paso de Reconocimiento de rostro de utilizó el **modelo pre-entrenado FaceNet** el cual nos brinda buenos vectores caracteristicos debido a que usa la **función de perdida en tripleta (triplet loss function)**.

Se tomo como base el siguiente proyecto: [FaceRecog](https://github.com/susantabiswas/FaceRecog)
  
## 1. Requisitos
Ingresar a la carpeta "Implementación del sistema". Descargar el modelo en el siguiente enlace [facnet_keras.h5](https://drive.google.com/file/d/1wsJs5ZnhI7meqdOX6S9Indm9l1zmsisH/view?usp=sharing), posteriormente guardarlo dentro de la carpeta "models".

## 2. Reconocimiento Facial
1. Iniciar la base de datos

    - Ejecutar `user_db = ini_user_database()`
    
2. Agregar un usuario
    - Ejecutar `add_user_webcam(user_db, FRmodel, "Nombre de usuario")`
    
3. Reconocimiento facial
    - Ejecutar `face_identification(user_db, FRmodel)`
    
## 3. Reconocimiento Facial en tiempo real
Por medio del aprametro "threshold" se puede controlar la precision del reconocmiento.

Ejecutar `detect_face_realtime(user_db, FRmodel, threshold = 6)`
