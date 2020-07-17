# Reconocimiento facial de colaboradores usando Facenet y Siamese Networks


## Problematica

Actualmente existen modelos de clasificación con gran rendimiento. Esto se da en base al entrenamiento con grandes cantidades de datos y clases definidas. La situación se complica cuando se requiere agregar nuevas clases a las ya existentes. El modelo requerirá gran cantidad de datos de las nuevas clases y volver a entrenarse con el objetivo de ajustar sus pesos. El aprendizaje "One Shot" aparece para mitigar este problema porque puede realizar predicciones con solo un ejemplo de cada nueva clase. La idea es entrenar un modelo que aprenda a diferenciar entre distintas imagenes.

## Estado del arte en One Shot Learning

### Facenet
FaceNet es una red neuronal que aprende un mapeo de imágenes faciales a un espacio euclidiano compacto, donde las distancias corresponden a una medida de similitud de caras. Entonces, cuanto más similares sean las dos imágenes faciales, menor será la distancia entre ellas. El método de Triplet Loss fue introducido por FaceNet para la tarea de reconocimiento de rostros, donde los autores proponen un nuevo enfoque en el entrenamiento de redes neuronales siamesas que permitan discriminar si dado 2 imágenes de rostros, estos pertenecen a la misma persona. Triplet Loss optimiza el espacio de búsqueda de tal manera que la entidades del mismo tipo se mantienen cercanos y al mismo tiempo entidades de diferente tipo se mantienen alejados. En la imagen se puede observar como se ingresan 3 imagenes diferentes al mismo modelo para efectuar el Triplet Loss.

![alt text](https://github.com/Ceviche98/reconocimiento-facial-de-colaboradores/blob/master/Implementacion%20del%20sistema/assets/triplet_loss_example.png?raw=true)

La tarea del Triplet Loss es que con el paso de las iteraciones la distancia entre el anchor y el positive(imagenes similares) sea pequeña y la distancia entre el anchor y el negative (imagenes diferentes) sea grande, todo usando el mismo modelo.

![alt text](https://github.com/vcarlosrb/reconocimiento-facial-de-colaboradores/blob/master/Implementacion%20del%20sistema/assets/triplet_loss_function.png?raw=true)

### Siamese Network
La arquitectura de las Redes Siamesas contiene dos subredes convolucionales que poseen los mismos pesos teniendo imágenes de entrada diferentes. Luego de la última capa convolucional de cada subred, se transforma la capa a un solo vector. Por último, se calcula la distancia entre los vectores de salida de cada subred y este vector de distancia pasa a una unidad sigmoidal. Las Redes Siamesas son muy útiles, ya que aprenden a diferenciar imágenes, en lugar de aprender a clasificar. Además, las dos subredes comparten parámetros por lo que hay una menor tendencia a tener overfitting.

![alt text](https://github.com/Ceviche98/reconocimiento-facial-de-colaboradores/blob/master/Implementacion%20del%20sistema/assets/siamese.png?raw=true)

## Bases de datos utilizadas

### MS-Celeb-1M dataset
Esta base de datos contiene 10 millones de imagenes recogidas de internet con el proposito de mejorar la tecnologia de reconocimiento de rostros. Contiene imagenes de alrededor de 100,000 personas. Una gran parte de esta base de datos esta compuesta por actores norteamericanos y britanicos.MS-Celeb-1M dataset fue utilizada para generar el modelo preentrenado de Facenet que utilizamos en este proyecto. 

![alt text](https://github.com/Ceviche98/reconocimiento-facial-de-colaboradores/blob/master/Implementacion%20del%20sistema/assets/msceleb.jpg?raw=true)

### Georgia Tech face database
Esta base contiene imagenes de personas del Center for Signal and Image Processing de Georgia Institute of Technology. Cada persona fue representada con 15 imagenes en formato JPG y hay una opcion para descargar solo las imagenes de los rostros con un tamaño promedio de 150x150. Lo interesante de esta base es que muestra imagenes con diferentes expresiones, posiciones o condiciones de luz. Esta base de datos la utilizamos para entrenar una Red Siamesa desde 0, generando 10000 pares de imagenes similares y otros 10000 pares de imagenes diferentes. La expliacion del proceso mas detallado se encuentra dentro del notebook **[Experimentación]Redes Siamesas.ipynb**

![alt text](https://github.com/Ceviche98/reconocimiento-facial-de-colaboradores/blob/master/Implementacion%20del%20sistema/assets/Georgia-Tech-Faces-dataset.png?raw=true)

## Implementación del proyecto
En este proyecto se han implementado los dos modelos: Facenet y Siamese Network. Se decidio entrenar al modelo de Siamese Network desde cero, debido a que queriamos utilizar imagenes a colores en un tamaño de imagen cercano a los 160x160 y otros modelos utilizaban imagenes de menor tamaño. Con respecto a Facenet, los resultados de identificacion que dio desde el principio fueron muy buenos por lo que no nos vimos en la necesidad de reentrenarlo.

En el notebook **Face identification.ipynb** que se encuentra dentro de la carpeta **Implementacion del Sistema**, se ha desarrollado un programa realizar la identificacion de usuarios utilizando Facenet. Primero, se realiza la deteccion del rostro utilizando la función Haarcascades de OpenCV.Luego, se se pasa la imagen a una representacion vectorial utilizando el modelo Facenet. A partir de esta representacion, se calcula la distancia de la imagen con las otras imagenes de los usuarios. Dependiendo de la distancia resultante y el threshold, se puede identificar al usuario o dejarlo como usuario no identificado.

Para realizar lo anteriormente mencionado se necesita utilizar la funcion **detect_face_realtime** 

Con respecto a la Red Siamesa, su entrenamiento y prediccion se encuentra dentro del notebook **[Experimentación]Redes Siamesas.ipynb**. Sin embargo, decidimos utilizar Facenet debido a que mostro mejores resultados en la identificacion de rostros de los participantes de este proyecto. Es un resultado esperado, dado que el dataset que escogimos no es igual de grande al utilizado por Facenet y que utilizar una mayor cantidad de imagenes conllevaria mucho tiempo de procesamiento. Para futuros trabajos, se podria entrenar la Siamese Network con el mismo dataset(MS-Celeb-1M dataset) para comparar de mejor manera los dos modelos.


# Ejecución del proyecto
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
