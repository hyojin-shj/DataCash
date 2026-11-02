[English](README.md) · [한국어](README.ko.md) · **Español** · [中文](README.zh-CN.md) · [日本語](README.ja.md)

## DataCash Promotion Video

<p align="center">
  <a href="https://youtu.be/gw3gqoHlFvA"><img src="docs/demo_video.jpg" width="80%" alt="Video de presentación de DataCash (2:35)"></a>
</p>

## DataCash Technique Video

<p align="center">
  <a href="https://youtu.be/o1sGUME0GHw"><img src="docs/demo_3min.en.jpg" width="80%" alt="Demo técnica de DataCash de 3 minutos"></a>
</p>

<p align="center">Demo técnica de 3 minutos · <a href="https://youtu.be/o1sGUME0GHw">English</a> · <a href="https://youtu.be/vClUIqrZjBY">한국어</a></p>

---

# DataCash

**Proyecto de datos para IA física**<br>
**Solo con el teléfono y el reloj que ya hay en casa, recogemos datos de manos humanas de los que aprenderán los robots.**

Las personas graban en casa misiones de habilidades básicas para robots con su teléfono y su reloj, y reciben una recompensa.<br>
Las empresas de humanoides e IA física pueden comprar **datos de cinco sentidos (vista, oído, tacto, movimiento y fisiología)** alineados en un mismo eje temporal, directamente en formato **LeRobot v3.0**.

Meta Global AI Developer Hackathon (Indie Developer)

<p align="center">
  <img src="docs/1026_datacash_brand.png" width="90%" alt="DataCash">
</p>

---

## Tecnologías clave

### 1. Sustitución por personas virtuales

- **Sustitución de rostro:** en lugar de desenfocar, el rostro del participante se cambia por una persona virtual creada con Muse Image (LivePortrait + MediaPipe).<br>
  Se eligen candidatas del mismo género y edad similar; se prueban 20 fotogramas y, si el reconocimiento facial (SFace) considera "la misma persona" a más de la mitad, se pasa a la siguiente.
- **Continuidad:** si un fotograma sustituido sale demasiado parecido, se regenera con menos expresión en lugar de pixelarlo.
- **Ocultación:** SAM 3.1 oculta a las personas que no consintieron la recogida de datos, niños, pantallas, espejos, cuadros y documentos.
- **Originales borrados:** en el servicio, el video y el audio originales se borran en cuanto termina el procesamiento.<br>
  En el servidor de demostración del hackathon solo esa llamada está desactivada, para poder mostrar la comparación original ↔ sustituido.

#### Cómo evolucionó la sustitución de rostros

| Fecha | Cambio | Resultado |
|---|---|---|
| 10-27 | Sustituir por una persona virtual en vez de desenfocar (LivePortrait, 34 personas virtuales de Muse Image), comprobación de reidentificación en cada fotograma | La cara queda, solo cambia la persona · 3 de 387 fotogramas pixelados |
| 10-27 | Velocidad: el Conv3d lento en la GPU del Mac dividido en 7 convoluciones 2D + fp16 | Una grabación 636 → 84 s |
| 11-01 | Perfiles: 24 fotos en ángulo de la misma persona virtual (edición de Muse Image) → la foto que coincide con el ángulo del fotograma es la fuente | Cabezas inclinadas y perfiles menos deformados |
| 11-01 | Elección de la persona virtual: dentro del mismo género · edad · grupo, se prueba cada candidata sustituyendo fotogramas de prueba y se elige la menos parecida al participante (+36 personas virtuales de Asia oriental, 9 $) | Fotogramas reconocidos con reidentificación estricta 3,3 / 11,1 / 17,3 / 30,9 % → 2,4 / 6,4 / 1,7 / 5,0 % |

<p align="center">
  <img src="docs/faceswap_views.jpg" width="85%" alt="Fotos en ángulo de cuatro personas virtuales (frente + 8)"><br>
  <sub>Varias fotos en ángulo de la misma persona para cada persona virtual</sub>
</p>

<p align="center">
  <img src="docs/faceswap_angle_source.es.png" width="85%" alt="Perfiles y cabeza inclinada desde una sola foto frontal vs los mismos fotogramas desde la foto de ángulo más cercano"><br>
  <sub>Perfil desde una sola foto frontal vs el mismo fotograma desde la foto de ángulo más cercano</sub>
</p>

<p align="center">
  <img src="docs/faceswap_donor_pick.es.png" width="85%" alt="Antes: la persona de foto más parecida / Después: la menos parecida en pruebas con mismo género · edad · grupo, y la tasa de fotogramas reconocidos"><br>
  <sub>Antes y después de la elección de la persona virtual: tasa de fotogramas reconocidos</sub>
</p>

<p align="center">
  <img src="docs/faceswap_more_donors.es.png" width="85%" alt="Nuevas personas virtuales y fotogramas reconocidos antes/después"><br>
  <sub>Fotogramas reconocidos antes y después de añadir más personas virtuales</sub>
</p>

### 2. Tacto sin guante — método GripSense

Trasladamos *GripSense* de Goel, Wobbrock y Patel (UIST 2012) a la muñeca para medir el tacto.<br>
Cuando el reloj vibra brevemente, cuanto más fuerte agarras, más vibración absorbe la mano: baja la alta frecuencia del giroscopio y sube el temblor de 4–12 Hz de los dedos que sostienen.

<p align="center">
  <img width="560" alt="Cómo la vibración del reloj lee la fuerza de agarre" src="https://github.com/user-attachments/assets/cf02f9f1-a91b-472a-8073-cc072cd7790a" />
</p>

**Medido en un Galaxy Watch4** (IMU de hasta 100 Hz, 3 grabaciones de calibración personal, entrenar en la ronda 1 → evaluar en la 2, y validación cruzada al revés)

- **Clasificación:** un bosque aleatorio (25 árboles) + suavizado de 2 s llega a una precisión del **59%** en 4 niveles (soltar · suave · normal · fuerte) (azar 25%, fuerte 93%)
- **Regresión:** en los tramos en que el video detecta contacto mano-objeto, un bosque de regresión (50 árboles) estima la fuerza de agarre como puntuación continua 0–100 (p. ej. «medio (57)»)<br>
  → dividido en bajo · medio · alto acierta el **83%**, error absoluto medio (MAE) de 15 puntos
- La frecuencia cardíaca, los intervalos entre latidos y la temperatura de la piel del reloj vienen del Samsung Health Sensor SDK (`docs/samsung_health_sdk.md`).

<p align="center">
  <img src="docs/grip_quality.es.png" width="85%" alt="Calibración de agarre una vez al inicio → modelo personal → agarre 0–100 en cada grabación; en la ronda no vista 97% suave como bajo, 56% normal como medio, 95% fuerte como alto">
</p>

### 3. Sincronización de los cinco sentidos

Los datos de los dispositivos se sincronizan con la ida y vuelta del reloj teléfono↔reloj, una baliza acústica de 18–20 kHz (sonido de alta frecuencia) y pulsos de vibración del reloj.

- De 37 grabaciones reales con teléfono + reloj, **24 se alinearon con la baliza acústica** (residuo por par 0,1–1,9 ms).
- El resto usa el reloj vía servidor, pero como este se desvía 24,7 ms (±3,5) de la baliza, ese sesgo se mide por par de dispositivos y se resta.
- Resultado: error de tiempo reloj ↔ video **24,7 → 3,7 ms** (umbral de venta 10 ms)

<p align="center">
  <img src="docs/sync_quality.es.png" width="85%" alt="Desfase reloj–video: 24,7 ms antes de corregir, hasta 16,7 ms con palmada, 3,7 ms ahora (umbral de venta 10 ms)">
</p>

### 4. Mayor precisión de etiquetado con Spark 1.3

Para mejorar la precisión de las etiquetas por tramo y por acción:

1. El video se divide en fragmentos de 20 s a 0,5× y se **etiqueta 5 veces y se compara** (se combinan los resultados para elegir un solo orden)
2. **Reverificación con Spark Open 816B** de pesos abiertos (vuelve a elegir los fotogramas límite donde cambia la acción)
3. **Verificación con etiquetas humanas**
4. **Generación propia de etiquetas por misión con Spark**

En 8 clips no vistos de EPIC-KITCHENS-100 (271 pasos de referencia humana), la precisión de etiquetado pasa de **36% → 54% (+18 pp)** y el error de límite de **430 → 182 ms**.

> También consideramos la reverificación con Muse Wonderful, pero como no está disponible en Corea usamos un modelo de pesos abiertos.

### 5. Ahorro de API con Glimmer

Para no depender siempre de la API de Muse Spark para etiquetar, entrenamos el modelo de pesos abiertos Muse Glimmer 30B con nuestros datos (LoRA en Fireworks).<br>
Spark crea un borrador de etiquetas, y Glimmer aprende tomando como referencia los pasos que una persona confirmó y corrigió.

- Antes de entrenar **41,8%** → entrenado con etiquetas confirmadas **54,1%** → añadiendo decodificación con el orden de acciones por misión **64,1%**
- A medida que se acumulen datos, Glimmer podrá asumir las decisiones, reducir las llamadas a la API de Spark y mejorar el margen del negocio.

### 6. Menor costo de detección de objetos con SAM

- Las grabaciones inútiles (muy cortas, manos no visibles, muy oscuras) se rechazan en este Mac **antes de llamar a la API**.
- Un detector de objetos en este Mac revisa primero, y SAM solo recibe lo necesario.<br>
  (sin consulta "persona" si se graba a solas, pantallas solo cuando aparecen, objetos estáticos a 1 fotograma por segundo)
- En 4 grabaciones reales: costo de SAM **$2,92 → $0,27 (−91%)**, unos $5–7 por hora de video (precio de venta ₩60.000/hora (~43 US$))
- Decodificación de máscaras de SAM reescrita con Numba: tiempo de procesamiento **593 s → 8 s**

---

## Datos de venta LeRobot (LeRobot v3.0)

<p align="center">
  <img width="720" alt="Vista LeRobot v3.0 en la web para empresas compradoras" src="https://github.com/user-attachments/assets/8131397c-a152-4891-8100-d7066f90d2fa" />
</p>

| Elemento | Contenido |
|---|---|
| Video | `observation.images.ego` (video ocultado y con rostro sustituido) |
| Articulaciones | `observation.hands` · `observation.body` · `observation.face` |
| Movimiento | `observation.imu.phone` · `observation.imu.wrist` |
| Sonido | `observation.audio` |
| Tacto | **`observation.touch`** (contacto · agarre bajo · medio · alto + puntuación 0–100) · `observation.touch.vibration` |
| Fisiología | **`observation.physio`** (frecuencia cardíaca · temperatura de la piel, solo con consentimiento biométrico) |
| Objetos · etiquetas | `observation.objects` · `annotation.step` · `annotation.primitive` (reach·grasp·…·release) |
| Sincronización | `sync.error_ms` + certificado de calidad |

Se carga con `lerobot` 0.6.1 real (`scripts/validate_lerobot.py`).

Los sensores se compran por paquetes (por hora, `datacash/pricing.py`).

| Video (obligatorio) | Articulaciones | Sonido | IMU de muñeca | Tacto | Fisiología |
|---|---|---|---|---|---|
| ₩35.000 | ₩20.000 | ₩5.000 | ₩15.000 | ₩25.000 | ₩50.000 |

---

## Arquitectura

**Flujo general** — la empresa pide en palabras, se abren misiones, los participantes graban con teléfono y reloj y suben, el servidor procesa, entrega los datos y paga la recompensa.

<p align="center"><img src="docs/architecture.en.svg" width="95%" alt="Arquitectura general"></p>

**Cómo se procesa una grabación** (`datacash/jobs.py`) — primero los pasos sin coste, así las grabaciones inservibles se rechazan antes de llamar a un modelo de Meta.

<p align="center"><img src="docs/pipeline.en.svg" width="95%" alt="Orden de procesamiento"></p>

**Tecnologías y despliegue** — hoy no hay Docker ni nube: un solo Mac se encarga del servidor y de todo el procesamiento, y las apps del teléfono y del reloj lo encuentran por mDNS en la misma Wi-Fi y suben directamente.

<p align="center"><img src="docs/tech_stack.en.svg" width="95%" alt="Tecnologías"></p>

## Ejecutar

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" torch pyyaml rich
echo "MODEL_API_KEY=..." > .env                       # nunca se sube al repositorio
# en data/models: MediaPipe hand/pose/face landmarker .task, efficientdet_lite0.tflite, OpenCV yunet.onnx·sface.onnx
.venv/bin/python -m pytest                            # verificado con sesiones sintéticas de cinco sentidos de resultado conocido
DATACASH_TOKEN=... .venv/bin/python -m datacash.server.app --port 8800   # servidor + web para empresas compradoras
cd app && ./gradlew assembleDebug                     # app de Android + app de reloj
```

## Carpetas

| Carpeta | Qué hace |
|---|---|
| `app/` | App de Android (consentimiento · misiones · verificación facial · grabación · marcar errores · preguntas · subida · recompensas · apelaciones) + app de reloj |
| `web/` | Web para empresas compradoras (peticiones en lenguaje natural · línea de tiempo de cinco sentidos · presupuestos · pedidos · descargas), consola del operador |
| `datacash/privacy` | Ocultación con SAM · sustitución de rostro · privacidad del audio · revisión previa |
| `datacash/sync` · `touch` · `labels` | Sincronización de cinco sentidos · tacto · articulaciones de manos y cuerpo |
| `datacash/understand` | Comprensión con Muse Spark · segunda verificación con Spark Open · verificación con sensores |
| `datacash/export` · `jobs.py` | Exportación LeRobot · certificado · orden de procesamiento |
| `datacash/server` | Servidor de recolección + API para empresas compradoras |

## Uso de los modelos de Meta

| Modelo | Uso |
|---|---|
| **SAM 3.1** | Oculta personas sin consentimiento, niños, pantallas, espejos, cuadros y documentos; reconoce los objetos manipulados y las acciones |
| **Muse Spark 1.3** | Video ocultado → etiquetado de pasos, instrucciones, mano activa, objetos y errores<br>Petición en palabras de la empresa → diseño de misiones → generación de etiquetas por misión |
| **Muse Image** | Genera 70 personas virtuales que no existen (para sustituir rostros), con 24 fotos en ángulo de la misma persona para cada una |
| **Muse Voice Transcribe** | Silencia los tramos de la grabación en que se oye información personal |
| **Muse Spark Open** | El "modelo superior" que da la segunda verificación de cada grabación y revisa las apelaciones.<br>Los pasos en que los dos modelos difieren se corrigen; los pasos en que coinciden y que los sensores verifican se vuelven ejemplos de referencia que se añaden a las consultas de Spark 1.3 de las siguientes grabaciones. |
| **Muse Glimmer 30B** | Aprende reconocimiento de video con los borradores de Spark 1.3 y las etiquetas verificadas por personas.<br>Cuantos más datos se reúnen, más mejora el rendimiento y menos se depende de la API. |

## Rendimiento medido

| Qué | Antes → ahora | Medido en |
|---|---|---|
| Precisión por fotograma de etiquetas de paso (Muse Spark 1.3) | **35,6% → 54,3%** | EPIC-KITCHENS-100, 8 clips nuevos, 271 pasos de referencia humana |
| Error de límite de paso (mediana) | **430 → 182 ms** | igual |
| Precisión de pasos de Muse Glimmer 30B (LoRA, Fireworks) | **41,8% → 54,1% → 64,1%**<br>(v80 entrenado con referencias humanas · v82 + decodificación del orden por misión, rango 60,7–67,2% · v79, entrenado con etiquetas de Spark, 32,8%) | 122 pasos de referencia humana en 8 grabaciones no usadas para entrenar |
| Error de tiempo reloj ↔ video | **24,7 → 3,7 ms** (umbral 10 ms) | 37 grabaciones con teléfono + reloj |
| Tiempo de sustitución de rostro por grabación | **636 → 84 s** (7,5×) | `docs/privacy_eval.md` |
| Coste de SAM 3.1, 4 grabaciones | **$2,92 → $0,27** (−91%) | 4 grabaciones reales |
| Fuerza de agarre bajo · medio · alto | **83%** (azar 33%) | 3 grabaciones de calibración, Galaxy Watch4 |

<p align="center">
  <img src="docs/perf/label_accuracy.en.png" width="85%" alt="Precisión de etiquetas de paso"><br><br>
  <img src="docs/perf/confusion.en.png" width="85%" alt="Matriz de confusión"><br><br>
  <img src="docs/perf/glimmer.en.png" width="85%" alt="Entrenamiento de Glimmer"><br><br>
  <img src="docs/perf/monitoring.en.png" width="85%" alt="Monitorización del servicio">
</p>

**Monitorización del servicio** (los mismos valores que la consola del operador, `datacash/perf.py`) — si se supera un umbral se emite una alerta y esa grabación pasa a revisión.

- Concordancia Spark 1.3 ↔ Spark Open por grabación y media de las últimas 5 — alerta si baja del 50%
- Tiempo de procesamiento por grabación — alerta si supera el doble de la mediana
- Fotogramas que tras la sustitución de rostro siguen pareciendo la misma persona (SFace 0.363) — alerta si superan el 5%

---

## Antes y durante el hackathon

| | Contenido |
|---|---|
| Antes del hackathon (hasta el 26 de octubre) | Solo el documento de planificación y la imagen de marca. Sin código |
| Durante el hackathon (desde el 27 de octubre, 01:00 KST) | Todo el código, la app, la web, las personas virtuales de Muse Image y los documentos de este repositorio |

## Licencias y fuentes

MediaPipe (Apache 2.0) · OpenCV YuNet y SFace (Apache 2.0) · LivePortrait (MIT) · LeRobot (Apache 2.0) · Numba (BSD)<br>
Los videos de prueba son de Wikimedia Commons (solo CC BY, CC BY-SA o dominio público; fuentes en `docs/privacy_eval.md`)<br>
Samsung Health Sensor SDK
