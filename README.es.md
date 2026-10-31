[English](README.md) · [한국어](README.ko.md) · **Español** · [中文](README.zh-CN.md) · [日本語](README.ja.md)

# DataCash

**Datos de manos humanas para robots, grabados por personas comunes con el teléfono y el reloj que ya tienen.**

Las personas graban en casa misiones cortas de habilidades para robots (sacar un pañuelo, mover una taza, girar una tapa — 12 en total) con su teléfono y su reloj inteligente, y reciben un pago.
Las empresas de humanoides e IA física compran el resultado: datos de **cinco sentidos — vista, oído, tacto (fuerza de agarre), movimiento y fisiología (frecuencia cardíaca)** sin datos personales, sincronizados en un mismo reloj
y entregados como un conjunto de datos **LeRobot v3.0** listo para entrenar.

Proyecto para el Meta Global AI Developer Hackathon (Indie Developer · Push the Frontier).

![DataCash](docs/1026_datacash_brand.png)

## En resumen

| | |
|---|---|
| Participantes | App de Android + app de reloj. Elige una misión → verificación facial → graba → 1–2 preguntas → *ding*, recibes tu pago |
| Compradores | **Piden en lenguaje natural** desde la web ("datos de levantar cajas para robots de almacén, 20–50 años, presupuesto 3.000 USD") → la IA diseña las misiones y las abre en todo el mundo |
| Privacidad | El rostro del participante se reemplaza por **una persona que no existe**; se ocultan transeúntes, pantallas y documentos; los originales se borran tras el procesamiento |
| Tacto | Sin guante — **el reloj vibra y mide cuánto absorbe la mano** para estimar la fuerza de agarre (GripSense, UIST 2012) |

## Dónde se usan los modelos de Meta

| Modelo | Función | Código |
|---|---|---|
| **SAM 3.1** | Oculta personas sin consentimiento, niños, pantallas, espejos, cuadros y documentos (seguimiento en video); segmenta el objeto manipulado → `observation.objects` | `datacash/meta/sam.py`, `datacash/privacy/mask.py` |
| **Muse Spark 1.3** | Video ocultado → pasos, instrucciones, mano activa, objetos, errores (JSON estructurado); petición en lenguaje natural → diseño de misiones | `datacash/understand/`, `datacash/requests.py` |
| **Muse Image** | 34 personas que no existen (7 regiones × mujeres/hombres × 20–60 años) para reemplazar rostros | `scripts/make_donors.py`, `datacash/donors/` |
| **Muse Voice Transcribe** | Detecta información personal hablada en la grabación y la silencia | `datacash/privacy/speech.py` |
| **Muse Spark Open** (pesos abiertos, 816B) | Segunda opinión sobre cada grabación y revisor de apelaciones (Baseten, con Fireworks de respaldo). Si los dos modelos difieren, corrige el paso; los pasos en que coinciden y que los sensores verifican se vuelven ejemplos que se añaden a las siguientes consultas de Spark 1.3 — las etiquetas mejoran con el uso sin cambiar pesos | `datacash/understand/spark_open.py`, `datacash/evolve.py` ([docs/model_evolution.md](docs/model_evolution.md)) |

Cada llamada a la API se registra con su costo en `data/api_ledger.jsonl`; si se alcanza el presupuesto, solo se omite ese paso.

## Ideas clave

### 1. Privacidad que conserva el valor de los datos
- **Reemplazo de rostro, no desenfoque:** el rostro del participante se cambia por una persona de Muse Image (LivePortrait + MediaPipe). Elegimos a la más parecida del mismo género y edad similar, probamos 20 fotogramas y pasamos a la siguiente si el reconocimiento facial (SFace) consideraría "la misma persona" a más de la mitad.
- **Continuidad:** si un fotograma reemplazado sale demasiado parecido, se regenera con una expresión más suave en lugar de pixelarlo — el video nunca se corta.
- **Ocultación:** SAM 3.1 oculta solo lo que no tiene consentimiento: otras personas, niños, pantallas, espejos, cuadros y documentos. Los familiares que consienten siguen visibles, cada uno con su propio rostro virtual.
- **Originales borrados:** en producción, el video y el audio originales se borran al terminar el procesamiento (`purge_originals`). En el servidor de demostración del hackathon solo esa llamada está desactivada, para poder mostrar la comparación original vs. reemplazado.

### 2. Cinco sentidos, sincronizados automáticamente
Sin aplaudir: ida y vuelta del reloj teléfono↔reloj, una baliza acústica de 18–20 kHz y pulsos de vibración del reloj. En 37 grabaciones reales con teléfono + reloj, la baliza acústica alinea 24 (residuo por par 0,1–1,9 ms); el resto usa el reloj vía servidor, que se desvía 24,7 ms (±3,5) de la baliza, así que ese sesgo se mide por par de dispositivos y se resta (máx. 9,2 ms dejando una fuera). Mediciones en `docs/sync_eval.md`, código en `datacash/sync/`.

![Desfase reloj–vídeo: 24,7 ms antes de corregir, hasta 16,7 ms con palmada, 3,7 ms ahora (umbral de venta 10 ms)](docs/sync_quality.es.png)

### 3. Tacto sin guante — GripSense en la muñeca
Trasladamos *GripSense* de Goel, Wobbrock y Patel (UIST 2012) a la muñeca. El reloj vibra brevemente; cuanto más fuerte agarras, más vibración absorbe la mano — baja la energía de alta frecuencia del giroscopio y sube el temblor de 4–12 Hz de los dedos. Un árbol de decisión por persona, entrenado con niveles autoevaluados (soltar · suave · normal · fuerte, sin báscula), da el nivel de agarre 0–3 (`datacash/touch/`). Medido en un Galaxy Watch4 (IMU limitado a 100 Hz, 1 calibración, entrenar en la ronda 1 → probar en la 2 y al revés): un bosque de 25 árboles con suavizado de 2 s llega al 59% en 4 niveles (azar 25%, fuerte 100%); mientras el vídeo ve la mano tocando un objeto, un bosque de regresión de 50 árboles da el agarre como puntuación 0–100 (p. ej. «medio (57)»); cortado en bajo · medio · alto acierta el 88%, error medio de 15 puntos.
La frecuencia cardíaca, los intervalos entre latidos y la temperatura de la piel vienen del Samsung Health Sensor SDK (`docs/samsung_health_sdk.md`).

### 4. Peticiones en lenguaje natural → misiones
El comprador escribe lo que necesita; Muse Spark (salida estructurada) diseña las misiones (acciones, objetos, condiciones a variar), los requisitos (edad, género) y los paquetes de sensores, y el servidor ajusta un presupuesto. Al confirmarse, las misiones solo aparecen para los participantes que cumplen los requisitos — la petición *es* la distribución de la recolección.

### 5. Costos de procesamiento que dejan margen
- Las grabaciones inútiles (muy cortas, manos no visibles, muy oscuras) se rechazan **en local antes de cualquier llamada a la API** — 0 USD
- Un detector de objetos local revisa primero; SAM solo recibe lo necesario (sin consulta "person" en grabaciones individuales, pantallas solo si aparecen, objetos estáticos a 1 fps)
- En 4 grabaciones reales: costo de SAM **2,92 → 0,27 USD (−91 %)**, unos 5–7 USD por hora de video (se vende a ~43 USD/hora)
- Decodificación de máscaras de SAM reescrita con Numba: 593 s → 8 s por grabación, idéntica bit a bit al analizador oficial

## El conjunto de datos (LeRobot v3.0)

`observation.images.ego` (video ocultado + rostro reemplazado) · `observation.hands` · `observation.body` · `observation.imu.phone` · `observation.imu.wrist` ·
`observation.audio` · **`observation.touch`** (contacto, agarre bajo · medio · alto + puntuación 0–100) · `observation.touch.vibration` · **`observation.physio`** (frecuencia cardíaca, temperatura de la piel; solo con consentimiento biométrico) ·
`observation.objects` · `annotation.step` · `annotation.primitive` (reach · grasp · … · release) · `sync.error_ms` + un certificado de calidad.
Se carga con `lerobot` 0.6.1 real (`scripts/validate_lerobot.py`).

Los compradores eligen paquetes de sensores por hora de datos — video (obligatorio) ₩35.000 · pose ₩20.000 · audio ₩5.000 · IMU de muñeca ₩15.000 · tacto ₩25.000 · biometría ₩50.000 (`datacash/pricing.py`).

## Ejecutar

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" torch pyyaml rich
echo "MODEL_API_KEY=..." > .env                       # nunca se sube al repositorio
# data/models: MediaPipe hand/pose/face landmarker .task, efficientdet_lite0.tflite, OpenCV yunet.onnx + sface.onnx
.venv/bin/python -m pytest                            # verificado con sesiones sintéticas de resultado conocido
DATACASH_TOKEN=... .venv/bin/python -m datacash.server.app --port 8800   # servidor + web para compradores
cd app && ./gradlew assembleDebug                     # app de Android + app de reloj
```

LivePortrait no se incluye — descarga [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) (MIT) y apunta `LP_ROOT` a esa carpeta.

## Estructura

| Carpeta | Qué hace |
|---|---|
| `app/` | App de Android (consentimiento, misiones, verificación facial, grabación, marcar errores, preguntas, subida, pagos, apelaciones) + app de reloj |
| `web/` | Web para compradores (peticiones en lenguaje natural, línea de tiempo de cinco sentidos, presupuestos, pedidos, descargas) y consola del operador |
| `datacash/privacy` | Ocultación con SAM · reemplazo de rostro · privacidad del habla · revisión previa |
| `datacash/sync` · `touch` · `labels` | Sincronización · tacto · pose de manos y cuerpo |
| `datacash/understand` | Comprensión con Muse Spark · segunda opinión con Spark Open · verificación con sensores |
| `datacash/export` · `jobs.py` | Exportación LeRobot · certificado · flujo de procesamiento |
| `datacash/server` | Servidor de recolección + API para compradores |

## Documentos (en coreano)

- Plan del hackathon: [docs/1026_meta_hackathon.md](docs/1026_meta_hackathon.md) · Registro del día 1: [docs/DAY1.md](docs/DAY1.md)
- Evaluación de ocultación y reemplazo de rostro: [docs/privacy_eval.md](docs/privacy_eval.md) · Verificación de modelos de Meta: [docs/model_check.md](docs/model_check.md)
- SDK del reloj: [docs/samsung_health_sdk.md](docs/samsung_health_sdk.md) · Guion del video demo: [docs/demo_video_script.md](docs/demo_video_script.md)

## Antes y durante el hackathon

| | Contenido |
|---|---|
| Antes (hasta el 26 de octubre) | Solo el documento de planificación y la imagen de marca. Sin código |
| Durante (desde el 27 de octubre, 01:00 KST) | Todo el código, la app, la web, las personas de Muse Image y los documentos de este repositorio |

## Licencias y fuentes

MediaPipe (Apache 2.0) · OpenCV YuNet y SFace (Apache 2.0) · LivePortrait (MIT) · LeRobot (Apache 2.0) · Numba (BSD).
Los videos de prueba son de Wikimedia Commons (solo CC BY, CC BY-SA o dominio público; fuentes en `docs/privacy_eval.md`) y no se suben al repositorio.
El Samsung Health Sensor SDK no se sube al repositorio, según sus términos.
