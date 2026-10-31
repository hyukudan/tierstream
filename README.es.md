# tierstream

[English](README.md) · [Español](README.es.md)

*Antes strix-stream.* Streaming de pesos todo en uno entre niveles de memoria, para modelos MoE y densos, en CUDA y HIP.

Una caché de pesos para motores de inferencia cuyo modelo no cabe en la memoria de la GPU. Mantiene en la GPU los pesos
que importan ahora, guarda más en RAM del host bloqueada en página (pinned) y lee el resto desde NVMe. Mueve *unidades*
de pesos (un experto, o el blob completo de una capa) entre estos niveles mientras el motor se ejecuta. Los motores leen
cada unidad a través de una tabla de direcciones, de modo que el lugar donde reside una unidad nunca cambia ni un solo
bit de la salida.

- ABI de C: `include/tierstream/ts.h` (prefijo `ts_`). Los nombres de strix-stream (`include/strix_stream/ss.h`, `ss_*`,
  el target de CMake `strix_stream` y el archivo de la biblioteca, `ss-recipe` / `ss-plan`, las opciones `SS_*` y `SS_TEST_DIR`)
  siguen funcionando sin cambios durante una versión (consulte Nombres más abajo).
- Diseño y contrato con el motor: [docs/design.md](docs/design.md). Recetas: [docs/recipes.md](docs/recipes.md).
- Backends: CPU (pruebas), CUDA (RTX PRO 6000, 4090), HIP para Strix Halo (memoria unificada; consulte Estado).

## Cómo funciona

### Tres niveles

```
     engine kernels ── read unit u of layer L at table[L][u] ──┐
                                                               v
  ┌─ 1. GPU memory (VRAM) ──────────────────────────────────────────────────────────┐
  │  resident slots (the plan's units) │ dynamic slots (adaptive) │ ring (temporary)│
  └─────────────────────────────────────────────────────────────────────────────────┘
        ^ promote: copy RAM -> VRAM          │ evict
        │                                    v   inclusive: drop it (RAM or the file still has it)
        │                                        exclusive: write it back VRAM -> RAM into a free block
  ┌─ 2. pinned host RAM ────────────────────────────────────────────────────────────┐
  │  unit blocks │ staging buffers │ read-ahead buffers (optional)                  │
  └─────────────────────────────────────────────────────────────────────────────────┘
        ^ on a miss: read the file into a staging buffer, then copy it to VRAM
        │ ahead of time (opt.): read-ahead fills its buffers before the GPU slot is free
  ┌─ 3. model files on NVMe ────────────────────────────────────────────────────────┐
  │  GGUF / safetensors / packed records; unbuffered (direct I/O) by default,       │
  │  buffered when the file system refuses it                                       │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

- **Ring.** Un pequeño conjunto de slots temporales de GPU, que usan los fallos que no obtienen un slot persistente y las
  precargas.
- **RAM inclusiva frente a exclusiva.** En modo inclusivo (el predeterminado), una unidad puede estar en RAM y en VRAM a
  la vez. En modo exclusivo, la RAM contiene las unidades que no están en un slot persistente de GPU; durante un forward
  una unidad puede estar brevemente en ambas (véase más abajo).
- **Memoria unificada (Strix Halo).** Los niveles 1 y 2 son la misma memoria, y no hay un nivel pinned separado. Una
  unidad residente es memoria del host mapeada, y un fallo se lee desde NVMe a su slot. Las lecturas con búfer llegan
  directamente al slot. Con E/S directa, el desplazamiento en el archivo, la longitud y la dirección de destino de cada
  parte deben estar alineados; en caso contrario, la lectura pasa por un búfer de staging y un memcpy.

Un **descriptor** enumera las unidades del modelo: sus archivos, desplazamientos, tamaños y partes (por ejemplo, las
matrices gate, up y down de un experto). Un **plan** asigna a cada unidad un nivel inicial a partir de un presupuesto de
memoria:
- Primero reserva los bytes fijos del descriptor, el espacio de trabajo del motor, el margen, los slots dinámicos y el ring.
- Después llena el resto de la VRAM:
  - con `order = ranked`, las unidades con mayor beneficio por byte (por defecto todas las unidades valen lo mismo; una
    frecuencia de enrutamiento medida lo hace más inteligente);
  - con `order = prefix`, el prefijo más largo de unidades, en el orden del modelo, que quepa (las unidades se recorren
    capa a capa, así que con un blob por capa esto es un prefijo de capas completas).
- Después la RAM pinned; el resto permanece en disco.

`ts-plan` imprime un plan; `ss_plan_from_tiers` acepta uno propio.

### Un forward, desde el lado del motor

`ss_begin` abre un forward y `ss_end` lo cierra. Para cada capa entre ambos:
1. El motor indica qué unidades necesita: `ss_layer_routes` (los ids de expertos del router), o `ss_layer_all` para una
   capa densa.
2. `ss_acquire` le entrega la tabla de la capa una vez que esas unidades están en la GPU.
3. `ss_release` indica que los kernels han terminado con ellas.

Las unidades ya residentes no necesitan ninguna transferencia de pesos. Las demás son *fallos*, que se copian desde la
RAM pinned o se leen desde disco a un slot libre. `ss_acquire_groups` permite al motor empezar con las unidades
residentes mientras los fallos aún están llegando ("aciertos primero").

### Decode con compuerta en el dispositivo (decode MoE capturado en grafo)

En el modo normal, el host gestiona las rutas de cada capa y proporciona su tabla de direcciones. El decode con
compuerta en el dispositivo (`forward_mode = SS_FORWARD_DEVICE_GATED`) permite al motor capturar un token completo en un
grafo CUDA: una secuencia grabada de operaciones de GPU. La GPU exporta las rutas y espera a los pesos que faltan. Un
hilo de servicio del host sigue procesando cada capa, pero una capa cuyos pesos ya están disponibles no necesita
respuesta del host.

```text
 engine's CUDA graph                         tierstream service thread (host)
 -------------------                         --------------------------------
 begin
 router for layer g
 export(g, ids) ---- mapped route record ----> validate ids; serve any misses
       |                                      demand copies and slot-table writes
       |                                      run on the demand copy stream
       |
 wait(g) <---------- ready flag -------------- written on that same stream,
       |                                      after the required copies
       |  No misses: wait returns without a host reply.
       v
 experts read slot = ss_dg_slot(&view, g, id)
 part address = part base + slot x part stride

 optional engine prediction for layer g+1:
 predict(g, ids) --- mapped prediction record -> prefetch on a separate copy stream,
                                                within slot and prefetch limits
 ... next layer ...

 end -------------- completion word ---------> ss_end waits for device and service;
                                                maintains the cache and folds stats
```

- **En el dispositivo:** `slot_of` asigna a cada id de experto un slot. Cada parte de pesos tiene su propio array. Los
  kernels la leen en `part base + slot x part stride`.
- **En el host:** el servicio lee un registro por cada capa desde memoria del host mapeada, a la que pueden acceder
  tanto la CPU como la GPU. Valida los ids y actualiza los contadores de uso y la recencia. Para los fallos, organiza
  copias desde la RAM pinned y publica las entradas de la tabla de slots. El stream de copias bajo demanda escribe el
  indicador de listo después de las copias y actualizaciones de tabla necesarias. El motor también puede proporcionar
  ids previstos para la capa siguiente. El servicio los precarga dentro de su reserva de slots y de los límites por capa
  y por token.
- **Gestión de fallos:** un tiempo de espera agotado o un id enrutado no válido activa un indicador de aborto
  persistente. Toda búsqueda de pesos debe usar `ss_dg_slot` (`ts_dg_slot` en `tierstream/ts.h`); tras un aborto,
  devuelve un slot de guarda lleno de ceros. `ss_end` devuelve un error y envenena el contexto. El motor debe confirmar
  su KV, su estado arrastrado (carry) y su estado especulativo solo después de un `ss_end` correcto. En caso de fallo,
  descarte las salidas del token y, una vez que el trabajo pendiente ya no pueda escribir el estado del modelo, restaure
  el último estado confirmado o termine la sesión. Reabrir solo la caché no repara el estado escrito por el paso
  fallido. Los lectores de pesos de cada capa deben terminar antes de la exportación de la capa siguiente. El kernel
  final debe ir después de todos los lectores y de las ramas unidas.
  - Cada `ss_begin` autoriza exactamente un paso con compuerta. El kernel de inicio del paso avanza la secuencia del
    dispositivo y la comprueba contra la autorización del host; una discrepancia aborta el paso, de modo que las
    búsquedas devuelven el slot de guarda. Un paso extraviado que siga en cola cuando se ejecuta `ss_begin` puede
    consumir esa autorización: `ss_end` puede tener éxito para ese token, y la desincronización se notifica en un
    `ss_begin` o `ss_end` posterior. No lance nunca pasos con compuerta adicionales fuera del par, y sincronice el stream
    de cualquier calentamiento previo antes de empezar un par.
  - Tras la espera de la capa, `ss_dg_slot` devuelve el slot de guarda ante una entrada de tabla ausente inesperada. Si
    la vista incluye un campo `absent_err` no nulo, la búsqueda escribe esa palabra; entonces `ss_end` hace fallar el
    token y envenena el contexto. Las vistas antiguas, más pequeñas, devuelven la guarda sin notificar la entrada
    ausente.
  - Antes de un paso con compuerta en modo eager, cargue todo kernel que pueda lanzarse por primera vez mientras un
    kernel de espera está girando. De lo contrario, la carga diferida de CUDA puede bloquear ese lanzamiento hasta el
    plazo de la espera. Establezca `CUDA_MODULE_LOADING=EAGER` antes de la inicialización de CUDA, o llame a
    `cudaFuncGetAttributes` sobre cada uno de esos kernels antes de lanzar el paso. Cualquier token completo de
    calentamiento con compuerta debe usar su propio par `ss_begin`/`ss_end` y terminar antes del par siguiente. Los
    grafos capturados cargan sus kernels durante la instanciación.
- **Presupuestos y estadísticas:** usa el planificador existente y después comprueba la disposición con compuerta al
  abrir. Toda unidad de pesos debe caber en la memoria de la GPU o en la RAM pinned; el decode no trae pesos desde disco.
  Las comprobaciones incluyen las tablas y controles del dispositivo, los registros mapeados, los búferes de staging y
  suficientes bloques libres de reserva en el host. Las API de estadísticas existentes siguen disponibles. Los contadores
  se actualizan tras las llamadas a `ss_end` correctas; las instantáneas exponen la última publicación. El reinicio de
  estadísticas solo se permite entre tokens.
- **Arranque en caliente y límites:** `dg_seed` proporciona entradas `(gated ordinal, unit index)` por orden de
  prioridad, de la más reciente a la más antigua. Al abrir, los slots residentes se llenan con entradas semilla
  distintas, después con los residentes planificados y después con las unidades restantes en orden de clave, hasta el
  número de slots persistentes menos el objetivo de reserva. Las entradas semilla fuera de rango se rechazan; mantenga
  el array válido durante `ss_open`.
  - `dg_spare_slots` fija un único número global de slots de reserva; cero usa los recuentos por capa del plan sumados
    en todas las capas con compuerta. `dg_spare_target` es el objetivo de slots libres en la frontera, como máximo el
    número de reserva contabilizado; su valor predeterminado es ese número completo. Entre tokens, `ss_dg_set_budgets`
    cambia el objetivo para la frontera siguiente y los límites de precarga para el token siguiente. La reserva de
    precarga predeterminada sigue al objetivo; una reserva explícita permanece fija. Los write-backs de la frontera están
    limitados por los bloques libres disponibles de RAM pinned, de modo que las víctimas restantes siguen residentes y
    puede que no se alcance el objetivo.
- **Planificación de write-backs (activada por defecto):** una frontera que copia de inmediato a la RAM sus expertos
  desalojados hace que las primeras copias bajo demanda del token siguiente se encolen detrás de ellas en una GPU cuyo
  único motor de copia sirve ambas direcciones. Con `dg_wb_defer = 1` (el valor predeterminado; 0 restablece las copias
  inmediatas) la frontera solo elige las víctimas y las despublica. Después, el hilo de servicio las copia
  de vuelta parte a parte durante el token siguiente, justo después de la señal de listo y las precargas de una capa,
  cuando el motor de copia está por lo demás inactivo. Nunca copia después de la última capa con compuerta, y se detiene
  en cuanto el dispositivo ha exportado la capa siguiente. `dg_wb_layer_bytes` fija el presupuesto por capa (valor
  predeterminado: una parte).
  `dg_wb_max_outstanding` limita los bytes en curso (valor predeterminado: dos expertos). `dg_wb_stream` elige el
  stream de write-back (predeterminado: las copias se solapan con las copias de demanda de la capa siguiente) o el
  stream de demanda (0: se encolan delante de ellas). Una víctima que vuelve a ser enrutada o prevista antes de
  que se copie ninguno de sus bytes no cuesta ninguna copia. Una demanda que no encuentra un slot libre espera a un
  write-back ya en curso o, si no, termina el diferido más antiguo. Los write-backs sin terminar pasan al token
  siguiente, hasta aproximadamente las víctimas de una frontera, y la frontera siguiente copia el exceso más antiguo. Los
  resultados son idénticos bit a bit. `ss_stats_dg` informa de los recuentos y bytes diferidos, cancelados, forzados y
  vaciados en la frontera.
- **Entre peticiones:** `ss_dg_set_sched` activa o desactiva el aplazamiento de write-backs y el vaciado de listos entre
  tokens, de modo que un motor puede elegir la planificación por petición; las salidas siguen siendo idénticas bit a
  bit. `ss_dg_reset_open` restaura el estado de caché resuelto en el momento de apertura. `ss_dg_residency` toma una
  instantánea de las unidades residentes en orden de recencia para el `dg_seed` de una carga posterior. Estas API y el
  cargador de arranque en pipeline están integrados en [PR #1](https://github.com/hyukudan/tierstream/pull/1).
  El aplazamiento de write-backs está activado por defecto a través de `ss_options_defaults` desde
  [PR #2](https://github.com/hyukudan/tierstream/pull/2); una estructura de opciones inicializada a cero sigue
  seleccionando copias inmediatas.
- **Prefill de capa completa (capas mantenidas residentes):** `ss_dg_make_resident(c, g, ids, n)` hace que unidades de
  la capa con compuerta g sean residentes y se mantengan en la misma caché que usa el decode (una copia de cada experto:
  slot de la arena, RAM pinned, archivo), entre tokens y de forma síncrona, desalojando expertos no mantenidos según la
  política con las reglas de write-back habituales; devuelve `SS_E_CAPACITY` sin cambiar nada cuando las unidades
  mantenidas de todas las capas más el objetivo de reserva no caben en los slots persistentes. Una capa retenida nunca
  se exporta: un paso con `rows_per_layer[h] = 0` para todas las demás capas lee los expertos de g a través de
  `ss_dg_slot`, y después `ss_dg_keep(c, g, NULL, 0)` la libera. Una capa con compuerta con 0 filas simplemente no se
  exporta en ese paso (no hay exportación, predicción ni espera para ella); el servicio la omite y sus contadores e
  historial de puntuaciones quedan como estaban. Sus copias solo cuentan en `ss_dg_stats.resident_*`. Consulte
  docs/design.md §3.9.
- **Rondas de colocación y fallos sin admisión:** `ss_dg_round(c, &params, &result)`, entre tokens, mueve los expertos
  hacia el rango de un recuento de exportaciones con decaimiento (`s = decay x s + exports since the last round`,
  contado en el host): los expertos pinned más calientes sustituyen a los más fríos de la GPU y, con tres niveles, los
  expertos de archivo más calientes sustituyen a los pinned más fríos, solo cuando el recuento del recién llegado supera
  `hysteresis` (valor predeterminado 1.25) x el del sustituido, con un máximo de `max_units` (valor predeterminado 96)
  promociones y `budget_bytes` bytes por ronda. Los expertos mantenidos y retenidos nunca se mueven. Cada movimiento es
  transaccional como `ss_dg_make_resident` (un fallo de lectura es `SS_E_IO` con la caché coherente; un fallo del
  dispositivo envenena); `ss_dg_round_cancel` (desde cualquier hilo) detiene una ronda en el experto siguiente, y
  `ss_dg_round_seed_counts` inicia los recuentos a partir de un perfil de enrutamiento. Con `dg_no_admit = 1` un fallo
  se sirve desde un slot del banco y se libera en la frontera sin desalojar nada ni cambiar la ubicación base del
  experto, de modo que la colocación solo cambia mediante rondas (y
  `ss_dg_make_resident` / `ss_seed`). Los resultados son idénticos bit a bit; `ss_dg_stats.round_*` cuenta las rondas.
  Consulte docs/design.md §3.10.
- **Desalojo por puntuación (de activación explícita):** con `policy = SS_POLICY_SCORED` la frontera sigue manteniendo
  libre el mismo número de slots, pero elige los expertos con menos probabilidad de necesitarse pronto en lugar de los
  usados menos recientemente. Cada experto recibe una puntuación entera de tres componentes: con qué frecuencia, en este
  contexto, los expertos enrutados en el token anterior fueron seguidos por este (dados los expertos enrutados en el
  token actual), con qué frecuencia apareció en los últimos 16 tokens de la capa y lo recientemente que fue enrutado.
  Los pesos predeterminados son 700 / 200 / 100 sobre 1000
  (`dg_score_w_trans`, `dg_score_w_freq`, `dg_score_w_recency`). El hilo de servicio ordena cada capa justo después de
  servirla, de modo que la frontera solo fusiona las listas ordenadas. Los recuentos empiezan vacíos y aprenden en
  línea, o parten de un perfil de rutas grabadas (`dg_score_profile`, ligado al modelo por sus hashes por unidad y
  rechazado ante cualquier discrepancia). El historial dura tanto como el contexto; `ss_dg_reset_open` lo devuelve a su
  estado de apertura. Los resultados son idénticos bit a bit; solo cambia qué expertos permanecen en la GPU. En la traza
  de rutas de DeepSeek usada para la planificación (8,576 expertos residentes y 96 de reserva) reduce los fallos de 8.14
  a 7.65 por token solo con aprendizaje en línea.
  `ss_stats_dg` informa de las víctimas tomadas en orden de puntuación y del tiempo de ordenación.

Es de activación explícita y solo para CUDA, para decode. Requiere colocación exclusiva y desalojo LRU o por
puntuación, con `pin_planned = 0`.
Antes de que un motor lo convierta en su opción predeterminada, debe superar las comprobaciones de corrección bit a bit
y de rendimiento frente a la propia caché de ese motor, y después recibir la aprobación del mantenedor.

### Promoción y desalojo, en ambas direcciones

- **Promoción (RAM -> VRAM, disco -> VRAM).** Un fallo se copia a un slot.
  - Con `pin_planned = true` (prefill), los residentes planificados nunca se mueven; solo se adaptan los slots
    dinámicos.
  - Con `pin_planned = false` (decode), los residentes planificados son solo un arranque en caliente, y el conjunto
    residente sigue lo que el router elige realmente.
- **Desalojo (VRAM -> RAM, o a ninguna parte).** Cuando se necesita un slot, la caché elige una víctima:
  - **LRU** (predeterminado) desaloja la unidad usada menos recientemente que este forward no esté usando. En nuestras
    trazas de decode de DeepSeek acertó en el 95-98 % de las búsquedas, frente al 84-93 % de una colocación fija del
    mismo tamaño (design.md §3.1).
  - **SCORED** ordena las unidades por una combinación de transiciones de enrutamiento (qué expertos siguen a cuáles),
    frecuencia reciente y antigüedad. Es de activación explícita hasta que supere a LRU en trazas reales.
- **Inclusivo frente a exclusivo** (`tier_mode`) determina lo que cuesta un desalojo.
  - **Inclusivo** (predeterminado): una unidad promovida desde la RAM pinned conserva su copia en RAM, así que el
    desalojo solo descarta la copia en VRAM. Es sencillo, pero la misma unidad puede ocupar VRAM y RAM a la vez.
  - **Exclusivo**: en cada frontera de forward, cada unidad reside exactamente en uno de tres lugares: un slot de VRAM,
    un bloque de RAM o solo su archivo. (Durante una transferencia está brevemente en dos; los archivos siempre se
    mantienen como almacenamiento de respaldo.)
    - **Promoción.** Cuando un fallo se promueve a un slot persistente, su bloque de RAM se libera en `ss_end`. Un fallo
      servido desde el ring conserva su bloque.
    - **Desalojo.** La víctima se escribe de vuelta desde la VRAM a un bloque libre de RAM de su tamaño: una copia de
      dispositivo a host en su propio stream. En una GPU con un solo motor de copia (la RTX PRO 6000 de aquí informa de
      uno), esa copia comparte el motor con las copias de los fallos, así que no es gratuita. Sin bloque libre, el fallo
      se sirve desde el ring y la víctima se queda. Con los slots llenos y sin bloques de reserva, el conjunto residente
      deja de adaptarse, pero nada recurre al disco.
    - **Capacidad.** VRAM + RAM pueden entonces contener el modelo completo cuando, tras los bytes fijos, el espacio de
      trabajo, el ring y la alineación, su capacidad en unidades lo cubre. Ejemplo de diseño (design.md §3.5): para la
      réplica de decode de DeepSeek V4.1 en RTX, 80 GiB de slots de VRAM más ~63 GiB de bloques de RAM contendrían los
      142.4 GiB de expertos, sin ninguna lectura de archivo tras la apertura. El presupuesto real de la receta pendiente
      publicada aún deja 5.8 GiB en NVMe.

### Traer por adelantado

- **Precarga cierta** (`ss_prefetch_layer`): el prefill capa a capa sabe que necesita la capa L+1 a continuación, así
  que esa capa se copia mientras la capa L calcula (parámetro de motor de la receta `prefetch_layers`).
- **Lookahead especulativo** (`ss_prefetch` con `SS_PREFETCH_SPECULATIVE`):
  - Para decode MoE, el motor ejecuta el router de la capa L+1 sobre la entrada de la capa L, y `ss_lookahead_select`
    convierte las puntuaciones en los pocos expertos con más probabilidad de ser los siguientes (`lookahead_units`,
    valor predeterminado 2).
  - Se leen con prioridad baja, solo mientras no haya ninguna demanda esperando.
  - Por defecto llegan al ring y nunca desalojan a un residente por una conjetura (`spec_persistent = true` les deja
    entrar).
- **Lectura anticipada en el host** (`readahead_units`; el motor llama a `ss_host_prefetch_layer`, parámetro de motor
  de la receta `host_prefetch_layers`):
  - El disco es el nivel más lento, y una lectura de disco normalmente no puede empezar hasta que haya un slot de GPU
    libre.
  - La lectura anticipada lee pronto las unidades en disco de las capas siguientes a búferes de RAM pinned, de modo que
    cuando el slot se libera solo queda la copia RAM -> VRAM.
  - Antes de que existiera, el prefill de MiniMax en RTX a 8K esperaba ~0.45 s por capa en disco (design.md §3.6).
- **Semilla** (`ss_seed`): entre peticiones, carga las unidades favoritas de un perfil de uso en slots libres.

Los parámetros de motor (`prefetch_layers`, `host_prefetch_layers`, …) solo tienen efecto en un motor que los
implemente.

### Varias unidades de disco (streaming multidisco)

Un modelo leído en streaming desde una sola unidad NVMe está limitado por esa unidad. tierstream puede leer el mismo
modelo desde varias unidades a la vez, sin una segunda copia completa:
- `ts-stripe make MODEL.gguf RANGES.json STRIPE_DIR` copia parte del modelo a otra unidad como un
  **stripe**: fragmentos completos (16 MiB) en una cuadrícula fija, listados en un manifiesto con el SHA-256 del modelo y
  el xxh64 de cada fragmento. Un archivo de rangos indica qué bytes colocar; `interleave` pone en el stripe uno de cada
  dos fragmentos de un rango, de modo que un stream de expertos de capa completa o un lote de filas aleatorias de Engram
  se lee aproximadamente a medias de cada unidad. Para DeepSeek V4.1 Flash Q2, los expertos enrutados y ambas tablas de
  Engram intercaladas = 166 GiB en la segunda unidad.
- Los lectores pasan los directorios de stripe (`weight_dirs`, en los motores `--weight-dirs MODEL_DIR;STRIPE_DIR`).
  Las sesiones comparten **un pool de lectores por volumen y número de lectores** entre archivos (`drive_readers`, 1..32,
  valor predeterminado 4); las lecturas sin búfer siguen sin búfer. Las peticiones grandes (8 MiB o más) se reparten
  entre las unidades con **cuotas adaptativas** que siguen la velocidad medida de cada unidad (una unidad lenta recibe
  poco o nada, y una de cada 16 peticiones grandes lee solo del original como referencia en vivo); las lecturas pequeñas
  y limitadas por latencia se quedan en la unidad original, como con una sola unidad.
- **Verificación y confianza.** Un fragmento de stripe solo se sirve después de que su xxh64 haya coincidido con el
  manifiesto (`weight_verify`: `always` comprueba cada fragmento al abrir y vuelve a comprobar en línea las lecturas de
  fragmento completo; `cached` comprueba una vez por versión del archivo de datos). Los errores de lectura, las lecturas
  incompletas y las discrepancias recurren al archivo original; los stripes ausentes u obsoletos desactivan esa
  ubicación. Los directorios de stripe deben ser tan fiables como el primario: xxh64 detecta la corrupción accidental, no
  la manipulación deliberada. Un `primary_sha256` proporcionado se acepta como fiable sin comprobarlo y debe describir el
  primario actual. Windows mantiene los archivos en servicio solo con `FILE_SHARE_READ`, lo que excluye a los escritores.
  Linux no tiene una exclusión de solo lectura compartida: los datos del stripe se publican con modo 0444, pero su
  propietario o un proceso privilegiado aún pueden modificarlos. La identidad, incluido el ctime de Linux, se comprueba
  al reutilizar y tras la verificación. Mantenga inmutables los archivos primarios y de stripe mientras haya sesiones
  activas. Use `weight_verify=always` en Linux cuando los directorios de stripe no sean de propiedad exclusiva; las
  lecturas parciales siguen dependiendo de la verificación en la apertura. La identidad de bytes presupone estos
  requisitos de confianza e inmutabilidad.
- **Primera apertura.** Sin un `primary_sha256` admitido ni un acierto en la caché de identidad, abrir un primario con
  stripes calcula el hash de todo el primario y examina el stripe. Cientos de GiB pueden tardar minutos; las aperturas
  posteriores con caché evitan estos exámenes. La API de C `ss_mfile` permite lecturas y consultas concurrentes con
  búferes de destino separados. El cierre no debe competir con ninguna llamada que use ese manejador; los manejadores
  separados de una sesión compartida pueden cerrarse de forma independiente.
- Sin directorio de stripe (el valor predeterminado) no cambia nada: se ejecuta la ruta de código original de un solo
  archivo.

**Medido** (DeepSeek V4.1 Flash Q2 en la RTX PRO 6000 durante el prefill del prompt, perfil exacto, nivel pinned de
32 GiB en el host; el original en una NVMe PCIe 4.0, el stripe en una segunda NVMe PCIe 4.0; los mismos tokens y estados
en todas las variantes):

| prompt | una unidad | dos unidades | |
|---:|---:|---:|---:|
| 8K | 13.2 s | 10.6 s | -19 % |
| 32K | 19.2 s | 17.4 s | -9 % |
| 64K | 29.1 s | 27.8 s | -5 % |
| 128K | 49.9 s | 48.4 s | -3 % |

El propio stream de expertos pasa de 11.0 a 8.45 s por petición (-23 %, 29 % de los bytes desde la segunda unidad) y el
servicio arranca un 17 % más rápido; los prompts más largos ganan menos porque están limitados por el cálculo. Una
tercera unidad más lenta (un disco de sistema sin DRAM) no aportó nada: las cuotas le asignan poco. El decode lee
expertos sueltos y filas de Engram, que se quedan en la unidad original, así que la velocidad de decode no cambia.

### Packs preparados (un pack por modelo, para todas las disposiciones)

Los motores convierten algunos pesos en el momento de la carga (un rango de paralelismo de tensores conserva su mitad de
las filas de cada experto, un bloque cuantizado dividido, una disposición reempaquetada). Un **pack** almacena esos
bytes una sola vez, en el formato final de dispositivo de cada motor, de modo que una carga es una simple lectura
directamente a su sitio. Un pack por modelo sirve para todas las disposiciones:
- un rango de paralelismo de expertos lee los expertos completos de su grupo;
- un rango de paralelismo de tensores lee su intervalo de filas de cada experto en una sola lectura (o su propio
  registro por rango cuando un formato no se puede dividir por filas);
- la caché con compuerta en el dispositivo lee las unidades por clave: `ss_pack_dg_descriptor` convierte el descriptor
  del motor en uno cuyas partes de expertos apuntan al pack.

Un pack es un directorio: `pack.json` (el modelo, el SHA-256 de cada archivo fuente, cómo se produjo cada formato, el
SHA-256 de cada archivo de datos y del índice), `pack.index` (un registro por tensor o parte de experto: clave, formato,
desplazamiento, bytes, filas, xxh64) y `pack-NNNN.data` (registros alineados a 4 KiB, las partes de un experto
contiguas). Fije el SHA-256 de `pack.json` y quedará fijado todo lo demás; los registros se comprueban contra su xxh64
la primera vez que se leen (`cached`, predeterminado), en cada lectura (`always`) o nunca tras `ts-pack verify --full`
(`off`). Un pack construido a partir de archivos fuente distintos de los que carga la aplicación se rechaza. Los
archivos de datos del pack son archivos normales para `ts-stripe`, así que un pack puede repartirse entre varias
unidades como un archivo de modelo.

```sh
# The engine describes where its device bytes come from (tierstream/pack-layout-v1: byte ranges of the model file,
# optionally row selections and byte rearrangements inside rows); the tool prints the sizes first.
ts-pack build LAYOUT.json /models/x.pack --dry-run
ts-pack build LAYOUT.json /models/x.pack            # prints the pin: the SHA-256 of pack.json
ts-pack verify /models/x.pack --pin SHA256 --full   # every file's SHA-256 and every record's xxh64
ts-pack show /models/x.pack                         # sizes per format, files, sources
```

Un motor cuyo formato de dispositivo no sea una reordenación de bytes del archivo del modelo escribe el pack por sí mismo
mediante la API de escritura (`ss_pack_writer_begin` / `ss_pack_writer_add` / `ss_pack_writer_finish`: el pack solo se
hace visible cuando está completo). Los motores abren un pack con `ss_pack_open` y leen con `ss_pack_find` +
`ss_pack_readv`; sin un pack coincidente (o con el interruptor del motor desactivado) mantienen su cargador original.
Formato y API: [docs/pack.md](docs/pack.md).

### Lo que nunca cambia

Los bytes se copian, nunca se transforman, y los kernels del motor no deben depender del slot en el que se encuentra una
unidad. Por eso toda configuración de colocación, política y precarga produce salidas idénticas bit a bit. Con
`verify_hash` activado y hashes del descriptor proporcionados, los bytes de las unidades cargados desde archivos se
comprueban antes de aceptarse, y una discrepancia es un error.

### Una caché para todos los motores

Todos los motores de este laboratorio (DeepSeek, MiniMax, Qwen, GLM, en la tarjeta RTX y en los nodos Strix Halo) leen
sus pesos en streaming a través de tierstream en lugar de mantener su propia caché de expertos. La lógica de residencia,
precarga, predicción y write-back reside aquí una sola vez:
- los motores describen sus unidades en un descriptor;
- las recetas eligen los presupuestos;
- el código del lado del motor se reduce a un adaptador que nombra las unidades que necesita y las lee a través de las
  tablas.

### Presupuestos de dispositivo completo

`--vram-total-gib 24` significa "ejecutar como si esta fuera una tarjeta de 24 GB": un límite para todo lo que hay en el
dispositivo y otro para el host (`--ram-total-gib`, incluida la memoria pinned). `ss_budget_plan` resta el contexto CUDA
medido y las partes fijas con nombre del motor (tronco, KV para el contexto solicitado, espacios de trabajo, drafter,
búferes del host pinned y paginables, staging de carga); la arena de expertos recibe lo que queda del dispositivo, el
nivel pinned lo que queda del host y el nivel de archivo el resto. Cuando las partes fijas por sí solas no caben, se
niega a arrancar indicando cuáles son y el límite más pequeño que funcionaría. Sin límite, produce exactamente la
colocación en tres niveles actual. `ss_budget_ledger` (pasado como `ss_options.budget`) imputa cada asignación que hace
tierstream contra los mismos límites y conserva el pico por tipo con los componentes en ese pico, junto al pico que
midió el motor (cudaMemGetInfo / NVML). Detalles: [diseño §5.1](docs/design.md).

### Ejecución en su hardware (perfiles de dispositivo)

Al arrancar, tierstream puede describir la máquina en la que se ejecuta como un perfil de dispositivo
(`strix/device-profile-v1`): la RAM del host, si la GPU la comparte (Strix Halo y sus sucesores: un único pool de
memoria), la arquitectura, las unidades de cálculo y la memoria de cada GPU, los enlaces USB4 con una segunda caja y las
unidades NVMe. Lee el sysfs de Linux (topología KFD, totales de memoria de amdgpu, pares Thunderbolt), el runtime de HIP
o, en NVIDIA, el driver de CUDA y NVML (consultas de solo lectura, sin asignaciones). Una clase de dispositivo
(`profiles/*.json`: `strix-halo-395`, `rtx-pro-6000-blackwell`, `rtx-4090`, genérica) añade los márgenes de seguridad y
los valores medidos que la detección no puede conocer; una máquina sin clase (una gfx nueva, otra tarjeta) se ejecuta
con sus valores detectados y los márgenes actuales, con un aviso, y nunca recibe los valores ajustados de otra
arquitectura.

Lo que controla el perfil cuando no se indica ningún flag explícito:
- **Reparto GPU / RAM / disco**: `ss_devprof_budget` fija los límites de dispositivo completo (tarjeta discreta: VRAM -
  reserva y RAM disponible - suelo; unificada: un pool compartido por la arena de expertos y el nivel pinned);
  `ss_budget_plan` coloca después los expertos en el dispositivo, en RAM pinned y en disco como antes.
- **Una caja o dos**: `ss_devprof_topology` se ejecuta en una sola caja cuando las partes fijas del modelo y un nivel de
  expertos mínimo caben en el presupuesto de esta caja con el contexto solicitado; en caso contrario, entre las cajas con
  un enlace medido, repartiendo las capas según la proporción del presupuesto. Una receta cualificada puede fijar su
  elección.
- **Tamaño de pieza de prefill**: `ss_devprof_piece_rows` dimensiona la pieza a partir de la memoria que queda tras el
  plan, dentro de los límites propios del motor.

```
ts-devprof                                  # the effective profile (JSON) of this machine
ts-devprof --budget                         # the caps it gives the planner
ts-devprof --device-profile my-box.json     # override: a full or partial profile merged over the detected one
STRIX_DEVICE_PROFILE=my-box.json <engine>   # the same override for any engine that reads profiles
```

Los valores explícitos `--vram-total-gib` / `--ram-total-gib` (o los números de una receta) siempre prevalecen sobre el
perfil. Una sobrescritura parcial basta para corregir un valor; por ejemplo,
`{"margins": {"host_floor_bytes": 17179869184}}` mantiene 16 GiB libres en el host. Cada justificante puede llevar el
hash del perfil (`ss_devprof_hash`) y las rutas de las sobrescrituras fusionadas, de modo que un resultado indica qué
valores de la máquina lo determinaron. Detalles: [diseño §5.2](docs/design.md).

## Cuándo compensa (y cuándo no)

La cuestión es cuántos bytes de pesos debe mover cada token a través de PCIe, en comparación con lo que la GPU lee de su
propia memoria, que es mucho más rápida.
- En una RTX PRO 6000 (96 GB, Windows WDDM), los microbenchmarks de RAM pinned ↔ VRAM midieron unos 22 GB/s
  con un PC ocupado y 34-40 GB/s con un PC más tranquilo: el 40 % y el 62-73 %, respectivamente, de la referencia
  práctica de PCIe del laboratorio de 55 GB/s (no la velocidad teórica del enlace). El benchmark de comparación sin
  cambios muestra el mismo efecto, así que la pérdida dependiente de la carga no es específica de tierstream. Se
  sospecha de la contención de la memoria del host, pero no se ha aislado.
  El dispositivo expone un solo motor de copia; los write-backs en cola pueden retrasar las copias bajo demanda entre
  streams.
- NVMe es aún más lento.

| carga de trabajo | streaming | por qué |
|---|---|---|
| **Decode MoE** (DeepSeek, MiniMax, Qwen MoE, GLM) | puede compensar, con un conjunto caliente | Un token usa solo los top-k expertos de cada capa (6 de 384 en DeepSeek V4.1). Cuando el enrutamiento tiene un conjunto caliente, como en nuestras trazas de DeepSeek (95-98 % de aciertos LRU), la mayoría de las búsquedas no necesitan transferencia y un fallo cuesta una copia de experto. |
| **Prefill y lotes grandes** (densos o MoE) | puede compensar | Muchos tokens reutilizan cada byte de pesos, así que la copia de la capa L+1 puede ocultarse tras el cálculo de la capa L, siempre que ese cálculo dure más que la copia. |
| **Modelos que no caben en absoluto** | permite ejecutarlos | La alternativa es no ejecutarlos, o repartirlos entre máquinas. Es una ganancia de capacidad, no de velocidad. El modo exclusivo elimina las lecturas de disco solo cuando VRAM + RAM cubren las unidades. |
| **Decode denso** (uno o pocos tokens por paso) | **no** compensa | Cada token lee todos los pesos. Lo que no es residente cruza PCIe en cada token, mucho más lento que la VRAM, así que la fracción en streaming domina el tiempo del paso. Use en su lugar una cuantización más pequeña, más GPU o una máquina de memoria unificada. |
| **Decode MoE con enrutamiento casi uniforme** | compensa menos | Sin conjunto caliente, la tasa de fallos se aproxima a la fracción de expertos que no son residentes. |
| **Caché KV, adaptadores** (más adelante) | todavía no | Los mismos niveles sirven para otros rangos de bytes grandes y de uso desigual. Hoy las unidades son solo pesos. |

### Memoria adaptativa y mediciones de NVMe (en desarrollo)

La rama `adaptive` trata la memoria de expertos y el KV/estado como un único presupuesto de dispositivo: reserva estado
para el contexto seleccionado y después coloca los expertos entre VRAM, RAM pinned y NVMe. Una tabla
`tierstream/usage-v1` ordena los expertos por uso medido, recencia con decaimiento y orden estable de unidades. Incluye
un actualizador de uso, un simulador de curvas de fallos y un resolvedor de escenarios hipotéticos; todavía no se
publican. El nivel de archivo genérico ya existe; el soporte de NVMe con compuerta en el dispositivo es un desarrollo
aparte.

Medido en una RTX PRO 6000 (96 GB, Windows WDDM): reproducción de rutas de decode de código reservadas de DeepSeek V4.1
Flash, usando expertos GGUF reales (IQ2_XXS gate/up, Q2_K down) y un stub de cálculo dimensionado para un suelo de
6.18 ms con todo aciertos.
Las etiquetas de contexto de abajo son **presupuestos de memoria**, no benchmarks de longitud de contexto del modelo
completo. Los bytes de expertos muestreados coincidieron bit a bit (924 comprobaciones, sin fallos); esto no valida los
logits del modelo completo ni los tokens generados.

| Presupuesto de contexto / asignación de estado | Expertos: VRAM / pinned / NVMe | ms/token | Rendimiento como % del techo del stub | Fallos pinned / NVMe por token |
|---|---:|---:|---:|---:|
| 8K, capacidad máxima de expertos | 8,594 / 6,766 / 0 | 9.06 | 68.2 % | 8.39 / 0 |
| 128K, asignación de estado ajustada | 7,713 / 7,215 / 432 | 10.43 | 59.2 % | 11.71 / 0.036 |
| 128K, búferes de estado actuales | 5,113 / 7,233 / 3,014 | 32.7 | 18.9 % | 78.5 / 4.5 |

La lectura anticipada en el host no ayudó en esta reproducción de decode: las admisiones en slots persistentes volvían a
leer el archivo en lugar de consumir los búferes de lectura anticipada ya listos. PR #2 corrige esa ruta del modo host y
añade `exclusive_discard`, de activación explícita, para cuando no hay ningún bloque libre en el host. No se ha medido
una aceleración de la precarga en decode con la corrección.

Los pools retenidos de copias en el host no justificaron su coste de memoria en la simulación de trazas en CPU. La
compresión sin pérdidas tampoco aportó una ganancia útil: 298 expertos muestreados de las 40 capas MoE de DeepSeek V4.1
Flash, con los mismos bytes IQ2_XXS/Q2_K, se comprimieron solo 1.008× con zstd niveles 1-19 y se recuperaron bit a bit.
No se adopta ninguno de los dos enfoques.

## Parámetros de receta de un vistazo

Una receta (véase más abajo) se compila en el presupuesto y las opciones de la biblioteca. Los parámetros que dan forma
al streaming:

| parámetro | valor predeterminado | qué hace |
|---|---|---|
| `device_gib` | el `memory_min_gib` del conjunto de dispositivos | el presupuesto de VRAM |
| `host_pinned_gib` | 0 | el presupuesto de RAM pinned |
| `workspace_gib` | derivado del descriptor y de los límites de la ruta (se rechaza si no se puede derivar) | la VRAM propia del motor |
| `units`: `resident` / `pinned` / `rest` | `"fill"` / 0 / `"disk"` | cuántas unidades de cada grupo empiezan en VRAM, en RAM y en disco |
| `pin_planned` | true | true: los residentes planificados nunca se mueven (prefill). false: son un arranque en caliente para LRU / SCORED (decode). |
| `dynamic_slots_per_layer` | 0 | slots de VRAM adicionales por capa para el conjunto adaptativo |
| `ring_slots` | 0 (auto) | slots de VRAM temporales para los fallos no admitidos y para las precargas |
| `policy` | `lru` | desalojo `lru` o `scored` |
| `tier_mode`, `host_spare_units` | `inclusive`, 0 (auto) | inclusivo o exclusivo; bloques libres de RAM solicitados para los write-backs |
| `lookahead_units`, `spec_persistent` | 2, false | expertos especulativos por capa; si una conjetura puede ocupar un slot persistente |
| `readahead_units`, `host_prefetch_layers` | 0, 0 | búferes de lectura anticipada en el host y distancia |
| `prefetch_layers` | 1 | distancia de precarga cierta de capa completa |
| `reader_threads`, `staging_mib`, `direct_io` | 2, 256, true | el lector de disco |
| `weight_dirs`, `drive_readers`, `weight_verify` | ninguno, 4, cached | copias de stripe en otras unidades (varias unidades de disco) |

La lista completa, con unidades y comprobaciones, está en [docs/recipes.md](docs/recipes.md).

## Compilación

```
# Linux / WSL
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build   # -DTS_CUDA=ON, -DTS_HIP=ON
# Windows (MSVC + Ninja), CPU or CUDA
tools\build-win.bat          &  tools\build-win.bat cuda
```

Las pruebas escriben pequeños archivos de modelo sintéticos en `TS_TEST_DIR` (o `SS_TEST_DIR`), o en el directorio
temporal del sistema si no está definido. La
prueba CUDA `ts-test-cuda` necesita una GPU. Calcula la suma de comprobación de cada unidad transmitida en el
dispositivo y mide el ancho de banda pinned y de NVMe con unos 256 MiB cada uno.

`ts-plan DESC.json --device-gib X --host-gib Y` valida un descriptor e imprime su plan de niveles, en el mismo formato
que lee el planificador de hyperloom.

Para recompilar, volver a ejecutar las compuertas de validación y reproducir los resultados medidos de DeepSeek,
consulte [docs/reproduce.md](docs/reproduce.md).

## Recetas

Una receta es un pequeño archivo TOML (o JSON) editable por personas que indica cómo se ejecuta un modelo en un conjunto
de hardware, para cada tipo de petición. El esquema completo, con cada campo, unidad y valor predeterminado, está en
[docs/recipes.md](docs/recipes.md).

Una receta tiene cuatro partes:

- **match**: el modelo y el hardware al que se destina: conjuntos de dispositivos, memoria, RAM del host, ancho de banda
  de disco y de PCIe, y los enlaces entre conjuntos.
- **routes**: reglas ordenadas que eligen qué dispositivos atienden una petición, a partir de su fase (prefill o
  decode), la longitud del prompt y el lote.
- **placement**: para cada ruta y dispositivo, el presupuesto de memoria, los parámetros de streaming y qué unidades de
  pesos son residentes en el dispositivo, cuáles vienen de la memoria del host pinned, cuáles se leen desde NVMe y
  cuáles calcula otro dispositivo.
- **límites y alternativas**: qué ocurre cuando una ruta no puede ejecutarse.

Un ejemplo breve (un modelo que hace el prefill de prompts largos en una tarjeta discreta y el decode en un par de dos
nodos):

```toml
schema = "strix-stream/recipe-v1"
name = "example-lab"

[match]
model = "example-moe"

[[match.devices]]
id = "card"
kind = "cuda"
memory_min_gib = 90.0

[[match.devices]]
id = "pair"                       # two unified-memory nodes run as one set
kind = "hip"
count = 2
memory_min_gib = 100.0
internal_link = { kind = "usb4" }

[[match.links]]
id = "card-pair"
between = ["card", "pair"]

[[routes]]
id = "prefill_long"               # long prompts: prefill on the card, hand the state to the pair
phase = "prefill"
devices = ["card"]
execute = "single"
handoff_route = "decode"
links = ["card-pair"]
fallback = "prefill_pair"         # card busy, link down or low RAM: the pair prefills
when = { prompt_tokens_min = 4096 }
x-evidence = "the measured crossover goes here"

[routes.placement.card]
device_gib = 90.0
host_pinned_gib = 48.0
workspace_gib = 4.0
max_context_tokens = 32768
max_batch = 1

[[routes.placement.card.units]]   # 60 layer blobs: 32 resident, 16 pinned, 12 from NVMe
id = "layers"
class = "blob"
resident = 32
pinned = 16
rest = "disk"

[[routes]]
id = "prefill_pair"
phase = "prefill"
devices = ["pair"]
execute = "tp"
handoff_route = "decode"

[routes.placement.pair]
workspace_gib = 2.0
max_context_tokens = 32768
max_batch = 4

[[routes]]
id = "decode"
phase = "decode"
devices = ["pair"]
execute = "tp"

[routes.placement.pair]
workspace_gib = 2.0
max_context_tokens = 32768
max_batch = 8
```

**Rutas y alternativas.** Las rutas se prueban en el orden del archivo y se toma la primera cuya condición `when` se
cumpla, si puede ser admitida: sus dispositivos están disponibles, sus enlaces están activos, la petición cabe en sus
límites de contexto y de lote, y cada host que usa conserva su suelo de RAM libre tras la memoria compilada de la ruta.
En caso contrario, se prueba su `fallback` y después las rutas siguientes. Cada fase termina con una ruta comodín. Una
petición en ejecución nunca cambia de ruta.

**Colocación.** Los grupos de unidades ordenan las unidades (orden del modelo, una lista fija o frecuencia medida) y
llenan los niveles en orden: primero los recuentos enteros, después `"fill"` (tantas como admita el presupuesto del
dispositivo o del host, con la misma regla que el planificador) y después el resto. El resultado es una colocación
`strix-stream/plan-v1` normal, de modo que un motor la abre como cualquier otro plan; una colocación que no cabe se
rechaza, nunca se reduce. Las rutas heterogéneas reparten las unidades de un modelo entre dispositivos, con exactamente
un propietario por unidad.

**Autoajuste y sobrescrituras.** Tres capas forman la receta efectiva: la receta, un perfil del autoajustador y las
sobrescrituras humanas (archivos o `--set path=value`), con mayor prioridad la última. Un perfil está ligado a los bytes
de la receta, al motor, al modelo, a los dispositivos, a sus drivers y a los descriptores, que deben seguir coincidiendo
todos, y a los anchos de banda sondeados, que pueden desviarse hasta `max_drift` (valor predeterminado 10 %). En caso
contrario, el perfil se rechaza y la receta se ejecuta sin ajustar. Una receta puede fijar campos que el autoajustador
no puede cambiar. `ts-recipe explain` muestra, para
cada campo, su valor, qué capa lo estableció y qué sustituyó.

**La exactitud no es una opción.** Toda colocación que una receta pueda expresar produce salidas idénticas bit a bit;
una receta que pida algo "exact" se rechaza.

```
ts-recipe validate RECIPE                 # strict check, prints the effective recipe
ts-recipe explain  RECIPE --set routes.prefill_long.when.prompt_tokens_min=3072
ts-recipe match    RECIPE PROBE.json      # does this hardware fit? which devices go where
ts-recipe select   RECIPE STATE.json      # which route admits this request, and why not the others
ts-recipe plan     RECIPE --route R --device-set S --desc DESC.json   # budget, options and plan-v1
```

Las mismas llamadas están en el ABI de C (`ss_recipe_*`).

Las **recetas de ejemplo** están en [recipes/examples/](recipes/examples/README.md), calibradas sobre las configuraciones
medidas del laboratorio (dos nodos Strix Halo unidos por USB4 y una RTX PRO 6000 (96 GB, Windows WDDM)):

- validadas por `ts-recipe` (la admisión en una máquina concreta aún puede rechazarlas): DeepSeek V4.1 Flash (par, y la
  colocación heterogénea medida RTX + par), MiniMax M3.1 Flash, Qwen3.8 Flash Next;
- `pending/`: recetas que esperan una ejecución del motor, un descriptor o hardware: una réplica de decode de DeepSeek en
  RTX, DeepSeek y MiniMax en una tarjeta de 32 GB, MiniMax con >= 192 GB de RAM del host, GLM 5.3 Flash.

Cada número que contienen lleva su evidencia en claves `x-` (`x-evidence`, `x-cites`, `x-predicted` con una base:
medido, derivado, proyectado, cota u objetivo). CTest valida cada archivo con `ts-recipe` y comprueba que los fixtures
negativos estructurales de `recipes/examples/tests/` se rechazan; `validate_v1.py` añade las comprobaciones de citas
(necesita las notas del laboratorio a las que apuntan los `ref`).

## Nombres

strix-stream pasó a llamarse tierstream. En esta versión el cambio es puramente aditivo:
- **ABI de C.** `ts.h` añade los nombres `ts_*` por encima del `ss.h` sin cambios. Los tipos son los mismos tipos y cada
  función `ts_*` llama a su función `ss_*`, de modo que el código antiguo y el nuevo enlazan con una misma biblioteca.
- **Compilación y herramientas.** El target de CMake es `tierstream` (alias `strix_stream`). El archivo de la biblioteca
  se genera con ambos nombres. Las herramientas son `ts-recipe` / `ts-plan`, y también se compilan `ss-recipe` /
  `ss-plan`, cada una imprimiendo su propio nombre. Las opciones son `TS_CUDA` / `TS_HIP` / `TS_TESTS`, y un valor
  `SS_*` sigue funcionando.
- **Formatos.** Todos los analizadores aceptan `tierstream/<format>` y `strix-stream/<format>`. Todas las salidas siguen
  escribiendo `strix-stream/<format>` en esta versión, de modo que no cambia ningún hash de receta, plan ni documento de
  estadísticas. Los archivos de receta no cambian.

Los nombres antiguos se eliminarán en una versión posterior, cuando todos los consumidores hayan migrado.

## Estado

v1. Las pruebas de CPU pasan en MSVC y en GCC con ASan, UBSan y TSan. El backend CUDA está compilado y su prueba de GPU
se ejecuta en una ventana corta. El modo exclusivo y la lectura anticipada en el host son de activación explícita.

El backend HIP para Strix Halo (memoria unificada) se compila con la toolchain ROCm de los nodos (`-DTS_HIP=ON`) y su
prueba de GPU pasa en un nodo Strix Halo (gfx1151). `ts-test-hip` es la prueba de GPU de CUDA compilada para HIP. Las
lecturas de decode, prefill y lectura anticipada se verifican correctamente en el dispositivo, y un fallo de lectura se
notifica y se recupera. La prueba también mide el rendimiento de llegada a la memoria del host mapeada para la GPU: unos
29 GB/s desde el nivel pinned y unos 6 GB/s desde un archivo sin búfer. El modo exclusivo requiere un dispositivo
discreto en esta implementación y el backend HIP de memoria unificada lo rechaza.

El decode con compuerta en el dispositivo es de activación explícita y solo para CUDA. Sus pruebas de GPU cubren el
decode capturado en grafo, la predicción, la presión de memoria, los fallos de protocolo, los write-backs diferidos, la
carga en pipeline, el reinicio, la siembra y el desalojo por puntuación. La excepción del motor de copia único y las
compuertas de validación necesarias están documentadas en [docs/reproduce.md](docs/reproduce.md). La comparación bit a
bit del modelo completo de DeepSeek frente a la propia caché de la réplica se superó, y también la paridad de velocidad
frente a esa caché (chat no más lento, A/B/A); tierstream es ahora la única caché de esa réplica.

Medido con DeepSeek V4.1 Flash en una RTX PRO 6000 (96 GB), Windows WDDM, modo con compuerta en el
dispositivo, expertos Q2 GGUF (IQ2_XXS gate/up, Q2_K down; densos Q8_0), caché exclusiva VRAM + host pinned.

| Funcionalidad integrada | Comparación medida | Exactitud / condiciones |
|---|---|---|
| Carga en pipeline (activada por defecto) | Carga completa: 108.7 → 61.8 s; llenado de expertos: 83.1 → 38.0 s; 2.25 → 6.73 GB/s | Serie frente a pipeline con 8 lectores; limitada por disco; colocación y salidas idénticas bit a bit. `DG_LOAD_SERIAL=1` restablece la carga en serie. |
| Reinicio al estado de apertura | Unos 3.4 s para restaurar el estado de caché de una carga nueva | Las peticiones tras el reinicio fueron idénticas bit a bit a las de un proceso nuevo. |
| Instantánea / semilla de residencia | Paso 0 en proceso nuevo: 332.6 → 156.0 ms; media de los pasos 0–15: 56.05 → 44.67 ms/step | Carga nueva sembrada con el conjunto residente de una petición anterior frente a carga nueva de referencia; salidas idénticas bit a bit. |

La precarga por ventana (`winpf`) está integrada, es exacta y está desactivada por defecto. La evaluación está cerrada:
las comparaciones en caliente emparejadas no mostraron ninguna ganancia repetible. Los workers persistentes cargados y
la precaptura/calentamiento de grafos están en desarrollo en el motor (réplica); tierstream proporciona las API de
reinicio de caché, planificación y residencia.

## Créditos

El diseño de esta biblioteca se inspiró en [mlx-stream](https://github.com/davidtai/mlx-stream) de David Tai, un
plugin de DeepSeek-V4.1-Flash para mlx-serve que transmite expertos desde SSD. Su enfoque del streaming de pesos MoE
(residencia por puntuación, lookahead del router de la capa siguiente, lecturas de expertos sin búfer a slots del
dispositivo, compuerta de aciertos primero) es el punto de partida de los mecanismos de este proyecto; gracias a su
autor por publicarlo.

No se incluye código fuente de terceros: todos los archivos de este repositorio son código nuevo escrito para este
proyecto (consulte NOTICE). Si alguna vez se añade un archivo portado, conservará su aviso de licencia original en el
archivo y en NOTICE.

## Licencia

MIT, consulte [LICENSE](LICENSE).
