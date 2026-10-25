"""Resúmenes auditables para el EDA MiBici; no materializa ni altera datos silver.

Procesa un mes a la vez. Los conteos son censales; solo los diagramas de dispersión
y distribuciones detalladas utilizan una muestra estratificada por archivo.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import numpy as np
import pandas as pd

from proyecto_mlops_equipo_9.dataset import (
    COLUMNAS, cargar_estaciones, cargar_mes, columnas_mes, meses_disponibles,
    nombre_estandar, ruta_viajes,
)


# Las reglas vienen del pipeline de limpieza: una sola fuente para la limpieza y el EDA.
from proyecto_mlops_equipo_9.cleaning import (
    POLITICA_DEFAULT, anotar_viajes, reglas_seleccion, seleccionar_viajes,
)


def huella(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _leer_ids(path):
    kw = dict(usecols=lambda c: nombre_estandar(c) == 'viaje_id')
    try:
        return pd.read_csv(path, encoding='utf-8', **kw).iloc[:, 0], 'utf-8'
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', **kw).iloc[:, 0], 'latin-1'


def inventariar():
    """Lee IDs para detectar coincidencias ENTRE archivos y fija las fuentes con SHA-256."""
    rows, ids = [], []
    for mes in meses_disponibles():
        p = ruta_viajes(mes)
        s, encoding = _leer_ids(p)
        cols = columnas_mes(mes)
        if set(COLUMNAS) - set(cols):
            raise ValueError(f'{mes}: faltan columnas del contrato')
        ids.append(s.dropna().drop_duplicates())
        rows.append(dict(mes=mes, archivo=p.name, filas=len(s), bytes=p.stat().st_size,
                         encoding=encoding, columnas=' | '.join(cols), sha256=huella(p)))
    if not rows:
        raise FileNotFoundError('No hay archivos mensuales en data/raw/viajes')
    todos = pd.concat(ids, ignore_index=True)
    repetidos = frozenset(todos[todos.duplicated(keep=False)].unique())
    return pd.DataFrame(rows), repetidos


def _concat(frames):
    return pd.concat(frames, ignore_index=True)


def zonificar(estaciones, celda_m=1000):
    """Cuadrícula exploratoria fija en proyección local aproximada, sin ajustar a demanda.

    No representa polígonos administrativos ni zonas operativas aprobadas.
    El origen y la latitud de referencia son constantes para reproducibilidad.
    """
    s = estaciones[['id','latitude','longitude']].copy()
    valid = s.latitude.between(-90,90) & s.longitude.between(-180,180)
    x = 6371000*np.radians(s.longitude)*np.cos(np.radians(20.67))
    y = 6371000*np.radians(s.latitude)
    s['x_m'],s['y_m'] = x,y
    s['zona'] = pd.Series(pd.NA,index=s.index,dtype='string')
    s.loc[valid,'zona'] = ('G'+str(celda_m)+'_'+np.floor(x[valid]/celda_m).astype(int).astype(str)
                           +'_'+np.floor(y[valid]/celda_m).astype(int).astype(str))
    return s


def explorar(inventario, ids_cruzados, muestra_por_mes=15000, semilla=9):
    """Devuelve tablas censales y muestras identificadas; utiliza la política vigente como escenario."""
    estaciones = cargar_estaciones()
    if estaciones['id'].isna().any() or not estaciones['id'].is_unique:
        raise ValueError('El catálogo requiere IDs presentes y únicos')
    coords = estaciones.set_index('id')[['latitude', 'longitude']].copy()
    fuera = ~coords['latitude'].between(-90, 90) | ~coords['longitude'].between(-180, 180)
    coords.loc[fuera, :] = np.nan
    politica = dict(POLITICA_DEFAULT)
    zonas = zonificar(estaciones)
    mapa_zonas = zonas.set_index('id')['zona']
    tablas = {k: [] for k in ['mensual', 'calidad', 'nulos', 'diario', 'horario',
        'estacion', 'muestra', 'sensibilidad', 'bitacora', 'demografia', 'sin_catalogo',
        'duraciones', 'horario_dia', 'primer_uso', 'zona_hora']}
    for mes in inventario['mes']:
        raw = cargar_mes(mes)
        if len(raw) != int(inventario.set_index('mes').loc[mes, 'filas']):
            raise ValueError('El archivo cambió entre inventario y exploración')
        a = anotar_viajes(raw, mes, coords, politica, ids_cruzados)
        a, bit = seleccionar_viajes(a, politica)
        final = a['es_valido'].to_numpy(dtype=bool)
        obligatorias = [bandera for _, bandera, _ in reglas_seleccion(politica)][:7]
        base = ~a[obligatorias].any(axis=1).to_numpy()
        if (final & ~base).any():
            raise AssertionError('La política completa debe ser subconjunto de integridad')
        if len(a) != int(final.sum()) + int(bit['filas_excluidas'].sum()):
            raise AssertionError('El balance de filas no cierra')
        tablas['bitacora'].append(bit)
        flags = [c for c in a if c.startswith('flag_')] + ['duracion_larga']
        tablas['calidad'].append(pd.DataFrame([
            dict(mes=mes, indicador=f, filas=int(a[f].sum()), porcentaje=100*a[f].mean())
            for f in flags]))
        tablas['nulos'].append(pd.DataFrame([
            dict(mes=mes, columna=c, filas=int(raw[c].isna().sum()), porcentaje=100*raw[c].isna().mean())
            for c in COLUMNAS]))
        fecha = raw['inicio'].dt.normalize()
        p = pd.Period(mes.replace('_', '-'), freq='M')
        dias = pd.date_range(p.start_time, p.end_time.normalize(), freq='D')
        daily = pd.DataFrame({'fecha': dias})
        for nombre, mask in [('original', np.ones(len(a), dtype=bool)), ('integridad', base), ('politica_actual', final)]:
            conteo = raw.loc[mask].groupby(fecha[mask]).size()
            daily[nombre] = daily['fecha'].map(conteo).fillna(0).astype(int)
        daily['mes'] = mes
        daily['dia_semana'] = daily['fecha'].dt.dayofweek
        daily['sin_registros'] = daily['original'].eq(0)
        # Estos ceros representan ausencia de registros; NO demuestran ausencia de demanda.
        tablas['diario'].append(daily)
        dur = raw['duracion_min']
        edades = raw['inicio'].dt.year - pd.to_numeric(raw['anio_nacimiento'], errors='coerce')
        resumen = dict(mes=mes, original=len(raw), integridad=int(base.sum()), politica_actual=int(final.sum()),
            dias_calendario=len(dias), dias_con_registros=int((daily['original']>0).sum()),
            estaciones_origen=raw['origen_id'].nunique(), estaciones_destino=raw['destino_id'].nunique(),
            usuarios=raw['usuario_id'].nunique(), mediana_duracion_original=dur.median(),
            mediana_duracion_politica=dur[final].median(), p99_duracion=dur.quantile(.99),
            inicio_min=str(raw['inicio'].min()), inicio_max=str(raw['inicio'].max()))
        resumen['exclusion_pct'] = 100*(1-final.mean())
        resumen['viajes_por_dia_calendario'] = len(raw)/len(dias)
        tablas['mensual'].append(pd.DataFrame([resumen]))
        for vista, mask in [('original', np.ones(len(a), dtype=bool)), ('politica_actual', final)]:
            subset = raw.loc[mask]
            h = subset.groupby(subset['inicio'].dt.hour).size().reindex(range(24), fill_value=0)
            tablas['horario'].append(pd.DataFrame({'mes':mes, 'vista':vista, 'hora':range(24), 'viajes':h.values}))
            hh = subset.groupby([subset['inicio'].dt.dayofweek.rename('dia_semana'),
                                  subset['inicio'].dt.hour.rename('hora')]).size().rename('viajes').reset_index()
            tablas['horario_dia'].append(hh.assign(mes=mes, vista=vista))
            origen = subset['origen_id'].value_counts()
            destino = subset['destino_id'].value_counts()
            st = pd.concat([origen.rename('salidas'), destino.rename('llegadas')],axis=1).fillna(0)
            st.index.name='id'
            tablas['estacion'].append(st.reset_index().assign(mes=mes, vista=vista))
            z = subset['origen_id'].map(mapa_zonas).rename('zona')
            zh = subset.groupby([z,subset['inicio'].dt.dayofweek.rename('dia_semana'),
                                  subset['inicio'].dt.hour.rename('hora')]).size().rename('salidas').reset_index()
            tablas['zona_hora'].append(zh.assign(mes=mes,vista=vista))
        first = raw.groupby('origen_id')['inicio'].min().rename('primera_observacion').reset_index().rename(columns={'origen_id':'id'})
        tablas['primer_uso'].append(first)
        unknown = pd.concat([raw.loc[~raw['origen_id'].isin(coords.index),'origen_id'],
                             raw.loc[~raw['destino_id'].isin(coords.index),'destino_id']]).value_counts()
        tablas['sin_catalogo'].append(unknown.rename_axis('id').rename('apariciones_extremos').reset_index().assign(mes=mes))
        misma = raw['origen_id'].eq(raw['destino_id'])
        for limite in [15, 16, 18, 80, 90, 100]:
            tablas['demografia'].append(pd.DataFrame([dict(mes=mes, corte=limite,
                edad_menor=int(edades.lt(limite).sum()), edad_mayor=int(edades.gt(limite).sum()))]))
        genero = raw['genero'].astype('string').fillna('Sin dato').value_counts()
        for categoria, count in genero.items():
            tablas['demografia'].append(pd.DataFrame([dict(mes=mes, genero=categoria, viajes=count)]))
        # Sensibilidad un factor a la vez, sobre TODOS los meses, manteniendo las demás reglas.
        scenarios = [('misma_estacion_s',t) for t in [30,60,90]] + [('velocidad_kmh',t) for t in [25,30,40,60]] + [('duracion_min',t) for t in [120,180,240]]
        reglas = dict((f, a[f].fillna(False).to_numpy(dtype=bool)) for _,f,activa in reglas_seleccion(politica) if activa)
        for factor, valor in scenarios:
            repl = {'misma_estacion_s':'flag_misma_estacion_corta','velocidad_kmh':'flag_velocidad_imposible','duracion_min':'duracion_larga'}[factor]
            rechazo = np.zeros(len(a), dtype=bool)
            for bandera, v in reglas.items():
                if bandera != repl: rechazo |= v
            if factor=='misma_estacion_s': nuevo = misma & dur.ge(0) & dur.lt(valor/60)
            elif factor=='velocidad_kmh': nuevo = a['velocidad_kmh'].gt(valor)
            else: nuevo = dur.gt(valor)
            rechazo |= nuevo.fillna(False).to_numpy(dtype=bool)
            retenidos = int((~rechazo).sum())
            tablas['sensibilidad'].append(pd.DataFrame([dict(mes=mes, factor=factor, umbral=valor,
                retenidos=retenidos, diferencia_vs_actual=retenidos-int(final.sum()))]))
        edges = [-np.inf,0,1,5,10,15,30,60,120,180,np.inf]
        grupos = pd.cut(dur, edges, right=False)
        for grupo in ['misma','distinta']:
            mask = misma if grupo=='misma' else ~misma
            counts = grupos[mask].value_counts(sort=False)
            tablas['duraciones'].append(pd.DataFrame({'mes':mes, 'grupo':grupo,
                'intervalo_min':counts.index.astype(str), 'viajes':counts.values}))
        # Muestra uniforme por mes, conservando probabilidades de inclusión para ponderar.
        sample = a.sample(n=min(muestra_por_mes,len(a)), random_state=semilla)
        columnas = ['mes_archivo','inicio','duracion_min','origen_id','destino_id','velocidad_kmh','es_valido']
        sample = sample[columnas].copy()
        sample['edad_original'] = edades.loc[sample.index]
        sample['peso'] = len(a)/len(sample)
        tablas['muestra'].append(sample)
        print(f'{mes}: {len(raw):,} filas; política actual conserva {final.sum():,}', flush=True)
    r = {k:_concat(v) for k,v in tablas.items()}
    r['estaciones'] = estaciones
    r['zonas'] = zonas
    r['inventario'] = inventario
    r['politica'] = politica
    r['ids_entre_archivos'] = len(ids_cruzados)
    diario = r['diario']
    referencia = diario.groupby(['mes','dia_semana'])['politica_actual'].transform('median')
    diario['ratio_mediana_dia_semana'] = diario['politica_actual'].div(referencia.replace(0,np.nan))
    diario['dia_bajo_por_investigar'] = diario['ratio_mediana_dia_semana'].lt(politica['umbral_dia_bajo'])
    return r


def panel_horario(r, zona_id, inicio, fin):
    """Demostración de agregación horaria; conserva faltantes para horas no observadas.

    Cero en un grupo no significa servicio activo sin demanda. Se exige validación
    operativa antes de rellenar y entrenar un panel completo.
    """
    inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
    if fin <= inicio or inicio.to_period('M') != (fin-pd.Timedelta(seconds=1)).to_period('M'):
        raise ValueError('La demostración debe estar dentro de un mes y tener fin > inicio')
    mes = inicio.strftime('%Y_%m')
    raw = cargar_mes(mes)
    coords = r['estaciones'].set_index('id')[['latitude','longitude']].copy()
    invalid = ~coords.latitude.between(-90,90) | ~coords.longitude.between(-180,180)
    coords.loc[invalid,:] = np.nan
    # La exploración ya identificó IDs cruzados; obtenerlos nuevamente evita esconder estado.
    _, cruzados = inventariar()
    a = anotar_viajes(raw,mes,coords,r['politica'],cruzados)
    a,_ = seleccionar_viajes(a,r['politica'])
    viajes = a[a.es_valido]
    mapa = r['zonas'].set_index('id')['zona']
    salidas = viajes[viajes.origen_id.map(mapa).eq(zona_id)].groupby(viajes.inicio.dt.floor('h')).size()
    llegadas = viajes[viajes.destino_id.map(mapa).eq(zona_id)].groupby(viajes.fin.dt.floor('h')).size()
    horas = pd.date_range(inicio, fin, freq='h', inclusive='left')
    p = pd.DataFrame({'fecha_hora':horas})
    p['zona'] = zona_id
    p['salidas_observadas'] = p.fecha_hora.map(salidas).astype('Float64')
    p['llegadas_observadas'] = p.fecha_hora.map(llegadas).astype('Float64')
    p['sin_salidas_registradas'] = p.salidas_observadas.isna()
    p['hora_sin'] = np.sin(2*np.pi*p.fecha_hora.dt.hour/24)
    p['hora_cos'] = np.cos(2*np.pi*p.fecha_hora.dt.hour/24)
    p['dia_semana'] = p.fecha_hora.dt.dayofweek
    # Se desplazan primero: ningún predictor usa el conteo del instante objetivo.
    p['salidas_lag_24h'] = p.salidas_observadas.shift(24)
    p['media_previa_24h'] = p.salidas_observadas.shift(1).rolling(24,min_periods=24).mean()
    p['particion'] = np.where(p.fecha_hora < inicio+(fin-inicio)*.6,'entrenamiento',
                     np.where(p.fecha_hora < inicio+(fin-inicio)*.8,'validacion','prueba'))
    return p


def exportar(r, destino, fuentes_codigo=(), notebook_fuentes_sha256=None):
    """Exporta tablas y manifiesto; nunca exporta IDs individuales de usuarios."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in r.items():
        if isinstance(tabla,pd.DataFrame) and nombre != 'muestra':
            tabla.to_csv(destino/f'{nombre}.csv', index=False)
    from proyecto_mlops_equipo_9.config import RAW_DATA_DIR
    catalogo = sorted((RAW_DATA_DIR/'estaciones').glob('nomenclatura_*.csv'))[-1]
    entorno = {}
    for paquete in ['pandas','numpy','matplotlib','scikit-learn','nbclient','mlflow']:
        try: entorno[paquete] = importlib.metadata.version(paquete)
        except importlib.metadata.PackageNotFoundError: entorno[paquete] = 'no instalado'
    manifest = dict(politica=r['politica'], semillas=9, muestra_por_mes=15000,
        catalogo=dict(archivo=catalogo.name,sha256=huella(catalogo)),
        fuentes=r['inventario'].to_dict('records'),
        codigo={Path(p).name:huella(p) for p in fuentes_codigo},entorno=entorno,
        notebook_fuentes_sha256=notebook_fuentes_sha256,
        zonificacion=dict(tipo='cuadricula_exploratoria',celda_m=1000,latitud_referencia=20.67,
                         estado='pendiente de validación con negocio'),
        alcance='EDA de registros observados; no estima demanda insatisfecha ni efecto causal de e-bikes')
    manifest['identidad_analisis'] = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (destino/'manifiesto_eda.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest
