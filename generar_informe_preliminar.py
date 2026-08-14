"""
GENERADOR DE INFORME PRELIMINAR - OPTIMIZACION DE DESPACHO SEIN (75 barras, CDF)

Lee los resultados y parametros ya producidos por dispatch_sein_cdf.py (CSVs y
graficos PNG en esta misma carpeta) y arma un informe autocontenido, en HTML
y/o PDF, pensado para que un especialista (asesor de tesis, ingeniero COES,
jurado) lo revise y pueda:
  1) ver de un vistazo los parametros y supuestos con los que se corrio el
     modelo (clasificacion tecnologica ASUMIDA, costos, margenes, reserva,
     parametros BESS, etc.), todos extraidos en vivo del codigo fuente y de
     los CSV de resultados (no estan hardcodeados aqui, para no desincronizarse
     si el modelo cambia);
  2) ver los resultados por escenario (tablas + graficos ya generados);
  3) revisar una lista de HALLAZGOS AUTOMATICOS (anomalias detectadas en los
     propios resultados, p.ej. BESS que nunca se usa, ERNC con participacion
     casi nula) y un CHECKLIST de posibles huecos de modelamiento, para
     orientar la critica tecnica.

No vuelve a resolver el despacho (usa los CSV/PNG ya generados por
dispatch_sein_cdf.py) -> es rapido y siempre corre "python dispatch_sein_cdf.py"
primero si cambiaste el modelo.

Requisitos (ya usados por dispatch_sein_cdf.py):
    pip install numpy pandas scipy matplotlib

Uso:
    python generar_informe_preliminar.py
    python generar_informe_preliminar.py --formato html
    python generar_informe_preliminar.py --formato pdf
    python generar_informe_preliminar.py --dir "ruta/a/la/carpeta/del/proyecto"
    python generar_informe_preliminar.py --abrir      (abre el HTML al terminar)
"""

import os
import re
import ast
import sys
import base64
import argparse
import textwrap
import contextlib
import io as _io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_MODULE_NAME = "dispatch_sein_cdf"


# ============================================================
# 0. CARGA DEL MODULO ORIGINAL (reusa constantes/clases, no re-optimiza)
# ============================================================

def load_model_module(base_dir):
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    import importlib
    if MODEL_MODULE_NAME in sys.modules:
        importlib.reload(sys.modules[MODEL_MODULE_NAME])
    dsc = importlib.import_module(MODEL_MODULE_NAME)
    return dsc


# ============================================================
# 1. EXTRACCION DE PARAMETROS/ESCENARIOS DESDE EL CODIGO FUENTE
# ============================================================
# Se usa ast (no regex/hardcode) sobre run_scenarios() para que la tabla de
# escenarios del informe siempre refleje el codigo actual, aunque se agreguen,
# quiten o modifiquen escenarios en dispatch_sein_cdf.py.

def extract_scenarios_and_horizon(source_path):
    with open(source_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    T_val, scenarios = None, []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_scenarios":
            for n in ast.walk(node):
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                    name = n.targets[0].id
                    if name == "T":
                        try:
                            T_val = ast.literal_eval(n.value)
                        except Exception:
                            pass
                    elif name == "scenarios":
                        for elt in n.value.elts:
                            d = {}
                            for kw in elt.keywords:
                                try:
                                    d[kw.arg] = ast.literal_eval(kw.value)
                                except Exception:
                                    d[kw.arg] = None
                            scenarios.append(d)
    return T_val, scenarios


def extract_gen_notes(source_path):
    """Extrae el comentario en linea (p.ej. '# sin certeza -> revisar') de cada
    entrada de GEN_TECH_ASSUMPTIONS, para no perder esa senal de confianza que
    el autor original dejo marcada linea por linea."""
    notes = {}
    pattern = re.compile(r'^\s*"([A-Za-z0-9_]+)":\s*dict\(.*?\)\s*,?\s*(#\s*(.*))?\s*$')
    with open(source_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                key, _, comment = m.groups()
                if comment:
                    notes[key] = comment.strip()
    return notes


def extract_scalar_params(dsc):
    params = [
        ("PMAX_MARGIN_FACTOR", dsc.PMAX_MARGIN_FACTOR,
         "Factor aplicado sobre PG del CDF (potencia despachada en el snapshot) para "
         "aproximar la capacidad instalada (Pmax) de cada central. Debe reemplazarse "
         "por Pmax real (COES/MINEM) para resultados definitivos."),
        ("HYDRO_BUDGET_FACTOR", dsc.HYDRO_BUDGET_FACTOR,
         "Fraccion de (capacidad x 24h) usable como energia hidraulica diaria total. "
         "No representa hidrologia/caudal real ni estacionalidad."),
        ("COST_CURTAILMENT (USD/MWh)", dsc.COST_CURTAILMENT, "Costo asumido por vertimiento de ERNC."),
        ("COST_ENS (USD/MWh)", dsc.COST_ENS, "Costo asumido por energia no suministrada."),
        ("BASE_MVA", dsc.BASE_MVA, "Potencia base del DC-OPF (IEEE CDF)."),
    ]
    for bess in dsc.BESS_UNITS_DEFAULT:
        label = f"BESS {bess['name']} ({bess['bus_name']})"
        params.append((f"{label} pmax/emax base", f"{bess['pmax']} MW / {bess['emax']} MWh",
                        "Se reescala por escenario via bess_scale (ver tabla de escenarios)."))
        params.append((f"{label} eta_ch/eta_dis/soc_init/cost_deg",
                        f"{bess['eta_ch']} / {bess['eta_dis']} / {bess['soc_init']} / {bess['cost_deg']} USD/MWh",
                        "Eficiencias, SOC inicial (=final, t=T-1) y costo de degradacion."))
    return params


def compute_unmatched_generators(dsc, cdf_path):
    buses, _ = dsc.parse_ieee_cdf(cdf_path)
    gens = [b for b in buses if b["pg"] > 0]
    return [g["name"] for g in gens if g["name"] not in dsc.GEN_TECH_ASSUMPTIONS]


def get_system_summary_text(system):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        system.summary()
    return buf.getvalue().strip()


# ============================================================
# 2. CARGA DE RESULTADOS YA GENERADOS (CSV / PNG)
# ============================================================

def load_results(base_dir):
    summary_path = os.path.join(base_dir, "resumen_escenarios_sein75_cdf.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"No se encontro {summary_path}. Ejecuta primero: python dispatch_sein_cdf.py"
        )
    summary_df = pd.read_csv(summary_path)

    despacho = {}
    for name in summary_df["scenario"]:
        p = os.path.join(base_dir, f"despacho_{name}.csv")
        if os.path.isfile(p):
            despacho[name] = pd.read_csv(p)

    images = {}
    for name in summary_df["scenario"]:
        p = os.path.join(base_dir, f"grafico_despacho_{name}.png")
        if os.path.isfile(p):
            images[name] = p

    comparison_image = os.path.join(base_dir, "grafico_comparacion_escenarios.png")
    comparison_image = comparison_image if os.path.isfile(comparison_image) else None

    gen_class_path = os.path.join(base_dir, "clasificacion_generadores_ASUMIDA_validar.csv")
    gen_class_df = pd.read_csv(gen_class_path) if os.path.isfile(gen_class_path) else None

    return summary_df, despacho, images, comparison_image, gen_class_df


# ============================================================
# 3. HALLAZGOS AUTOMATICOS (deteccion de posibles huecos)
# ============================================================

def compute_auto_findings(summary_df, despacho, scenarios_def, gen_assump_df, unmatched):
    findings = []

    max_share = summary_df["renewable_share_used_percent"].max()
    if max_share < 5:
        findings.append(dict(
            nivel="ALERTA",
            titulo="Participacion ERNC usada muy baja en todos los escenarios",
            detalle=(
                f"El maximo de participacion ERNC en la demanda entre todos los escenarios es "
                f"{max_share:.2f}%. El caso CDF base solo trae 2 unidades ERNC (solar en 'tacna', "
                f"eolica en 'marco220'). Escalar su disponibilidad x2/x3 apenas mueve la metrica "
                f"porque la capacidad base es pequena frente a la demanda del sistema (~80 MW vs "
                f"miles de MW de carga). Si la tesis busca representar 'alta penetracion ERNC', "
                f"revisar si conviene anadir mas puntos/capacidad de inyeccion ERNC en el caso base, "
                f"en vez de solo escalar los 2 existentes."
            )))

    if summary_df["curtailment_MWh"].sum() == 0:
        findings.append(dict(
            nivel="INFO",
            titulo="Vertimiento (curtailment) nulo en todos los escenarios",
            detalle=(
                "Puede ser un resultado legitimo (la red absorbe toda la ERNC disponible) o un "
                "sintoma de que la ERNC instalada es demasiado pequena para generar excedentes "
                "en algun momento del dia. Contrastar con el hallazgo de participacion ERNC baja."
            )))

    for sc in scenarios_def:
        name = sc.get("name")
        scale = sc.get("bess_scale", 0) or 0
        df = despacho.get(name)
        if df is None or scale <= 0.001:
            continue
        activity = df["bess_charge_MW"].sum() + df["bess_discharge_MW"].sum()
        if activity < 1e-6:
            findings.append(dict(
                nivel="ALERTA",
                titulo=f"BESS inactivo en el escenario '{name}'",
                detalle=(
                    f"El BESS esta habilitado (bess_scale={scale}) pero no carga ni descarga en "
                    "ninguna de las 24 horas en NINGUNA de sus 3 unidades (SOC constante en el "
                    "valor inicial). Con la formulacion LP continua (sin unit commitment ni curva "
                    "de oferta por bloques) el costo marginal puede quedar plano en algunos puntos "
                    "de escala de renovable/demanda, generando soluciones optimas degeneradas donde "
                    "ciclar el BESS no aporta ni resta costo y el solver elige no hacerlo. Contrastar "
                    "con escenarios de escala vecina (renewable_scale/load_scale) donde si se activa, "
                    "antes de concluir que el BESS 'no funciona' en este punto especifico."
                )))

    if gen_assump_df is not None and "baja_confianza" in gen_assump_df.columns:
        low = gen_assump_df[gen_assump_df["baja_confianza"]]
        if len(low):
            total_pmax = gen_assump_df["pmax_MW"].sum()
            low_pmax = low["pmax_MW"].sum()
            findings.append(dict(
                nivel="ALERTA",
                titulo=f"{len(low)} central(es) con clasificacion tecnologica/costo SIN CERTEZA",
                detalle=(
                    f"Representan {low_pmax:,.0f} MW de {total_pmax:,.0f} MW totales "
                    f"({100*low_pmax/max(total_pmax,1e-6):.1f}%) del parque asumido: "
                    f"{', '.join(low['nombre'].tolist())}. Estos costos/tecnologias fueron asignados "
                    "por defecto (sin fuente de dato real) y deben validarse con COES/MINEM o el "
                    "estudio de tesis antes de usar los resultados de forma definitiva."
                )))

    if unmatched:
        findings.append(dict(
            nivel="ALERTA",
            titulo=f"{len(unmatched)} generador(es) del CDF sin entrada en GEN_TECH_ASSUMPTIONS",
            detalle=(
                f"Nombres: {', '.join(unmatched)}. El modelo los trata como termicos genericos "
                "(costo=80 USD/MWh, rampa=0.35) por defecto. Verificar si corresponde y, de ser "
                "necesario, anadirlos explicitamente al diccionario de supuestos."
            )))

    if (summary_df["ENS_MWh"] > 0).any():
        peor = summary_df.loc[summary_df["ENS_MWh"].idxmax()]
        findings.append(dict(
            nivel="ALERTA",
            titulo="Energia no suministrada (ENS) > 0 en al menos un escenario",
            detalle=(
                f"Maximo en '{peor['scenario']}': {peor['ENS_MWh']:.2f} MWh. Revisar si se debe a "
                "restricciones de red (congestion), insuficiencia de capacidad firme, o a la reserva "
                "operativa exigida; podria indicar infactibilidad estructural en ese escenario."
            )))
    else:
        findings.append(dict(
            nivel="INFO",
            titulo="ENS = 0 en todos los escenarios",
            detalle=(
                "Ningun escenario presenta energia no suministrada. Verificar que esto no se deba a "
                "una reserva operativa/margenes demasiado holgados que faciliten artificialmente la "
                "factibilidad (ver PMAX_MARGIN_FACTOR y la formula de reserva operativa)."
            )))

    costs_by_scale = summary_df[["scenario", "total_cost_USD"]].copy()
    if not costs_by_scale["total_cost_USD"].is_monotonic_decreasing and len(costs_by_scale) > 1:
        pass  # informativo unicamente si se desea profundizar; no siempre es anomalia real

    return findings


# ============================================================
# 4. HTML
# ============================================================

def _img_to_data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _df_to_html_table(df, css_class="tbl", float_format="{:,.2f}", highlight_col=None):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].map(lambda v: float_format.format(v) if pd.notnull(v) else "")
    if highlight_col and highlight_col in df.columns:
        def row_style(row):
            return ' class="warn-row"' if str(row[highlight_col]).lower() in ("true", "1") else ""
        rows_html = []
        cols = [c for c in df.columns if c != highlight_col]
        rows_html.append("<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
        for _, row in df.iterrows():
            rows_html.append(f"<tr{row_style(row)}>" + "".join(f"<td>{row[c]}</td>" for c in cols) + "</tr>")
        return f'<table class="{css_class}">' + "".join(rows_html) + "</table>"
    return df.to_html(classes=css_class, index=False, border=0, escape=True)


def build_html(base_dir, dsc, scenarios_def, T_val, scalar_params, gen_assump_df,
                summary_df, despacho, images, comparison_image, findings,
                system_summary_text, unmatched, output_path):

    gen_dt = datetime.now().strftime("%Y-%m-%d %H:%M")

    css = """
    body { font-family: Segoe UI, Arial, sans-serif; margin: 0; padding: 0 0 60px 0; color: #222; background:#fff; }
    .wrap { max-width: 1000px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 26px; border-bottom: 3px solid #0072B2; padding-bottom: 10px; }
    h2 { font-size: 19px; color: #0072B2; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
    h3 { font-size: 15px; color: #333; margin-top: 22px; }
    .cover { background:#0072B2; color:#fff; padding: 40px 24px; }
    .cover h1 { color:#fff; border:none; }
    .cover p { opacity:.9; }
    table.tbl { border-collapse: collapse; width: 100%; margin: 10px 0 20px 0; font-size: 12.5px; }
    table.tbl th, table.tbl td { border: 1px solid #ddd; padding: 5px 8px; text-align: right; }
    table.tbl th { background: #f2f6fa; text-align: center; }
    table.tbl td:first-child, table.tbl th:first-child { text-align: left; }
    tr.warn-row { background: #fff2e0; }
    .finding { border-left: 5px solid #999; background:#f8f8f8; padding: 10px 14px; margin: 10px 0; border-radius: 3px; }
    .finding.ALERTA { border-left-color:#CC0000; background:#fdecea; }
    .finding.INFO { border-left-color:#0072B2; background:#eaf2fa; }
    .finding .tag { font-weight:700; font-size:11px; letter-spacing:.5px; text-transform:uppercase; }
    .finding .tag.ALERTA { color:#CC0000; }
    .finding .tag.INFO { color:#0072B2; }
    img.plot { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }
    .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
    .badge { display:inline-block; background:#eee; border-radius:12px; padding:2px 10px; font-size:12px; margin-right:6px; }
    .checklist li { margin: 6px 0; }
    .comment-box { border: 1px dashed #999; border-radius: 4px; min-height: 90px; margin: 8px 0 20px 0; }
    .params-note { font-size: 12px; color:#555; }
    footer { max-width:1000px; margin: 40px auto 0 auto; padding: 16px 24px; color:#888; font-size:12px; border-top:1px solid #eee; }
    .toc a { display:block; padding: 3px 0; color:#0072B2; text-decoration:none; }
    .toc a:hover { text-decoration:underline; }
    pre.sys { background:#f5f5f5; padding:10px 14px; border-radius:4px; font-size:12.5px; overflow-x:auto; }
    """

    findings_html = "".join(
        f'<div class="finding {f["nivel"]}"><span class="tag {f["nivel"]}">{f["nivel"]}</span> '
        f'&nbsp;<strong>{f["titulo"]}</strong><p>{f["detalle"]}</p></div>'
        for f in findings
    ) or '<p>No se detectaron anomalias automaticas con las reglas actuales.</p>'

    scenarios_df = pd.DataFrame(scenarios_def)
    params_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td><td class='params-note'>{note}</td></tr>"
        for k, v, note in scalar_params
    )

    gen_table_html = ""
    if gen_assump_df is not None:
        show_cols = [c for c in ["bus", "nombre", "tecnologia", "pmax_MW", "costo_USD_MWh",
                                  "rampa_MW_h", "nota"] if c in gen_assump_df.columns]
        gtmp = gen_assump_df[show_cols + (["baja_confianza"] if "baja_confianza" in gen_assump_df.columns else [])]
        gen_table_html = _df_to_html_table(gtmp, float_format="{:,.1f}", highlight_col="baja_confianza")
    else:
        gen_table_html = "<p><em>No se encontro clasificacion_generadores_ASUMIDA_validar.csv</em></p>"

    summary_cols = ["scenario", "total_cost_USD", "CMO_USD_MWh", "renewable_share_used_percent",
                     "curtailment_percent", "ENS_MWh", "thermal_generation_MWh",
                     "hydro_generation_MWh", "renewable_used_MWh"]
    summary_cols = [c for c in summary_cols if c in summary_df.columns]
    summary_table_html = _df_to_html_table(summary_df[summary_cols])

    scenario_sections = []
    for sc in scenarios_def:
        name = sc.get("name")
        img_tag = ""
        if name in images:
            img_tag = f'<img class="plot" src="{_img_to_data_uri(images[name])}" alt="Despacho {name}">'
        else:
            img_tag = "<p><em>Grafico no encontrado (ejecuta dispatch_sein_cdf.py).</em></p>"
        row = summary_df[summary_df["scenario"] == name]
        met_html = _df_to_html_table(row[summary_cols]) if len(row) else ""
        params_txt = ", ".join(f"{k}={v}" for k, v in sc.items() if k != "name")
        scenario_sections.append(f"""
        <h3 id="esc-{name}">{name}</h3>
        <p class="params-note">Parametros del escenario: {params_txt}</p>
        {met_html}
        {img_tag}
        """)
    scenario_sections_html = "".join(scenario_sections)

    comparison_html = (
        f'<img class="plot" src="{_img_to_data_uri(comparison_image)}" alt="Comparacion de escenarios">'
        if comparison_image else "<p><em>Grafico comparativo no encontrado.</em></p>"
    )

    checklist_items = [
        "Validar la clasificacion tecnologica ASUMIDA por nombre de central (tabla de generadores) contra datos reales de COES/MINEM, en especial las marcadas 'sin certeza'.",
        "Reemplazar PMAX (capacidad instalada) hoy aproximada como PG_CDF x PMAX_MARGIN_FACTOR por la capacidad instalada real de cada central.",
        "Validar los costos variables (USD/MWh) asumidos por tecnologia/central; hoy son valores tipicos, no provienen de una fuente documentada por central.",
        "El DC-OPF no modela perdidas ohmicas, ni limites de tension/reactivos (solo activa, sin restricciones de Q); evaluar si esto es aceptable para el alcance de la tesis.",
        "No hay Unit Commitment (variables binarias de encendido/apagado, costos de arranque/parada, tiempos minimos on/off); el despacho es puramente economico continuo.",
        "El presupuesto de energia hidraulica diaria (HYDRO_BUDGET_FACTOR=0.70 x capacidad x 24h) no representa hidrologia/caudal real ni estacionalidad; revisar si se dispone de series de caudal.",
        "Los perfiles horarios de demanda, solar y eolica son funciones sinusoidales sinteticas, no series historicas/medidas del SEIN; verificar si existen series reales a incorporar.",
        "El BESS se modela como 3 unidades en nodos fijos (BESS_UNITS_DEFAULT: chilc220/Lima, tacna/solar, marco220/eolico); validar si el numero, tamano y ubicacion corresponden a proyectos BESS reales/concretos (COES/MINEM) en vez de ser solo parametros de estudio.",
        "Revisar por que el BESS no se activa en varios escenarios (ver hallazgos automaticos) -- podria ser un hueco de modelamiento (falta de senal de precio horaria/UC) mas que un resultado economico real.",
        "La reserva operativa (8% de la carga + 15% de la ERNC disponible) es una regla ad-hoc; contrastar contra el criterio de reserva que use el COES o la norma tecnica aplicable.",
        "No se modelan contingencias N-1 ni seguridad de la red bajo falla de lineas/generadores; el analisis es solo del despacho economico en condiciones normales.",
        "La capacidad ERNC instalada en el caso base (2 unidades) es pequena frente a la demanda total; revisar si los escenarios 'x2/x3' logran representar 'alta penetracion ERNC' de forma realista (ver hallazgos automaticos).",
        "No se modela Pmin/generacion minima tecnica de las termicas (bounds inferiores en 0); evaluar si corresponde restriccion de operacion minima.",
        "El horizonte es un dia tipico de 24 horas; no se evalua variabilidad estacional/anual ni dias criticos (hidrologia seca, demanda maxima anual, etc.).",
    ]
    checklist_html = "".join(f"<li>{c}</li>" for c in checklist_items)

    sys_summary_html = system_summary_text.replace("\n", "<br>")
    unmatched_html = (
        f"<p><strong>Generadores sin clasificacion explicita</strong> (tratados como termicos "
        f"genericos por defecto): {', '.join(unmatched)}</p>" if unmatched else
        "<p>Todos los generadores con PG&gt;0 del CDF tienen entrada explicita en GEN_TECH_ASSUMPTIONS.</p>"
    )

    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Informe preliminar - Optimizacion de despacho SEIN</title>
<style>{css}</style>
</head>
<body>

<div class="cover">
  <div class="wrap">
    <h1>Informe preliminar de optimizacion de despacho</h1>
    <p>Sistema Electrico Interconectado Nacional (SEIN) - red IEEE CDF de 75 barras / 102 lineas</p>
    <p>DC-OPF multi-periodo (24h), escenarios de penetracion ERNC/BESS</p>
    <p>Generado automaticamente el {gen_dt} &middot; para revision de un especialista</p>
  </div>
</div>

<div class="wrap">

<p><strong>Como usar este informe:</strong> este documento se genera automaticamente a partir de
los resultados y del propio codigo del modelo (<code>dispatch_sein_cdf.py</code>). No reemplaza el
juicio de un especialista: su objetivo es exponer, de forma explicita y trazable, los supuestos,
parametros y posibles vacios metodologicos para facilitar la critica tecnica y la mejora del
modelo antes de considerar los resultados definitivos.</p>

<div class="toc">
  <a href="#hallazgos">1. Hallazgos automaticos (revisar primero)</a>
  <a href="#sistema">2. Descripcion del sistema</a>
  <a href="#metodologia">3. Metodologia</a>
  <a href="#parametros">4. Parametros y supuestos del modelo</a>
  <a href="#escenarios">5. Escenarios simulados</a>
  <a href="#generadores">6. Clasificacion tecnologica de generadores (ASUMIDA)</a>
  <a href="#resultados">7. Resultados comparativos</a>
  <a href="#detalle">8. Resultados por escenario</a>
  <a href="#checklist">9. Checklist de revision para el especialista</a>
  <a href="#comentarios">10. Espacio para comentarios del revisor</a>
</div>

<h2 id="hallazgos">1. Hallazgos automaticos (revisar primero)</h2>
{findings_html}

<h2 id="sistema">2. Descripcion del sistema</h2>
<pre class="sys">{sys_summary_html}</pre>
{unmatched_html}

<h2 id="metodologia">3. Metodologia</h2>
<ul>
  <li>Formulacion: DC-OPF (flujo de potencia optimo en corriente continua) multi-periodo, horizonte T={T_val} horas.</li>
  <li>Solver: <code>scipy.optimize.linprog</code>, metodo HiGHS (programacion lineal).</li>
  <li>Variables de decision por hora: generacion termica, hidro, solar y eolica por unidad; carga/descarga y SOC del BESS; energia no suministrada (ENS) por barra; angulo de fase (theta) por barra.</li>
  <li>Restricciones: balance nodal DC por barra y hora, limites de flujo por linea (+-fmax), dinamica de SOC del BESS (con SOC inicial = SOC final), presupuesto de energia hidraulica diaria por unidad, rampas horarias de las termicas, y una reserva operativa agregada.</li>
  <li>Funcion objetivo: minimizar costo variable termico + costo del "agua" hidro (shadow cost) + costo de degradacion del BESS + costo de ENS + costo de vertimiento (curtailment) de ERNC.</li>
</ul>

<h2 id="parametros">4. Parametros y supuestos del modelo</h2>
<table class="tbl">
<tr><th>Parametro</th><th>Valor</th><th>Nota</th></tr>
{params_rows}
</table>

<h2 id="escenarios">5. Escenarios simulados</h2>
{_df_to_html_table(scenarios_df)}

<h2 id="generadores">6. Clasificacion tecnologica de generadores (ASUMIDA -- validar)</h2>
<p class="params-note">Filas resaltadas: clasificacion/costo marcado en el codigo fuente como
"sin certeza" o "revisar". Estos supuestos <strong>deben</strong> contrastarse con datos reales
(COES/MINEM/estudio de tesis) antes de usar los resultados de forma definitiva.</p>
{gen_table_html}

<h2 id="resultados">7. Resultados comparativos entre escenarios</h2>
{summary_table_html}
{comparison_html}

<h2 id="detalle">8. Resultados por escenario</h2>
{scenario_sections_html}

<h2 id="checklist">9. Checklist de revision sugerido para el especialista</h2>
<ul class="checklist">
{checklist_html}
</ul>

<h2 id="comentarios">10. Espacio para comentarios del revisor</h2>
<p class="params-note">(Para completar a mano en la version impresa/PDF, o editando este HTML.)</p>
<div class="comment-box"></div>
<div class="comment-box"></div>
<div class="comment-box"></div>

</div>

<footer>
  Informe preliminar generado automaticamente por generar_informe_preliminar.py a partir de los
  resultados de dispatch_sein_cdf.py. No constituye version final de la tesis; los valores y
  supuestos aqui listados deben ser validados por un especialista.
</footer>

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================
# 5. PDF (matplotlib, sin dependencias externas adicionales)
# ============================================================

A4 = (8.27, 11.69)


def _new_page(title=None):
    fig = plt.figure(figsize=A4)
    if title:
        fig.text(0.06, 0.965, title, fontsize=13, fontweight="bold", color="#0072B2")
        fig.text(0.06, 0.95, "_" * 100, fontsize=6, color="#0072B2")
    return fig


def _add_text_page(pdf, title, paragraphs, fontsize=9):
    fig = _new_page(title)
    y = 0.90
    for para in paragraphs:
        wrapped = textwrap.wrap(para, width=100) or [""]
        for line in wrapped:
            fig.text(0.06, y, line, fontsize=fontsize, family="monospace")
            y -= 0.022
            if y < 0.06:
                pdf.savefig(fig)
                plt.close(fig)
                fig = _new_page(title + " (cont.)")
                y = 0.90
        y -= 0.012
    pdf.savefig(fig)
    plt.close(fig)


def _add_table_page(pdf, title, df, max_rows=32, fontsize=6.5, col_widths=None):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].map(lambda v: f"{v:,.2f}" if pd.notnull(v) else "")
    df = df.astype(str)
    rows = df.values.tolist()
    cols = df.columns.tolist()

    for start in range(0, max(len(rows), 1), max_rows):
        chunk = rows[start:start + max_rows]
        fig = _new_page(title if start == 0 else f"{title} (cont.)")
        ax = fig.add_axes([0.04, 0.06, 0.92, 0.82])
        ax.axis("off")
        tbl = ax.table(cellText=chunk, colLabels=cols, loc="upper center", cellLoc="right")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        tbl.scale(1, 1.3)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#0072B2")
                cell.set_text_props(color="white", fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)


def _add_image_page(pdf, title, image_path, caption=""):
    fig = _new_page(title)
    if image_path and os.path.isfile(image_path):
        img = mpimg.imread(image_path)
        ax = fig.add_axes([0.06, 0.10, 0.88, 0.80])
        ax.imshow(img)
        ax.axis("off")
    else:
        fig.text(0.06, 0.5, "[grafico no encontrado -- ejecuta dispatch_sein_cdf.py]", fontsize=10)
    if caption:
        fig.text(0.06, 0.05, caption, fontsize=8, color="#555")
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf(base_dir, dsc, scenarios_def, T_val, scalar_params, gen_assump_df,
              summary_df, despacho, images, comparison_image, findings,
              system_summary_text, unmatched, output_path):

    with PdfPages(output_path) as pdf:
        # Portada
        fig = plt.figure(figsize=A4)
        fig.text(0.5, 0.65, "Informe preliminar de optimizacion de despacho",
                  fontsize=18, fontweight="bold", ha="center", color="#0072B2")
        fig.text(0.5, 0.60, "Sistema Electrico Interconectado Nacional (SEIN)\n"
                             "Red IEEE CDF de 75 barras / 102 lineas -- DC-OPF multi-periodo (24h)",
                  fontsize=11, ha="center")
        fig.text(0.5, 0.53, "Escenarios de penetracion ERNC / BESS", fontsize=11, ha="center")
        fig.text(0.5, 0.44, f"Generado automaticamente el {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  fontsize=9, ha="center", color="#555")
        fig.text(0.5, 0.40, "Documento de trabajo para revision de un especialista.\n"
                             "No constituye version final.", fontsize=9, ha="center", color="#CC0000")
        pdf.savefig(fig)
        plt.close(fig)

        # Hallazgos automaticos
        paras = []
        if findings:
            for f in findings:
                paras.append(f"[{f['nivel']}] {f['titulo']}")
                paras.append(f["detalle"])
                paras.append("")
        else:
            paras.append("No se detectaron anomalias automaticas con las reglas actuales.")
        _add_text_page(pdf, "1. Hallazgos automaticos (revisar primero)", paras)

        # Descripcion del sistema
        sys_paras = system_summary_text.split("\n") + [""] + (
            [f"Generadores sin clasificacion explicita (tratados como termicos genericos): "
             f"{', '.join(unmatched)}"] if unmatched else
            ["Todos los generadores con PG>0 del CDF tienen entrada explicita en GEN_TECH_ASSUMPTIONS."]
        )
        _add_text_page(pdf, "2. Descripcion del sistema", sys_paras)

        # Metodologia
        metodologia = [
            f"Formulacion: DC-OPF multi-periodo, horizonte T={T_val} horas. Solver: scipy.optimize.linprog (HiGHS).",
            "Variables por hora: generacion termica/hidro/solar/eolica por unidad; carga/descarga y SOC del BESS; "
            "energia no suministrada (ENS) por barra; angulo de fase (theta) por barra.",
            "Restricciones: balance nodal DC por barra y hora, limites de flujo por linea, dinamica de SOC del BESS "
            "(SOC inicial = SOC final), presupuesto de energia hidraulica diaria por unidad, rampas horarias de las "
            "termicas, y una reserva operativa agregada.",
            "Objetivo: minimizar costo termico + costo hidro (shadow cost) + costo de degradacion BESS + costo de "
            "ENS + costo de vertimiento (curtailment) de ERNC.",
        ]
        _add_text_page(pdf, "3. Metodologia", metodologia)

        # Parametros
        params_df = pd.DataFrame(scalar_params, columns=["Parametro", "Valor", "Nota"])
        _add_table_page(pdf, "4. Parametros y supuestos del modelo", params_df, max_rows=14, fontsize=6.5)

        # Escenarios
        scenarios_df = pd.DataFrame(scenarios_def)
        _add_table_page(pdf, "5. Escenarios simulados", scenarios_df, max_rows=20)

        # Generadores
        if gen_assump_df is not None:
            show_cols = [c for c in ["bus", "nombre", "tecnologia", "pmax_MW", "costo_USD_MWh",
                                      "rampa_MW_h", "nota"] if c in gen_assump_df.columns]
            _add_table_page(pdf, "6. Clasificacion tecnologica de generadores (ASUMIDA -- validar)",
                             gen_assump_df[show_cols], max_rows=22, fontsize=6)
        else:
            _add_text_page(pdf, "6. Clasificacion tecnologica de generadores",
                            ["No se encontro clasificacion_generadores_ASUMIDA_validar.csv"])

        # Resultados comparativos
        summary_cols = ["scenario", "total_cost_USD", "CMO_USD_MWh", "renewable_share_used_percent",
                         "curtailment_percent", "ENS_MWh", "thermal_generation_MWh",
                         "hydro_generation_MWh", "renewable_used_MWh"]
        summary_cols = [c for c in summary_cols if c in summary_df.columns]
        _add_table_page(pdf, "7. Resultados comparativos entre escenarios", summary_df[summary_cols], max_rows=10, fontsize=6.5)
        _add_image_page(pdf, "7. Comparacion grafica de escenarios", comparison_image)

        # Detalle por escenario
        for sc in scenarios_def:
            name = sc.get("name")
            params_txt = ", ".join(f"{k}={v}" for k, v in sc.items() if k != "name")
            _add_image_page(pdf, f"8. Despacho horario -- {name}", images.get(name),
                             caption=f"Parametros: {params_txt}")

        # Checklist
        checklist = [
            "9. Checklist de revision sugerido para el especialista:",
            "",
            "- Validar la clasificacion tecnologica ASUMIDA por nombre de central contra datos reales "
            "(COES/MINEM), en especial las marcadas 'sin certeza'.",
            "- Reemplazar PMAX aproximado (PG_CDF x PMAX_MARGIN_FACTOR) por capacidad instalada real.",
            "- Validar los costos variables (USD/MWh) asumidos por tecnologia/central.",
            "- El DC-OPF no modela perdidas ohmicas ni limites de tension/reactivos.",
            "- No hay Unit Commitment (arranque/parada, tiempos minimos on/off).",
            "- El presupuesto hidro diario (factor 0.70) no representa hidrologia/caudal real.",
            "- Los perfiles horarios (demanda/solar/eolica) son sinteticos, no series historicas.",
            "- El BESS son 3 unidades en nodos fijos (Lima/tacna/marco220); validar contra proyectos reales.",
            "- Revisar por que el BESS no se activa en varios escenarios (ver hallazgos automaticos).",
            "- La reserva operativa (8% carga + 15% ERNC disponible) es una regla ad-hoc.",
            "- No se modelan contingencias N-1 ni seguridad de red.",
            "- La capacidad ERNC del caso base es pequena frente a la demanda total.",
            "- No se modela Pmin/generacion minima tecnica de las termicas.",
            "- El horizonte es un dia tipico de 24h, sin variabilidad estacional/anual.",
        ]
        _add_text_page(pdf, "9. Checklist de revision", checklist, fontsize=8.5)

        # Comentarios
        fig = _new_page("10. Espacio para comentarios del revisor")
        y = 0.85
        for _ in range(14):
            fig.text(0.06, y, "_" * 100, fontsize=9, color="#999")
            y -= 0.055
        pdf.savefig(fig)
        plt.close(fig)


# ============================================================
# 6. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Genera un informe preliminar (HTML/PDF) de la optimizacion de despacho SEIN.")
    parser.add_argument("--dir", default=BASE_DIR, help="Carpeta del proyecto (contiene dispatch_sein_cdf.py y los CSV/PNG de resultados).")
    parser.add_argument("--formato", choices=["html", "pdf", "ambos"], default="ambos")
    parser.add_argument("--abrir", action="store_true", help="Abre el HTML generado al terminar (Windows).")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.dir)
    dsc = load_model_module(base_dir)
    source_path = dsc.__file__

    print("Cargando resultados ya generados por dispatch_sein_cdf.py ...")
    summary_df, despacho, images, comparison_image, gen_class_df = load_results(base_dir)

    print("Extrayendo parametros y supuestos del codigo fuente ...")
    T_val, scenarios_def = extract_scenarios_and_horizon(source_path)
    scalar_params = extract_scalar_params(dsc)
    notes = extract_gen_notes(source_path)

    if gen_class_df is not None:
        gen_class_df = gen_class_df.copy()
        gen_class_df["nota"] = gen_class_df["nombre"].map(notes).fillna("")
        gen_class_df["baja_confianza"] = gen_class_df["nota"].str.contains(
            "revisar|sin certeza", case=False, regex=True)

    cdf_path = os.path.join(base_dir, "sein075cdf_10_100.txt")
    unmatched, system_summary_text = [], ""
    if os.path.isfile(cdf_path):
        unmatched = compute_unmatched_generators(dsc, cdf_path)
        system = dsc.CDFSystem(cdf_path)
        system_summary_text = get_system_summary_text(system)
    else:
        system_summary_text = "(No se encontro el archivo CDF original para recomputar la descripcion del sistema.)"

    print("Calculando hallazgos automaticos ...")
    findings = compute_auto_findings(summary_df, despacho, scenarios_def, gen_class_df, unmatched)
    for f in findings:
        print(f"  [{f['nivel']}] {f['titulo']}")

    html_path = os.path.join(base_dir, "Informe_Preliminar_Optimizacion_SEIN.html")
    pdf_path = os.path.join(base_dir, "Informe_Preliminar_Optimizacion_SEIN.pdf")

    if args.formato in ("html", "ambos"):
        build_html(base_dir, dsc, scenarios_def, T_val, scalar_params, gen_class_df,
                   summary_df, despacho, images, comparison_image, findings,
                   system_summary_text, unmatched, html_path)
        print(f"Informe HTML guardado en: {html_path}")

    if args.formato in ("pdf", "ambos"):
        build_pdf(base_dir, dsc, scenarios_def, T_val, scalar_params, gen_class_df,
                  summary_df, despacho, images, comparison_image, findings,
                  system_summary_text, unmatched, pdf_path)
        print(f"Informe PDF guardado en: {pdf_path}")

    if args.abrir and args.formato in ("html", "ambos") and os.name == "nt":
        os.startfile(html_path)


if __name__ == "__main__":
    main()
