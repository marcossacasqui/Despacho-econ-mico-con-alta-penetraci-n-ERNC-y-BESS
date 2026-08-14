"""
OPTIMIZACION DE COSTOS EN SISTEMAS HIDRO-TERMO-RENOVABLES CON ALTA PENETRACION ERNC Y BESS
Adaptacion al caso real: red IEEE Common Data Format (CDF) modificada - SEIN 75 barras / 102 lineas
NOTA: el archivo fuente rotula internamente el caso como "IEEE 69 Bus Test Case" (nombre historico
del caso academico UW Archive de 1993 del que deriva), pero el archivo sein075cdf_10_100.txt usado
aqui declara y contiene realmente 75 barras (BUS DATA FOLLOWS 75 ITEMS, numeradas 1-75 sin huecos).
Metodologia original: DC-OPF multi-periodo (24h) con linprog (HiGHS), escenarios de penetracion ERNC/BESS

Autor: Adaptable para tesis doctoral - Flexibilidad Operativa del SEIN

IMPORTANTE - LEER ANTES DE USAR LOS RESULTADOS:
El archivo IEEE CDF (formato estandar de flujo de potencia) NO contiene informacion de
tecnologia de generacion (termica/hidro/solar/eolica), costos variables, ni rampas.
Solo trae: barra, nombre, tipo, P/Q de carga, P/Q de generacion (capacidad instalada
aproximada tomada del caso base), y datos de lineas (R, X, B, capacidad MVA).

Para poder correr la metodologia de despacho economico fue necesario ASUMIR una
clasificacion tecnologica por barra de generacion, basada en el nombre de la
subestacion/central (p.ej. "mantaro" -> hidro, "chilca" -> termica a gas,
"marco220"/Marcona -> eolica, "tacna" -> solar). Estas asunciones estan en el
diccionario GEN_TECH_ASSUMPTIONS mas abajo, CLARAMENTE MARCADAS, y deben ser
revisadas/corregidas por el usuario con datos reales de la central (COES, MINEM,
o el propio estudio de tesis) antes de usar los resultados de forma definitiva.

Requisitos:
    pip install numpy pandas scipy matplotlib
"""

import os
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linprog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 0. PARSER DEL ARCHIVO IEEE CDF (formato estandar de flujo de potencia)
# ============================================================

def parse_ieee_cdf(path):
    """
    Lee un archivo IEEE Common Data Format y devuelve:
      buses  : lista de dicts (num, name, kv, area, type, pl, ql, pg, qg)
      lines  : lista de dicts (f, t, x, rating_mw)
    Layout de columnas (formato UW Archive clasico, separado por espacios):
      BUS DATA:    busnum name kV area owner type Vm Va PL QL PG QG kV2 Vdesired Qmax Qmin G B remoteBus
      BRANCH DATA: fbus tbus area zone circuit type R X B ratingA ratingB ratingC ctrlBus side ratio angle mintap maxtap step minlim maxlim rating name...
    NOTA: en este archivo las columnas estandar "Rating A/B/C" vienen en 0; la
    capacidad real de la linea/transformador (MVA) se encuentra en el ultimo
    campo numerico antes del nombre de la linea, y se usa aqui como limite MW.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read().splitlines()

    # --- Bus data ---
    bus_start = next(i for i, l in enumerate(raw) if "BUS DATA FOLLOWS" in l) + 1
    buses = []
    i = bus_start
    while not raw[i].strip().startswith("-999"):
        t = raw[i].split()
        buses.append(dict(
            num=int(t[0]), name=t[1], kv=float(t[2]), area=int(t[3]),
            type=int(t[5]), pl=float(t[8]), ql=float(t[9]),
            pg=float(t[10]), qg=float(t[11]),
        ))
        i += 1

    # --- Branch data ---
    branch_start = next(i for i, l in enumerate(raw) if "BRANCH DATA FOLLOWS" in l) + 1
    lines_ = []
    i = branch_start
    while not raw[i].strip().startswith("-999"):
        t = raw[i].split()
        fbus, tbus = int(t[0]), int(t[1])
        x = float(t[7])
        rating = float(t[21]) if len(t) > 21 else 100.0
        name = " ".join(t[22:]) if len(t) > 22 else f"{fbus}-{tbus}"
        lines_.append(dict(f=fbus, t=tbus, x=x, rating_mw=rating, name=name))
        i += 1

    return buses, lines_


# ============================================================
# 1. CLASIFICACION TECNOLOGICA DE GENERADORES (ASUNCION - VALIDAR)
# ============================================================
# tech: 'thermal' | 'hydro' | 'solar' | 'wind'
# cost: USD/MWh (costo variable / valor sombra del agua para hidro)
# ramp_frac: fraccion de Pmax por hora disponible como rampa (solo termicas)
GEN_TECH_ASSUMPTIONS = {
    # --- Termicas (gas/diesel) ---
    "chilca":   dict(tech="thermal", cost=50, ramp_frac=0.5),
    "talara":   dict(tech="thermal", cost=72, ramp_frac=0.4),
    "ventanil": dict(tech="thermal", cost=48, ramp_frac=0.5),
    "santaros": dict(tech="thermal", cost=64, ramp_frac=0.5),
    "chilc220": dict(tech="thermal", cost=50, ramp_frac=0.5),
    "chilc500": dict(tech="thermal", cost=50, ramp_frac=0.5),
    "aguaytia": dict(tech="thermal", cost=42, ramp_frac=0.4),
    "socab138": dict(tech="thermal", cost=95, ramp_frac=0.35),
    # Validadas ago-2026 contra fuentes secundarias (COES/OSINERGMIN oficiales no
    # descargables por certificado SSL vencido en osinerg.gob.pe -> ver investigacion
    # completa e informe con fuentes en el historial de la sesion). Las 4 con cambio
    # de tecnologia (paragsha, guadalup, huacho, paramong) tienen respaldo razonable
    # pero NO confirmado contra el Padron de Barras del COES; las 8 restantes siguen
    # "sin certeza" pero ahora con el hallazgo especifico anotado en vez de generico.
    "carhuama": dict(tech="thermal", cost=80, ramp_frac=0.35),   # sin certeza: no se hallo central real; S.E. Carhuamayo parece nudo de transito, no generador -> revisar contra Padron de Barras COES
    "parinas":  dict(tech="thermal", cost=85, ramp_frac=0.35),   # identidad confirmada (CT Refineria Talara/Petroperu, ~102 MW hallado vs 39 MW asumido); costo sin fuente -> revisar
    "santuari": dict(tech="thermal", cost=80, ramp_frac=0.35),   # sin certeza: no se hallo central real; S.E. Santuario parece nudo de subtransmision (Caylloma, Arequipa) -> revisar
    "independ": dict(tech="thermal", cost=80, ramp_frac=0.35),   # identidad media-alta (CT Pisco/Independencia EGASA, gas, ~71 MW hallado vs 117 MW asumido); costo sin fuente -> revisar
    "oroya":    dict(tech="thermal", cost=80, ramp_frac=0.35),   # AMBIGUO sin resolver: unica central hallada con este nombre es de 9 MW (vs 84.5 MW asumidos) -> posible error de clasificacion carga/generador en el CDF original, revisar antes de tocar
    "suriray":  dict(tech="thermal", cost=80, ramp_frac=0.35),   # sin certeza: no se hallo central real; S.E. Suriray parece nudo de interconexion minera (Cusco/Arequipa) -> posible error de clasificacion carga/generador, revisar
    "sanjuan":  dict(tech="thermal", cost=80, ramp_frac=0.35),   # sin certeza: hipotesis no confirmada de eolica San Juan de Marcona (nombre de S.E. real es "Marcona", no coincide) -> revisar antes de reclasificar
    "paramong": dict(tech="thermal", cost=52, ramp_frac=0.35),   # CONFIRMADO cogen. biomasa/bagazo (C. Biomasa Paramonga, Grupo Gloria, 23 MW); cost=52 USD/MWh es precio PPA Subasta RER 2010, no costo variable puro -> usar con esa salvedad

    # --- Hidro ---
    "carhuaqu": dict(tech="hydro", cost=18),
    "huall138": dict(tech="hydro", cost=15),
    "huall220": dict(tech="hydro", cost=15),
    "callahua": dict(tech="hydro", cost=15),
    "pachacha": dict(tech="hydro", cost=15),
    "huanza":   dict(tech="hydro", cost=16),
    "mantaro":  dict(tech="hydro", cost=12),
    "colcab2":  dict(tech="hydro", cost=12),
    "machupic": dict(tech="hydro", cost=20),
    "azangaro": dict(tech="hydro", cost=22),   # sin certeza: direccion hidro plausible por corredor San Gaban-Azangaro, pero ese corredor es 220kV y esta barra es 138kV -> central especifica no confirmada, revisar
    "paragsha": dict(tech="hydro", cost=15),   # RECLASIFICADA thermal->hydro (media confianza): S.E. Paragsha recibe la C.H. Chaglla (406-456 MW hallado vs 585 MW asumido); cost=15 es tipico hidro del propio diccionario, NO confirmado por fuente -> revisar
    "huacho":   dict(tech="hydro", cost=15),   # RECLASIFICADA thermal->hydro (media-alta confianza): S.E. Huacho recibe la C.H. Cheves (+posible Cahua), ~168-211 MW hallado vs 208 MW asumido; cost=15 es tipico hidro, NO confirmado por fuente -> revisar

    # --- Renovables (ERNC) ---
    "tacna":    dict(tech="solar"),
    "marco220": dict(tech="wind"),
    "guadalup": dict(tech="wind"),             # RECLASIFICADA thermal->wind (media confianza): S.E. Guadalupe evacua el Parque Eolico Cupisnique (83.15 MW hallado vs 143 MW asumido) -> revisar
}

# ADVERTENCIA CLAVE: el campo PG del CDF es la potencia DESPACHADA en el caso de flujo
# de potencia (snapshot), NO la capacidad instalada (Pmax real) de cada central. Usarla
# tal cual deja al sistema con ~6% de margen de reserva (7601 MW "aparentes" vs 7164 MW
# de demanda pico del propio caso), lo que vuelve el despacho infactible en varias horas
# (ENS artificial) apenas se exige una reserva operativa razonable. Se aplica un factor
# de margen sobre PG para aproximar Pmax; DEBE reemplazarse por la capacidad instalada
# real de cada central (COES/MINEM) para resultados definitivos.
PMAX_MARGIN_FACTOR = 1.30

# BESS: unidades multiples, cada una en una barra distinta (ninguna existe en el CDF ->
# parametros de estudio). Un unico BESS agregado en un nudo termico grande (chilc220) nunca
# muestra actividad: ese nudo tiene lineas de 600-700 MW y sobra generacion termica barata
# todas las horas, por lo que el costo marginal ahi es plano y no hay senal de precio que
# haga rentable cargar/descargar. Se anaden ademas dos BESS en los unicos puntos de inyeccion
# ERNC del CDF (tacna=solar, marco220=eolico), que si tienen exportacion limitada por linea
# (Moquegua-Tacna = 150 MW) y por lo tanto si generan tension de precio/congestion real con
# alta penetracion ERNC. Verificado empiricamente: con el BESS en chilc220 la actividad es
# 0 MWh/dia incluso vertiendo 20 GWh/dia; reubicado/anadido en tacna llega a ~400 MWh/dia.
BESS_UNITS_DEFAULT = [
    dict(name="BESS_Lima_Chilca", bus_name="chilc220", pmax=80, emax=200,
         eta_ch=0.95, eta_dis=0.95, soc_init=0.50, cost_deg=4),
    dict(name="BESS_Tacna_Solar", bus_name="tacna", pmax=30, emax=80,
         eta_ch=0.95, eta_dis=0.95, soc_init=0.50, cost_deg=4),
    dict(name="BESS_Marcona_Eolico", bus_name="marco220", pmax=30, emax=80,
         eta_ch=0.95, eta_dis=0.95, soc_init=0.50, cost_deg=4),
]

COST_CURTAILMENT = 8      # USD/MWh energia renovable vertida
COST_ENS = 3000           # USD/MWh energia no suministrada
HYDRO_BUDGET_FACTOR = 0.70  # fraccion de la capacidad*24h usable como energia diaria
BASE_MVA = 100.0          # potencia base estandar del IEEE CDF: P_MW = BASE_MVA * (1/x_pu) * dtheta


# ============================================================
# 2. CONSTRUCCION DEL SISTEMA A PARTIR DEL CDF
# ============================================================

class CDFSystem:
    def __init__(self, cdf_path):
        buses, lines_ = parse_ieee_cdf(cdf_path)
        self.buses_raw = buses
        self.bus_ids = [b["num"] for b in buses]
        self.n_buses = len(self.bus_ids)
        self.bus_index = {num: i for i, num in enumerate(self.bus_ids)}
        self.bus_name = {b["num"]: b["name"] for b in buses}

        # barra de referencia = tipo 3 (slack) en el CDF
        slack = [b["num"] for b in buses if b["type"] == 3]
        self.ref_bus_num = slack[0] if slack else self.bus_ids[0]
        self.ref_bus = self.bus_index[self.ref_bus_num]

        # demanda por barra (constante, tomada del caso base del CDF)
        self.base_load = np.array([b["pl"] for b in buses])

        # lineas: (i, j, x, rating_mw) con indices 0-based
        self.lines = []
        for l in lines_:
            i = self.bus_index[l["f"]]
            j = self.bus_index[l["t"]]
            x = l["x"] if l["x"] != 0 else 1e-4
            self.lines.append((i, j, x, l["rating_mw"], l["name"]))

        # generadores con PG>0 en el caso base -> capacidad instalada aproximada
        gens = [b for b in buses if b["pg"] > 0]

        unmatched = []
        self.thermal, self.hydro, self.solar, self.wind = [], [], [], []
        for g in gens:
            name = g["name"]
            bus_i = self.bus_index[g["num"]]
            pmax = g["pg"] * PMAX_MARGIN_FACTOR
            cfg = GEN_TECH_ASSUMPTIONS.get(name)
            if cfg is None:
                unmatched.append(name)
                cfg = dict(tech="thermal", cost=80, ramp_frac=0.35)  # respaldo conservador
            tech = cfg["tech"]
            if tech == "thermal":
                self.thermal.append(dict(name=name, bus=bus_i, pmax=pmax,
                                          cost=cfg["cost"], ramp=cfg["ramp_frac"] * pmax))
            elif tech == "hydro":
                self.hydro.append(dict(name=name, bus=bus_i, pmax=pmax, cost=cfg["cost"]))
            elif tech == "solar":
                self.solar.append(dict(name=name, bus=bus_i, pmax=pmax))
            elif tech == "wind":
                self.wind.append(dict(name=name, bus=bus_i, pmax=pmax))

        if unmatched:
            print(f"[AVISO] {len(unmatched)} generador(es) sin clasificacion explicita, "
                  f"asumidos como termicos genericos (revisar GEN_TECH_ASSUMPTIONS): {unmatched}")

        # BESS (multiples unidades, ver BESS_UNITS_DEFAULT)
        self.bess_units = []
        for u in BESS_UNITS_DEFAULT:
            bus_num = next(b["num"] for b in buses if b["name"] == u["bus_name"])
            unit = dict(u)
            unit["bus"] = self.bus_index[bus_num]
            self.bess_units.append(unit)

        self.cost_curtailment = COST_CURTAILMENT
        self.cost_ens = COST_ENS

    def set_bess_scale(self, scale):
        """Reescala pmax/emax de todas las unidades BESS a partir de BESS_UNITS_DEFAULT
        (no acumulativo entre escenarios). scale=0 -> BESS deshabilitado (emax minimo > 0
        para no romper los bounds del LP)."""
        for unit, base in zip(self.bess_units, BESS_UNITS_DEFAULT):
            unit["pmax"] = base["pmax"] * scale
            unit["emax"] = max(base["emax"] * scale, 1e-3)
            unit["soc_init"] = 0.0 if scale <= 0.001 else base["soc_init"]

    def summary(self):
        print(f"Barras: {self.n_buses} | Lineas: {len(self.lines)} | "
              f"Barra referencia: {self.bus_name[self.ref_bus_num]} (#{self.ref_bus_num})")
        print(f"Termicas: {len(self.thermal)} ({sum(g['pmax'] for g in self.thermal):.0f} MW) | "
              f"Hidro: {len(self.hydro)} ({sum(g['pmax'] for g in self.hydro):.0f} MW) | "
              f"Solar: {len(self.solar)} ({sum(g['pmax'] for g in self.solar):.0f} MW) | "
              f"Eolica: {len(self.wind)} ({sum(g['pmax'] for g in self.wind):.0f} MW)")
        bess_txt = ", ".join(f"{u['bus_name']} ({u['pmax']:.0f} MW/{u['emax']:.0f} MWh)"
                              for u in self.bess_units)
        print(f"BESS: {len(self.bess_units)} unidad(es) en {bess_txt}")
        print(f"Demanda total (caso base CDF): {self.base_load.sum():.1f} MW")


# ============================================================
# 3. PERFILES HORARIOS (misma logica que el script original)
# ============================================================

def create_profiles(system, T=24, load_scale=1.0, renewable_scale=1.0):
    hours = np.arange(T)

    load_shape = (
        0.62
        + 0.18 * np.sin((hours - 7) / 24 * 2 * np.pi)
        + 0.25 * np.sin((hours - 18) / 24 * 2 * np.pi) ** 2
    )
    load_shape = load_shape / load_shape.max()

    load_by_bus = np.outer(load_shape, system.base_load) * load_scale

    solar_shape = np.maximum(0, np.sin((hours - 6) / 12 * np.pi))
    wind_shape = 0.45 + 0.25 * np.sin((hours + 3) / 24 * 2 * np.pi) + 0.10 * np.sin(hours / 24 * 4 * np.pi)
    wind_shape = np.clip(wind_shape, 0.10, 0.95)

    solar_avail = {g["name"]: g["pmax"] * renewable_scale * solar_shape for g in system.solar}
    wind_avail = {g["name"]: g["pmax"] * renewable_scale * wind_shape for g in system.wind}
    hydro_avail = {g["name"]: np.ones(T) * g["pmax"] for g in system.hydro}

    return load_by_bus, solar_avail, wind_avail, hydro_avail


# ============================================================
# 4. OPTIMIZADOR DE DESPACHO DC-OPF (generalizado a N unidades / N barras)
# ============================================================

class DispatchOptimizer:
    def __init__(self, system, T, load_by_bus, solar_avail, wind_avail, hydro_avail, scenario_name):
        self.sys = system
        self.T = T
        self.load = load_by_bus
        self.solar_avail = solar_avail
        self.wind_avail = wind_avail
        self.hydro_avail = hydro_avail
        self.scenario_name = scenario_name

        self.n_th = len(system.thermal)
        self.n_hy = len(system.hydro)
        self.n_so = len(system.solar)
        self.n_wi = len(system.wind)
        self.n_bess = len(system.bess_units)
        self.n_bus = system.n_buses

        self.vars_per_t = (self.n_th + self.n_hy + self.n_so + self.n_wi
                            + 3 * self.n_bess  # bess: ch, dis, soc (por unidad)
                            + self.n_bus  # ens
                            + self.n_bus)  # theta
        self.offsets = self._build_offsets()

    def _build_offsets(self):
        o, k = {}, 0
        for key, count in [("th", self.n_th), ("hy", self.n_hy), ("so", self.n_so),
                            ("wi", self.n_wi), ("ch", self.n_bess), ("dis", self.n_bess),
                            ("soc", self.n_bess), ("ens", self.n_bus), ("theta", self.n_bus)]:
            o[key] = k
            k += count
        return o

    def idx(self, t, var, i=0):
        return t * self.vars_per_t + self.offsets[var] + i

    def solve(self):
        n = self.T * self.vars_per_t
        c = np.zeros(n)
        bounds = []

        for t in range(self.T):
            for g, gen in enumerate(self.sys.thermal):
                c[self.idx(t, "th", g)] = gen["cost"]
                bounds.append((0, gen["pmax"]))
            for g, gen in enumerate(self.sys.hydro):
                c[self.idx(t, "hy", g)] = gen["cost"]
                bounds.append((0, min(gen["pmax"], self.hydro_avail[gen["name"]][t])))
            for g, gen in enumerate(self.sys.solar):
                c[self.idx(t, "so", g)] = 0
                bounds.append((0, self.solar_avail[gen["name"]][t]))
            for g, gen in enumerate(self.sys.wind):
                c[self.idx(t, "wi", g)] = 0
                bounds.append((0, self.wind_avail[gen["name"]][t]))

            # offsets agrupa por bloque (todos los "ch", luego todos los "dis", luego
            # todos los "soc") -> los bounds deben respetar ese mismo orden por bloque,
            # no intercalado por unidad.
            for u, bess in enumerate(self.sys.bess_units):
                c[self.idx(t, "ch", u)] = bess["cost_deg"]
                bounds.append((0, bess["pmax"]))
            for u, bess in enumerate(self.sys.bess_units):
                c[self.idx(t, "dis", u)] = bess["cost_deg"]
                bounds.append((0, bess["pmax"]))
            for u, bess in enumerate(self.sys.bess_units):
                c[self.idx(t, "soc", u)] = 0
                bounds.append((0, max(bess["emax"], 1e-3)))

            for b in range(self.n_bus):
                c[self.idx(t, "ens", b)] = self.sys.cost_ens
                bounds.append((0, max(self.load[t, b], 0)))
            for b in range(self.n_bus):
                if b == self.sys.ref_bus:
                    bounds.append((0, 0))
                else:
                    bounds.append((-np.pi, np.pi))

        Aeq, beq, Aub, bub = [], [], [], []

        # Balance nodal DC
        for t in range(self.T):
            for b in range(self.n_bus):
                row = np.zeros(n)
                for g, gen in enumerate(self.sys.thermal):
                    if gen["bus"] == b:
                        row[self.idx(t, "th", g)] += 1
                for g, gen in enumerate(self.sys.hydro):
                    if gen["bus"] == b:
                        row[self.idx(t, "hy", g)] += 1
                for g, gen in enumerate(self.sys.solar):
                    if gen["bus"] == b:
                        row[self.idx(t, "so", g)] += 1
                for g, gen in enumerate(self.sys.wind):
                    if gen["bus"] == b:
                        row[self.idx(t, "wi", g)] += 1
                for u, bess in enumerate(self.sys.bess_units):
                    if bess["bus"] == b:
                        row[self.idx(t, "dis", u)] += 1
                        row[self.idx(t, "ch", u)] -= 1
                row[self.idx(t, "ens", b)] += 1

                for (i, j, x, fmax, _name) in self.sys.lines:
                    bij = BASE_MVA / x
                    if b == i:
                        row[self.idx(t, "theta", i)] -= bij
                        row[self.idx(t, "theta", j)] += bij
                    elif b == j:
                        row[self.idx(t, "theta", j)] -= bij
                        row[self.idx(t, "theta", i)] += bij

                Aeq.append(row)
                beq.append(self.load[t, b])

        # Limites de flujo por linea
        for t in range(self.T):
            for (i, j, x, fmax, _name) in self.sys.lines:
                bij = BASE_MVA / x
                row_pos = np.zeros(n)
                row_pos[self.idx(t, "theta", i)] = bij
                row_pos[self.idx(t, "theta", j)] = -bij
                Aub.append(row_pos); bub.append(fmax)
                Aub.append(-row_pos); bub.append(fmax)

        # Dinamica BESS (una cadena SOC_t = SOC_t-1 + eta_ch*ch - dis/eta_dis por unidad)
        for u, bess in enumerate(self.sys.bess_units):
            eta_ch, eta_dis = bess["eta_ch"], bess["eta_dis"]
            soc_init = bess["soc_init"] * bess["emax"]
            for t in range(self.T):
                row = np.zeros(n)
                row[self.idx(t, "soc", u)] = 1
                row[self.idx(t, "ch", u)] = -eta_ch
                row[self.idx(t, "dis", u)] = 1 / eta_dis
                if t == 0:
                    Aeq.append(row); beq.append(soc_init)
                else:
                    row[self.idx(t - 1, "soc", u)] = -1
                    Aeq.append(row); beq.append(0)
            row = np.zeros(n)
            row[self.idx(self.T - 1, "soc", u)] = 1
            Aeq.append(row); beq.append(soc_init)

        # Presupuesto energetico hidro (por unidad)
        for g, gen in enumerate(self.sys.hydro):
            budget = HYDRO_BUDGET_FACTOR * np.sum(self.hydro_avail[gen["name"]])
            row = np.zeros(n)
            for t in range(self.T):
                row[self.idx(t, "hy", g)] = 1
            Aub.append(row); bub.append(budget)

        # Rampas termicas
        for t in range(1, self.T):
            for g, gen in enumerate(self.sys.thermal):
                ramp = gen["ramp"]
                row = np.zeros(n)
                row[self.idx(t, "th", g)] = 1
                row[self.idx(t - 1, "th", g)] = -1
                Aub.append(row); bub.append(ramp)
                Aub.append(-row); bub.append(ramp)

        # Reserva operativa simplificada
        total_th = sum(g["pmax"] for g in self.sys.thermal)
        total_hy = sum(g["pmax"] for g in self.sys.hydro)
        total_bess_pmax = sum(bess["pmax"] for bess in self.sys.bess_units)
        for t in range(self.T):
            ren_avail_t = (sum(self.solar_avail[g["name"]][t] for g in self.sys.solar)
                           + sum(self.wind_avail[g["name"]][t] for g in self.sys.wind))
            reserve_req = 0.08 * np.sum(self.load[t, :]) + 0.15 * ren_avail_t
            total_headroom = total_th + total_hy + total_bess_pmax

            row = np.zeros(n)
            for g in range(self.n_th):
                row[self.idx(t, "th", g)] = 1
            for g in range(self.n_hy):
                row[self.idx(t, "hy", g)] = 1
            for u in range(self.n_bess):
                row[self.idx(t, "dis", u)] = 1
            Aub.append(row); bub.append(total_headroom - reserve_req)

        result = linprog(c=c, A_ub=np.array(Aub), b_ub=np.array(bub),
                          A_eq=np.array(Aeq), b_eq=np.array(beq),
                          bounds=bounds, method="highs")

        if not result.success:
            raise RuntimeError(f"No se encontro solucion optima ({self.scenario_name}): {result.message}")

        return self._extract_results(result.x)

    def _extract_results(self, x):
        rows = []
        for t in range(self.T):
            th_tot = sum(x[self.idx(t, "th", g)] for g in range(self.n_th))
            hy_tot = sum(x[self.idx(t, "hy", g)] for g in range(self.n_hy))
            so_tot = sum(x[self.idx(t, "so", g)] for g in range(self.n_so))
            wi_tot = sum(x[self.idx(t, "wi", g)] for g in range(self.n_wi))
            ch_by_unit = [x[self.idx(t, "ch", u)] for u in range(self.n_bess)]
            dis_by_unit = [x[self.idx(t, "dis", u)] for u in range(self.n_bess)]
            soc_by_unit = [x[self.idx(t, "soc", u)] for u in range(self.n_bess)]
            ch, dis, soc = sum(ch_by_unit), sum(dis_by_unit), sum(soc_by_unit)
            ens = sum(x[self.idx(t, "ens", b)] for b in range(self.n_bus))

            total_load = np.sum(self.load[t, :])
            solar_avail_t = sum(self.solar_avail[g["name"]][t] for g in self.sys.solar)
            wind_avail_t = sum(self.wind_avail[g["name"]][t] for g in self.sys.wind)
            curtailment = (solar_avail_t - so_tot) + (wind_avail_t - wi_tot)

            row_dict = dict(hour=t, load_MW=total_load, thermal_MW=th_tot, hydro_MW=hy_tot,
                             solar_used_MW=so_tot, wind_used_MW=wi_tot, bess_charge_MW=ch,
                             bess_discharge_MW=dis, bess_soc_MWh=soc, ENS_MWh=ens,
                             curtailment_MWh=curtailment,
                             renewable_available_MWh=solar_avail_t + wind_avail_t,
                             renewable_used_MWh=so_tot + wi_tot)
            for u, bess in enumerate(self.sys.bess_units):
                row_dict[f"bess_{bess['name']}_charge_MW"] = ch_by_unit[u]
                row_dict[f"bess_{bess['name']}_discharge_MW"] = dis_by_unit[u]
                row_dict[f"bess_{bess['name']}_soc_MWh"] = soc_by_unit[u]
            rows.append(row_dict)

        df = pd.DataFrame(rows)

        thermal_cost = sum(df["thermal_MW"].sum() * 0 for _ in [0])  # placeholder, recomputed below
        # costo termico exacto por unidad (por si hay costos distintos)
        thermal_cost = 0.0
        for g, gen in enumerate(self.sys.thermal):
            thermal_cost += sum(x[self.idx(t, "th", g)] for t in range(self.T)) * gen["cost"]
        hydro_cost = 0.0
        for g, gen in enumerate(self.sys.hydro):
            hydro_cost += sum(x[self.idx(t, "hy", g)] for t in range(self.T)) * gen["cost"]

        bess_cost = 0.0
        for u, bess in enumerate(self.sys.bess_units):
            bess_cost += (sum(x[self.idx(t, "ch", u)] for t in range(self.T))
                          + sum(x[self.idx(t, "dis", u)] for t in range(self.T))) * bess["cost_deg"]
        ens_cost = df["ENS_MWh"].sum() * self.sys.cost_ens
        curtailment_cost = df["curtailment_MWh"].sum() * self.sys.cost_curtailment
        total_cost = thermal_cost + hydro_cost + bess_cost + ens_cost + curtailment_cost

        total_demand = df["load_MW"].sum()
        renewable_avail = df["renewable_available_MWh"].sum()
        renewable_used = df["renewable_used_MWh"].sum()

        summary = dict(
            scenario=self.scenario_name, total_cost_USD=total_cost, thermal_cost_USD=thermal_cost,
            hydro_cost_USD=hydro_cost, bess_cost_USD=bess_cost, ens_cost_USD=ens_cost,
            curtailment_cost_USD=curtailment_cost, total_demand_MWh=total_demand,
            thermal_generation_MWh=df["thermal_MW"].sum(), hydro_generation_MWh=df["hydro_MW"].sum(),
            renewable_available_MWh=renewable_avail, renewable_used_MWh=renewable_used,
            curtailment_MWh=df["curtailment_MWh"].sum(),
            curtailment_percent=100 * df["curtailment_MWh"].sum() / max(renewable_avail, 1e-6),
            ENS_MWh=df["ENS_MWh"].sum(),
            renewable_share_used_percent=100 * renewable_used / max(total_demand, 1e-6),
            CMO_USD_MWh=total_cost / max(total_demand, 1e-6),
            max_bess_soc_MWh=df["bess_soc_MWh"].max(), min_bess_soc_MWh=df["bess_soc_MWh"].min(),
            bess_pmax_total_MW=sum(bess["pmax"] for bess in self.sys.bess_units),
            bess_emax_total_MWh=sum(bess["emax"] for bess in self.sys.bess_units),
            bess_cycled_MWh=df["bess_charge_MW"].sum() + df["bess_discharge_MW"].sum(),
        )
        return df, summary


# ============================================================
# 5. EJECUCION DE ESCENARIOS
# ============================================================

def run_scenarios(cdf_path):
    system = CDFSystem(cdf_path)
    system.summary()
    T = 24

    # NOTA sobre renewable_scale: con las 2 unicas unidades ERNC del CDF (tacna=39 MW solar,
    # marco220=41.6 MW eolico) frente a ~7164 MW de demanda pico, escalar x2/x3 (como en la
    # version anterior de este script) nunca genera excedente/congestion real, y por eso el
    # BESS jamas se activaba. Verificado empiricamente que la congestion real (linea
    # Moquegua-Tacna, 150 MW) empieza a manifestarse desde renewable_scale ~6-8. Se usan por
    # eso escalas mas altas en los escenarios de "alta penetracion ERNC" de abajo.
    scenarios = [
        dict(name="Base_CDF_ERNC_Actual", renewable_scale=1.00, bess_scale=1.00, load_scale=1.00),
        dict(name="ERNC_x3_BESS_Moderado", renewable_scale=3.00, bess_scale=1.00, load_scale=1.00),
        dict(name="ERNC_x8_BESS_Alto", renewable_scale=8.00, bess_scale=2.00, load_scale=1.00),
        dict(name="ERNC_x8_Sin_BESS", renewable_scale=8.00, bess_scale=0.00, load_scale=1.00),
        dict(name="ERNC_x8_Demanda_Alta", renewable_scale=8.00, bess_scale=2.00, load_scale=1.15),
    ]

    all_summaries, all_results = [], {}
    for sc in scenarios:
        system.set_bess_scale(sc["bess_scale"])

        load, solar, wind, hydro = create_profiles(
            system, T=T, load_scale=sc["load_scale"], renewable_scale=sc["renewable_scale"])

        opt = DispatchOptimizer(system, T, load, solar, wind, hydro, sc["name"])
        df, summary = opt.solve()
        all_summaries.append(summary)
        all_results[sc["name"]] = df

    return pd.DataFrame(all_summaries), all_results, system


# ============================================================
# 6. VISUALIZACION
# ============================================================
# Paleta categorica fija (Okabe-Ito, segura para daltonismo) - un color por
# tecnologia/serie, nunca reasignado segun el orden o el escenario.
COLOR_THERMAL = "#D55E00"
COLOR_HYDRO = "#0072B2"
COLOR_SOLAR = "#F0C808"
COLOR_WIND = "#009E73"
COLOR_BESS_DIS = "#7B3294"
COLOR_BESS_CHG = "#ABABAB"
COLOR_LOAD = "#333333"
COLOR_ENS = "#CC0000"      # color de estado (critico), no es una tecnologia
GRID_COLOR = "#DDDDDD"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_dispatch_stack(df, scenario_name, save_path):
    """Stack horario de despacho (MW) + demanda, para un escenario."""
    hours = df["hour"].values
    fig, ax = plt.subplots(figsize=(10, 5))

    series = [
        (df["hydro_MW"].values, COLOR_HYDRO, "Hidro"),
        (df["thermal_MW"].values, COLOR_THERMAL, "Termica"),
        (df["solar_used_MW"].values, COLOR_SOLAR, "Solar"),
        (df["wind_used_MW"].values, COLOR_WIND, "Eolica"),
        (df["bess_discharge_MW"].values, COLOR_BESS_DIS, "BESS (descarga)"),
    ]
    values = [s[0] for s in series]
    colors = [s[1] for s in series]
    labels = [s[2] for s in series]
    ax.stackplot(hours, values, colors=colors, labels=labels, edgecolor="white", linewidth=0.5, zorder=2)

    # carga de BESS: retiro de energia, se grafica como banda negativa
    if df["bess_charge_MW"].abs().sum() > 0:
        ax.fill_between(hours, 0, -df["bess_charge_MW"].values, color=COLOR_BESS_CHG,
                         label="BESS (carga)", zorder=2)

    ax.plot(hours, df["load_MW"].values, color=COLOR_LOAD, linewidth=2, label="Demanda", zorder=3)

    if df["ENS_MWh"].sum() > 0:
        ax.bar(hours, df["ENS_MWh"].values, bottom=df["load_MW"].values - df["ENS_MWh"].values,
               color=COLOR_ENS, width=0.6, label="ENS (no suministrada)", zorder=3)

    _style_axes(ax)
    ax.set_xlim(0, 23)
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("Potencia (MW)")
    ax.set_title(f"Despacho horario - {scenario_name}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_comparison(summary_df, save_path):
    """Comparacion entre escenarios de las metricas clave (small multiples)."""
    metrics = [
        ("total_cost_USD", "Costo total (USD)"),
        ("CMO_USD_MWh", "Costo marginal promedio (USD/MWh)"),
        ("renewable_share_used_percent", "Participacion ERNC usada (%)"),
        ("curtailment_percent", "Vertimiento ERNC (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    labels = summary_df["scenario"].values

    for ax, (col, title) in zip(axes.flat, metrics):
        vals = summary_df[col].values
        ax.barh(labels, vals, color=COLOR_HYDRO, zorder=2)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:,.1f}", va="center", ha="left", fontsize=8, color="#333333")
        _style_axes(ax)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Comparacion de escenarios - SEIN 75 barras (CDF)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6b. TABLA DE CLASIFICACION DE GENERADORES (sincronizada con el codigo, no a mano)
# ============================================================

def build_gen_classification_table(system, source_path):
    """Reconstruye clasificacion_generadores_ASUMIDA_validar.csv leyendo en vivo
    GEN_TECH_ASSUMPTIONS y sus comentarios en linea (bus, nombre, tecnologia, pmax,
    costo, rampa, nota, baja_confianza). Reemplaza una version que antes se hacia a
    mano y quedaba desincronizada del codigo apenas se reclasificaba un generador."""
    notes = {}
    pattern = re.compile(r'^\s*"([A-Za-z0-9_]+)":\s*dict\(.*?\)\s*,?\s*(#\s*(.*))?\s*$')
    with open(source_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                key, _, comment = m.groups()
                if comment:
                    notes[key] = comment.strip()

    uncertain_markers = ("sin certeza", "revisar", "ambiguo", "no se hallo",
                          "hipotesis", "no confirmad")

    def make_row(g, tech, cost="", ramp=""):
        name = g["name"]
        note = notes.get(name, "")
        return dict(
            bus=system.bus_ids[g["bus"]], nombre=name, tecnologia=tech,
            pmax_MW=round(g["pmax"], 1), costo_USD_MWh=cost,
            rampa_MW_h=round(ramp, 1) if ramp != "" else "",
            nota=note, baja_confianza=any(m in note.lower() for m in uncertain_markers),
        )

    rows = []
    rows += [make_row(g, "thermal", g["cost"], g["ramp"]) for g in system.thermal]
    rows += [make_row(g, "hydro", g["cost"]) for g in system.hydro]
    rows += [make_row(g, "solar") for g in system.solar]
    rows += [make_row(g, "wind") for g in system.wind]

    return pd.DataFrame(rows).sort_values("bus").reset_index(drop=True)


# ============================================================
# 6c. VISUALIZACION ADICIONAL (topologia + evidencia del efecto BESS)
# ============================================================

def plot_network_topology(system, save_path):
    """Diagrama de topologia (grafo) del sistema: barras coloreadas por tecnologia
    asumida, unidades BESS marcadas. El CDF no trae coordenadas geograficas -> layout
    de resorte (spring_layout) con semilla fija; NO es un diagrama unifilar geografico."""
    G = nx.Graph()
    G.add_nodes_from(range(system.n_buses))
    for (i, j, x, fmax, name) in system.lines:
        G.add_edge(i, j)

    tech_by_bus = {}
    for g in system.thermal:
        tech_by_bus[g["bus"]] = "thermal"
    for g in system.hydro:
        tech_by_bus[g["bus"]] = "hydro"
    for g in system.solar:
        tech_by_bus[g["bus"]] = "solar"
    for g in system.wind:
        tech_by_bus[g["bus"]] = "wind"

    color_map = {"thermal": COLOR_THERMAL, "hydro": COLOR_HYDRO,
                 "solar": COLOR_SOLAR, "wind": COLOR_WIND}
    node_colors = [color_map.get(tech_by_bus.get(n), "#B0B7BC") for n in G.nodes()]
    node_sizes = [260 if n in tech_by_bus else 70 for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42, k=0.35, iterations=200)

    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#C7CDD1", width=1.0, alpha=0.85)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                            edgecolors="white", linewidths=0.6)

    bess_buses = [bess["bus"] for bess in system.bess_units]
    ax.scatter([pos[b][0] for b in bess_buses], [pos[b][1] for b in bess_buses],
               s=520, facecolors="none", edgecolors=COLOR_BESS_DIS, linewidths=2.2,
               marker="o", zorder=5)

    rx, ry = pos[system.ref_bus]
    ax.scatter([rx], [ry], s=90, facecolors="none", edgecolors="#111111",
               linewidths=1.6, marker="s", zorder=6)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_THERMAL, markersize=10, label="Termica"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_HYDRO, markersize=10, label="Hidro"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SOLAR, markersize=10, label="Solar"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_WIND, markersize=10, label="Eolica"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#B0B7BC", markersize=7, label="Sin generacion (carga/transito)"),
        plt.Line2D([0], [0], marker="o", color=COLOR_BESS_DIS, markerfacecolor="none", markersize=13,
                   linewidth=0, markeredgewidth=2.2, label="Unidad BESS"),
        plt.Line2D([0], [0], marker="s", color="#111111", markerfacecolor="none", markersize=9,
                   linewidth=0, markeredgewidth=1.6, label=f"Barra de referencia ({system.bus_name[system.ref_bus_num]})"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=9)
    ax.set_title(f"Topologia SEIN {system.n_buses} barras / {len(system.lines)} lineas "
                 "(layout de resorte, sin coordenadas geograficas)", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bess_activity_by_location(all_results, system, save_path):
    """Barras agrupadas: ciclo diario (carga+descarga, MWh) de cada unidad BESS por
    escenario. Evidencia visual del hallazgo de ubicacion: la unidad en un nudo
    termico grande (chilc220) permanece inactiva en casi todos los escenarios,
    mientras que las unidades en las barras de inyeccion ERNC (tacna, marco220)
    si ciclan cuando hay tension real de precio/congestion."""
    scenario_names = list(all_results.keys())
    unit_names = [u["name"] for u in system.bess_units]
    unit_colors = [COLOR_BESS_DIS, COLOR_SOLAR, COLOR_WIND, COLOR_THERMAL, COLOR_HYDRO]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    n_units = max(len(unit_names), 1)
    bar_w = 0.8 / n_units
    x = np.arange(len(scenario_names))

    for u_idx, uname in enumerate(unit_names):
        vals = []
        for sc in scenario_names:
            df = all_results[sc]
            col_c, col_d = f"bess_{uname}_charge_MW", f"bess_{uname}_discharge_MW"
            vals.append(df[col_c].sum() + df[col_d].sum() if col_c in df.columns else 0.0)
        offset = (u_idx - (n_units - 1) / 2) * bar_w
        ax.bar(x + offset, vals, width=bar_w * 0.92,
               color=unit_colors[u_idx % len(unit_colors)],
               label=uname.replace("BESS_", "").replace("_", " "), zorder=2)

    _style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Ciclo BESS diario (MWh: carga + descarga)")
    ax.set_title("Actividad del BESS por ubicacion y escenario")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bess_curtailment_effect(summary_df, with_name, without_name, save_path):
    """Comparacion directa (2 barras) de vertimiento ERNC y costo total entre un
    escenario con BESS habilitado y su contraparte identica sin BESS, para aislar
    el efecto atribuible al BESS."""
    row_with = summary_df[summary_df["scenario"] == with_name].iloc[0]
    row_without = summary_df[summary_df["scenario"] == without_name].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    labels = ["Sin BESS", "Con BESS"]
    colors = ["#B0B7BC", COLOR_BESS_DIS]

    for ax, (vals, ylabel, title) in zip(axes, [
        ([row_without["curtailment_MWh"], row_with["curtailment_MWh"]],
         "Vertimiento ERNC (MWh/dia)", "Vertimiento"),
        ([row_without["total_cost_USD"], row_with["total_cost_USD"]],
         "Costo total del despacho (USD/dia)", "Costo"),
    ]):
        bars = ax.bar(labels, vals, color=colors, zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
        _style_axes(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle(f"Efecto del BESS: {with_name} vs {without_name}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":
    CDF_PATH = os.path.join(BASE_DIR, "sein075cdf_10_100.txt")

    summary_df, all_results, system = run_scenarios(CDF_PATH)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\n================ RESUMEN DE ESCENARIOS (red real 75 barras) ================\n")
    cols = ["scenario", "total_cost_USD", "CMO_USD_MWh", "renewable_share_used_percent",
            "curtailment_MWh", "curtailment_percent", "ENS_MWh",
            "thermal_generation_MWh", "hydro_generation_MWh", "renewable_used_MWh"]
    print(summary_df[cols].round(2).to_string(index=False))

    summary_df.to_csv(os.path.join(BASE_DIR, "resumen_escenarios_sein75_cdf.csv"), index=False)
    for name, df in all_results.items():
        df.to_csv(os.path.join(BASE_DIR, f"despacho_{name}.csv"), index=False)

    gen_class_path = os.path.join(BASE_DIR, "dispatch_sein_cdf.py")
    gen_class_df = build_gen_classification_table(system, gen_class_path)
    gen_class_df.to_csv(os.path.join(BASE_DIR, "clasificacion_generadores_ASUMIDA_validar.csv"), index=False)

    print(f"\nArchivos CSV guardados en {BASE_DIR}")

    for name, df in all_results.items():
        plot_dispatch_stack(df, name, os.path.join(BASE_DIR, f"grafico_despacho_{name}.png"))
    plot_scenario_comparison(summary_df, os.path.join(BASE_DIR, "grafico_comparacion_escenarios.png"))
    plot_network_topology(system, os.path.join(BASE_DIR, "grafico_topologia_sein75.png"))
    plot_bess_activity_by_location(all_results, system,
                                    os.path.join(BASE_DIR, "grafico_bess_actividad_ubicacion.png"))
    if "ERNC_x8_BESS_Alto" in all_results and "ERNC_x8_Sin_BESS" in all_results:
        plot_bess_curtailment_effect(summary_df, "ERNC_x8_BESS_Alto", "ERNC_x8_Sin_BESS",
                                      os.path.join(BASE_DIR, "grafico_bess_efecto_vertimiento_costo.png"))

    print(f"Graficos PNG guardados en {BASE_DIR}")
