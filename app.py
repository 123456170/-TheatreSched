import streamlit as st
import numpy as np
import pandas as pd
import math
import random
import time
import copy
import io
import os
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model
import plotly.graph_objects as go

from PIL import Image, ImageDraw, ImageFont

try:
    import imageio.v2 as imageio
except Exception:
    try:
        import imageio
    except Exception:
        imageio = None


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DAY_START = 6 * 60       # 06:00
DAY_END = 20 * 60        # 20:00
TARGET_END = 17 * 60     # 17:00
Z_80 = 0.8416212335729143

PRIORITY_RANK = {"emergency": 0, "urgent": 1, "elective": 2}
PRIORITY_COLORS = {
    "emergency": "#d62728",
    "urgent": "#ff7f0e",
    "elective": "#1f77b4",
}

SIZE_MAP = {
    "Small": (2, 10),
    "Medium": (4, 25),
    "Large": (8, 45),
}


def fmt_min(m: Optional[float]) -> str:
    if m is None:
        return "--:--"
    try:
        if isinstance(m, float) and math.isnan(m):
            return "--:--"
        m = int(round(m))
        return f"{m // 60:02d}:{m % 60:02d}"
    except Exception:
        return "--:--"


def safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        return int(v)
    except Exception:
        return None


def as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, float) and math.isnan(v):
        return []
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return list(v)
    return []


def fmt_num(x: Any, digits: int = 0, suffix: str = "") -> str:
    if x is None:
        return "N/A"
    try:
        xf = float(x)
        if math.isnan(xf):
            return "N/A"
        return f"{xf:.{digits}f}{suffix}"
    except Exception:
        return str(x)


def solver_best_bound(solver: cp_model.CpSolver) -> Optional[float]:
    """
    Compatibility helper for different OR-Tools CP-SAT Python APIs.
    Some versions expose BestObjectiveBound(), others may expose ObjectiveBound().
    """
    for name in ("BestObjectiveBound", "ObjectiveBound", "best_objective_bound"):
        fn = getattr(solver, name, None)
        if callable(fn):
            try:
                val = fn()
                if val is not None:
                    return float(val)
            except Exception:
                pass

    # Fallback: parse ResponseStats JSON if available.
    try:
        stats = json.loads(solver.ResponseStats())
        for key in ("best_bound", "objective_bound", "bestObjectiveBound"):
            if key in stats:
                return float(stats[key])
    except Exception:
        pass

    return None


# ----------------------------------------------------------------------------
# Domain objects (logical modules: data/, duration/, model/)
# ----------------------------------------------------------------------------
@dataclass
class Room:
    id: str
    name: str
    features: Set[str]
    open_start: int = DAY_START
    open_end: int = DAY_END
    failed_after: Optional[int] = None


@dataclass
class Surgeon:
    id: str
    name: str
    specialty: str
    windows: List[Tuple[int, int]]
    max_hours: float
    unavailable_after: Optional[int] = None


@dataclass
class Staff:
    id: str
    name: str
    role: str
    skills: Set[str]
    shift_start: int
    shift_end: int
    unavailable_after: Optional[int] = None


@dataclass
class Case:
    id: str
    procedure: str
    priority: str
    specialty: str
    required_features: Set[str]
    required_staff: Dict[str, int]
    required_skills: Set[str]
    earliest_start: int
    deadline: Optional[int]
    mu: float
    sigma: float
    planning_duration: int
    arrival_time: int
    preferred_surgeons: List[str] = field(default_factory=list)


@dataclass
class Instance:
    rooms: List[Room]
    surgeons: List[Surgeon]
    staff: List[Staff]
    cases: List[Case]
    cleaning_by_procedure: Dict[str, int]
    seed: int


PROCEDURES = {
    "laparoscopic_cholecystectomy": {
        "specialty": "general",
        "mean": 85,
        "sd": 22,
        "features": {"imaging"},
        "staff": {"anesthesiologist": 1, "scrub_nurse": 1, "circulating_nurse": 1},
        "skills": set(),
        "cleaning": 30,
    },
    "appendectomy": {
        "specialty": "general",
        "mean": 65,
        "sd": 18,
        "features": set(),
        "staff": {"anesthesiologist": 1, "scrub_nurse": 1, "circulating_nurse": 1},
        "skills": set(),
        "cleaning": 25,
    },
    "inguinal_hernia": {
        "specialty": "general",
        "mean": 70,
        "sd": 20,
        "features": set(),
        "staff": {"anesthesiologist": 1, "scrub_nurse": 1, "circulating_nurse": 1},
        "skills": set(),
        "cleaning": 25,
    },
    "colectomy": {
        "specialty": "general",
        "mean": 145,
        "sd": 35,
        "features": {"imaging"},
        "staff": {"anesthesiologist": 1, "scrub_nurse": 1, "circulating_nurse": 1},
        "skills": set(),
        "cleaning": 40,
    },
    "robotic_prostatectomy": {
        "specialty": "uro",
        "mean": 180,
        "sd": 40,
        "features": {"robotic", "imaging"},
        "staff": {"anesthesiologist": 1, "scrub_nurse": 1, "circulating_nurse": 1},
        "skills": set(),
        "cleaning": 40,
    },
    "hysterectomy": {
        "specialty": "gyn",
        "mean": 120,
        "sd": 30,
        "features": {"robotic"},
        "staff": {"anesthesiologist": 1, "scrub_nurse": 1, "circulating_nurse": 1},
        "skills": set(),
        "cleaning": 35,
    },
    "knee_arthroscopy": {
        "specialty": "ortho",
        "mean": 75,
        "sd": 20,
        "features": {"imaging"},
        "staff": {"anesthesiologist": 1, "scrub_nurse": 1, "circulating_nurse": 1},
        "skills": set(),
        "cleaning": 25,
    },
    "spinal_fusion": {
        "specialty": "spine",
        "mean": 210,
        "sd": 50,
        "features": {"navigation", "imaging"},
        "staff": {"anesthesiologist": 1, "scrub_nurse": 1, "circulating_nurse": 1},
        "skills": set(),
        "cleaning": 45,
    },
    "craniotomy": {
        "specialty": "neuro",
        "mean": 240,
        "sd": 55,
        "features": {"imaging", "navigation"},
        "staff": {"anesthesiologist": 1, "scrub_nurse": 1, "circulating_nurse": 1},
        "skills": set(),
        "cleaning": 45,
    },
    "cardiac_bypass": {
        "specialty": "cardio",
        "mean": 280,
        "sd": 60,
        "features": {"cardiac_bypass", "imaging"},
        "staff": {
            "anesthesiologist": 1,
            "scrub_nurse": 1,
            "circulating_nurse": 1,
            "perfusionist": 1,
        },
        "skills": set(),
        "cleaning": 50,
    },
}

ROOM_TEMPLATES = [
    ({"robotic", "imaging"}, "R1 Hybrid Robotic"),
    ({"cardiac_bypass", "imaging"}, "R2 Cardiac"),
    ({"imaging", "navigation"}, "R3 Neuro Imaging"),
    ({"robotic", "navigation"}, "R4 Spine Robotic"),
    ({"cardiac_bypass", "robotic"}, "R5 Complex Cardiac"),
    ({"imaging"}, "R6 General Imaging"),
    ({"navigation"}, "R7 Ortho Navigation"),
    ({"robotic", "imaging", "navigation"}, "R8 Premium Robotic"),
]


# ----------------------------------------------------------------------------
# Duration module
# ----------------------------------------------------------------------------
def lognormal_params_from_mean_sd(mean: float, sd: float) -> Tuple[float, float]:
    mean = max(1.0, float(mean))
    sd = max(1.0, float(sd))
    var = sd * sd
    mu = math.log((mean * mean) / math.sqrt(var + mean * mean))
    sigma = math.sqrt(math.log(1.0 + var / (mean * mean)))
    return mu, sigma


def fit_lognormal(samples: np.ndarray) -> Tuple[float, float]:
    samples = np.asarray(samples, dtype=float)
    mean = float(np.mean(samples))
    var = float(np.var(samples, ddof=1)) if len(samples) > 1 else 1.0
    return lognormal_params_from_mean_sd(mean, math.sqrt(var))


def lognormal_quantile(mu: float, sigma: float, q: float) -> float:
    z = {0.5: 0.0, 0.8: Z_80}.get(q, Z_80)
    return math.exp(mu + sigma * z)


def simulate_historical_durations(mu: float, sigma: float, n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.lognormal(mu, sigma, size=n)


# ----------------------------------------------------------------------------
# Synthetic data generation
# ----------------------------------------------------------------------------
def make_rooms(n_rooms: int, rng: random.Random) -> List[Room]:
    rooms = []
    for i in range(n_rooms):
        feats, name = ROOM_TEMPLATES[i % len(ROOM_TEMPLATES)]
        rooms.append(Room(id=f"RM{i+1}", name=name, features=set(feats)))
    return rooms


def make_surgeons(rng: random.Random) -> List[Surgeon]:
    specialties = ["general", "ortho", "cardio", "neuro", "gyn", "uro", "spine"]
    surgeons = []
    for i, spec in enumerate(specialties):
        start = DAY_START + rng.randint(0, 30)
        end = DAY_END - rng.randint(0, 60)
        surgeons.append(
            Surgeon(
                id=f"SURG{i+1}",
                name=f"Dr {spec.title()} {i+1}",
                specialty=spec,
                windows=[(start, end)],
                max_hours=rng.uniform(7.0, 9.5),
            )
        )
    surgeons.append(
        Surgeon(
            id="SURG8",
            name="Dr General 2",
            specialty="general",
            windows=[(DAY_START + 30, DAY_END - 30)],
            max_hours=8.0,
        )
    )
    return surgeons


def make_staff(rng: random.Random, n_rooms: int) -> List[Staff]:
    staff = []
    n_anesth = max(2, n_rooms // 2 + 1)
    n_scrub = max(2, n_rooms // 2 + 2)
    n_circ = max(2, n_rooms // 2 + 2)

    for i in range(n_anesth):
        staff.append(
            Staff(
                id=f"ANES{i+1}",
                name=f"Anesthesiologist {i+1}",
                role="anesthesiologist",
                skills={"basic"},
                shift_start=DAY_START,
                shift_end=DAY_END - rng.randint(0, 45),
            )
        )
    for i in range(n_scrub):
        staff.append(
            Staff(
                id=f"SCRUB{i+1}",
                name=f"Scrub Nurse {i+1}",
                role="scrub_nurse",
                skills={"basic"},
                shift_start=DAY_START + rng.randint(0, 20),
                shift_end=DAY_END - rng.randint(0, 30),
            )
        )
    for i in range(n_circ):
        staff.append(
            Staff(
                id=f"CIRC{i+1}",
                name=f"Circulating Nurse {i+1}",
                role="circulating_nurse",
                skills={"basic"},
                shift_start=DAY_START + rng.randint(0, 20),
                shift_end=DAY_END - rng.randint(0, 30),
            )
        )
    staff.append(
        Staff(
            id="PERF1",
            name="Perfusionist 1",
            role="perfusionist",
            skills={"cardiac"},
            shift_start=DAY_START,
            shift_end=DAY_END,
        )
    )
    return staff


def generate_instance(
    num_rooms: int = 2,
    num_cases: int = 10,
    seed: int = 42,
    emergency_rate: float = 0.12,
) -> Instance:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    rooms = make_rooms(num_rooms, rng)
    surgeons = make_surgeons(rng)
    staff = make_staff(rng, num_rooms)
    specialties = sorted({s.specialty for s in surgeons})

    cases: List[Case] = []
    for i in range(num_cases):
        u = rng.random()
        if u < emergency_rate:
            priority = "emergency"
        elif u < emergency_rate + 0.25:
            priority = "urgent"
        else:
            priority = "elective"

        spec = rng.choice(specialties)
        eligible_procs = [
            p for p, meta in PROCEDURES.items()
            if meta["specialty"] == spec
            and any(r.features.issuperset(meta["features"]) for r in rooms)
        ]
        if not eligible_procs:
            eligible_procs = list(PROCEDURES.keys())
        proc = rng.choice(eligible_procs)
        meta = PROCEDURES[proc]

        req_feats = set(meta["features"])
        if not any(r.features.issuperset(req_feats) for r in rooms):
            req_feats = set()

        mu, sigma = lognormal_params_from_mean_sd(meta["mean"], meta["sd"])
        samples = np_rng.lognormal(mu, sigma, size=300)
        planning = int(math.ceil(float(np.quantile(samples, 0.8))))
        planning = max(25, min(planning, 300))

        if priority == "emergency":
            arrival = int(max(DAY_START, min(DAY_END - planning - 30, rng.gauss(DAY_START + 120, 90))))
            earliest = arrival
            deadline = None
        elif priority == "urgent":
            arrival = int(rng.uniform(DAY_START, DAY_START + 120))
            earliest = arrival
            deadline = None
        else:
            arrival = int(rng.uniform(DAY_START, DAY_START + 180))
            earliest = int(rng.uniform(DAY_START, max(DAY_START, DAY_END - planning - 60)))
            deadline = min(DAY_END, earliest + int(rng.uniform(180, 420)))

        preferred = [s.id for s in surgeons if s.specialty == spec]
        rng.shuffle(preferred)
        preferred = preferred[:1]

        cases.append(
            Case(
                id=f"C{i+1:03d}",
                procedure=proc,
                priority=priority,
                specialty=spec,
                required_features=req_feats,
                required_staff=dict(meta["staff"]),
                required_skills=set(meta["skills"]),
                earliest_start=earliest,
                deadline=deadline,
                mu=mu,
                sigma=sigma,
                planning_duration=planning,
                arrival_time=arrival,
                preferred_surgeons=preferred,
            )
        )

    cleaning = {p: meta["cleaning"] for p, meta in PROCEDURES.items()}
    return Instance(
        rooms=rooms,
        surgeons=surgeons,
        staff=staff,
        cases=cases,
        cleaning_by_procedure=cleaning,
        seed=seed,
    )


def make_emergency_case(instance: Instance, now: int, seed: Optional[int] = None) -> Case:
    rng = random.Random(seed if seed is not None else int(now * 1000) + len(instance.cases))
    np_rng = np.random.default_rng(rng.randint(0, 10**9))

    specialties = sorted({s.specialty for s in instance.surgeons})
    spec = rng.choice(specialties)
    eligible_procs = [
        p for p, meta in PROCEDURES.items()
        if meta["specialty"] == spec
        and any(r.features.issuperset(meta["features"]) for r in instance.rooms)
    ]
    if not eligible_procs:
        eligible_procs = list(PROCEDURES.keys())
    proc = rng.choice(eligible_procs)
    meta = PROCEDURES[proc]

    req_feats = set(meta["features"])
    if not any(r.features.issuperset(req_feats) for r in instance.rooms):
        req_feats = set()

    mu, sigma = lognormal_params_from_mean_sd(meta["mean"], meta["sd"])
    samples = np_rng.lognormal(mu, sigma, size=300)
    planning = int(math.ceil(float(np.quantile(samples, 0.8))))
    planning = max(25, min(planning, 300))
    earliest = max(DAY_START, min(now, DAY_END - planning - 15))

    preferred = [s.id for s in instance.surgeons if s.specialty == spec]
    rng.shuffle(preferred)

    return Case(
        id=f"EMG{len(instance.cases)+1:03d}",
        procedure=proc,
        priority="emergency",
        specialty=spec,
        required_features=req_feats,
        required_staff=dict(meta["staff"]),
        required_skills=set(meta["skills"]),
        earliest_start=earliest,
        deadline=None,
        mu=mu,
        sigma=sigma,
        planning_duration=planning,
        arrival_time=now,
        preferred_surgeons=preferred[:1],
    )


# ----------------------------------------------------------------------------
# Helper functions for dynamic availability
# ----------------------------------------------------------------------------
def room_effective_end(r: Room) -> int:
    if r.failed_after is None:
        return r.open_end
    return min(r.open_end, r.failed_after)


def surgeon_effective_windows(s: Surgeon, dur: int) -> List[Tuple[int, int]]:
    out = []
    for ws, we in s.windows:
        if s.unavailable_after is not None:
            we = min(we, s.unavailable_after)
        if we - ws >= dur:
            out.append((ws, we))
    return out


def staff_effective_end(stf: Staff) -> int:
    if stf.unavailable_after is None:
        return stf.shift_end
    return min(stf.shift_end, stf.unavailable_after)


def get_case(instance: Instance, cid: str) -> Optional[Case]:
    for c in instance.cases:
        if c.id == cid:
            return c
    return None


def get_room(instance: Instance, rid: str) -> Optional[Room]:
    for r in instance.rooms:
        if r.id == rid:
            return r
    return None


def get_surgeon(instance: Instance, sid: str) -> Optional[Surgeon]:
    for s in instance.surgeons:
        if s.id == sid:
            return s
    return None


def get_staff(instance: Instance, sid: str) -> Optional[Staff]:
    for s in instance.staff:
        if s.id == sid:
            return s
    return None


# ----------------------------------------------------------------------------
# CP-SAT model / solver (logical modules: model/, solver/)
# ----------------------------------------------------------------------------
def default_params() -> Dict[str, float]:
    return {
        "urgent_wait": 5.0,
        "emergency_wait": 8.0,
        "utilization": 2.0,
        "overtime": 3.0,
        "preference": 20.0,
        "deadline": 4.0,
        "movement": 12.0,
        "deferral": 800.0,
    }


def empty_schedule_df(instance: Instance) -> pd.DataFrame:
    rows = []
    for c in instance.cases:
        rows.append({
            "case_id": c.id,
            "procedure": c.procedure,
            "priority": c.priority,
            "specialty": c.specialty,
            "room": None,
            "surgeon": None,
            "staff": [],
            "start": None,
            "end": None,
            "duration": c.planning_duration,
            "earliest_start": c.earliest_start,
            "deadline": c.deadline,
            "scheduled": False,
        })
    return pd.DataFrame(rows)


def build_and_solve(
    instance: Instance,
    params: Dict[str, float],
    fixed: Optional[Dict[str, Dict[str, Any]]] = None,
    anchors: Optional[Dict[str, int]] = None,
    mode: str = "optimize",
    time_limit: float = 10.0,
    min_start: Optional[int] = None,
    availability_exempt: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    mode:
      - feasibility: hard constraints only
      - optimize: hard constraints + soft objectives
      - repair: hard constraints + minimal-change objective

    min_start:
      - used in repair to prevent unscheduled/future cases from being moved into the past

    availability_exempt:
      - cases already started before an event are fixed and exempt from event-based
        availability truncation, while still participating in no-overlap/cleaning constraints
    """
    fixed = fixed or {}
    anchors = anchors or {}
    availability_exempt = availability_exempt or set()
    min_start = safe_int(min_start)

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max(0.5, time_limit))
    solver.parameters.num_search_workers = 8

    case_map = {c.id: c for c in instance.cases}
    vars: Dict[str, Dict[str, Any]] = {}

    # Case-level variables
    for c in instance.cases:
        fx = fixed.get(c.id, {})
        dur = int(c.planning_duration)
        exempt = c.id in availability_exempt

        scheduled = model.NewBoolVar(f"sched_{c.id}")
        if "scheduled" in fx:
            model.Add(scheduled == (1 if fx["scheduled"] else 0))
        elif c.priority in ("emergency", "urgent"):
            model.Add(scheduled == 1)

        lock_start = bool(fx.get("lock_start")) and bool(fx.get("scheduled", True))
        if lock_start:
            model.Add(scheduled == 1)

        lower = c.earliest_start
        if min_start is not None and not lock_start:
            lower = max(lower, min_start)

        upper = DAY_END - dur
        if lock_start and fx.get("start") is not None:
            locked_start = safe_int(fx.get("start"))
            if locked_start is not None:
                lower = min(lower, locked_start)
                upper = max(upper, locked_start)
        upper = max(lower, upper)

        start = model.NewIntVar(lower, upper, f"start_{c.id}")
        end_upper = max(lower + dur + 1, DAY_END + 240)
        end = model.NewIntVar(lower + 1, end_upper, f"end_{c.id}")
        model.Add(end == start + dur)

        if lock_start and fx.get("start") is not None:
            locked_start = safe_int(fx.get("start"))
            if locked_start is not None:
                model.Add(start == locked_start)

        # Room assignment
        p: Dict[str, Any] = {}
        for r in instance.rooms:
            if not c.required_features.issubset(r.features):
                continue
            if not exempt and room_effective_end(r) < r.open_start + dur:
                continue

            lit = model.NewBoolVar(f"room_{c.id}_{r.id}")
            p[r.id] = lit

            if not exempt:
                model.Add(start >= r.open_start).OnlyEnforceIf(lit)
                model.Add(end <= room_effective_end(r)).OnlyEnforceIf(lit)

        if p:
            model.Add(sum(p.values()) == scheduled)
        else:
            model.Add(scheduled == 0)

        if fx.get("lock_assignments"):
            if fx.get("room") is not None:
                if fx["room"] in p:
                    for rid, lit in p.items():
                        model.Add(lit == (1 if rid == fx["room"] else 0))
                else:
                    model.Add(scheduled == 0)

        # Surgeon assignment
        a: Dict[str, Any] = {}
        for s in instance.surgeons:
            if s.specialty != c.specialty:
                continue

            if exempt:
                lit = model.NewBoolVar(f"surg_{c.id}_{s.id}")
                a[s.id] = lit
            else:
                wins = surgeon_effective_windows(s, dur)
                if not wins:
                    continue
                lit = model.NewBoolVar(f"surg_{c.id}_{s.id}")
                a[s.id] = lit
                wbs = []
                for ws, we in wins:
                    wb = model.NewBoolVar(f"surgwin_{c.id}_{s.id}_{ws}_{we}")
                    wbs.append(wb)
                    model.Add(start >= ws).OnlyEnforceIf(wb)
                    model.Add(end <= we).OnlyEnforceIf(wb)
                model.Add(sum(wbs) == lit)

        if a:
            model.Add(sum(a.values()) == scheduled)
        else:
            model.Add(scheduled == 0)

        if fx.get("lock_assignments"):
            if fx.get("surgeon") is not None:
                if fx["surgeon"] in a:
                    for sid, lit in a.items():
                        model.Add(lit == (1 if sid == fx["surgeon"] else 0))
                else:
                    model.Add(scheduled == 0)

        # Staff assignment
        t: Dict[str, Any] = {}
        role_lits: Dict[str, List[Any]] = defaultdict(list)
        for stf in instance.staff:
            req = c.required_staff.get(stf.role, 0)
            if req <= 0:
                continue
            if c.required_skills and not stf.skills.issuperset(c.required_skills):
                continue
            if not exempt and staff_effective_end(stf) < stf.shift_start + dur:
                continue

            lit = model.NewBoolVar(f"staff_{c.id}_{stf.id}")
            t[stf.id] = lit
            role_lits[stf.role].append(lit)

            if not exempt:
                model.Add(start >= stf.shift_start).OnlyEnforceIf(lit)
                model.Add(end <= staff_effective_end(stf)).OnlyEnforceIf(lit)

        for role, req in c.required_staff.items():
            if req <= 0:
                continue
            lits = role_lits.get(role, [])
            if lits:
                model.Add(sum(lits) == req * scheduled)
            else:
                model.Add(scheduled == 0)

        if fx.get("lock_assignments"):
            if fx.get("staff") is not None:
                locked_staff = set(as_list(fx["staff"]))
                for sid, lit in t.items():
                    model.Add(lit == (1 if sid in locked_staff else 0))

        vars[c.id] = {
            "scheduled": scheduled,
            "start": start,
            "end": end,
            "p": p,
            "a": a,
            "t": t,
            "dur": dur,
        }

    # Room no-overlap, cleaning, emergency-not-delayed-by-elective
    for r in instance.rooms:
        items = []
        for cid, cv in vars.items():
            if r.id in cv["p"]:
                c = case_map[cid]
                iv = model.NewOptionalIntervalVar(
                    cv["start"], c.planning_duration, cv["end"], cv["p"][r.id],
                    f"riv_{cid}_{r.id}"
                )
                items.append((cid, iv, cv["p"][r.id]))

        if items:
            model.AddNoOverlap([iv for _, iv, _ in items])

        for ii in range(len(items)):
            cid_i, _, lit_i = items[ii]
            ci = case_map[cid_i]
            for jj in range(ii + 1, len(items)):
                cid_j, _, lit_j = items[jj]
                cj = case_map[cid_j]
                order = model.NewBoolVar(f"ord_{r.id}_{cid_i}_{cid_j}")
                clean_i = instance.cleaning_by_procedure.get(ci.procedure, 30)
                clean_j = instance.cleaning_by_procedure.get(cj.procedure, 30)

                model.Add(vars[cid_i]["end"] + clean_i <= vars[cid_j]["start"]).OnlyEnforceIf(
                    [lit_i, lit_j, order]
                )
                model.Add(vars[cid_j]["end"] + clean_j <= vars[cid_i]["start"]).OnlyEnforceIf(
                    [lit_i, lit_j, order.Not()]
                )

                # Emergency must not be delayed by elective in same room
                if ci.priority == "emergency" and cj.priority == "elective":
                    ebit = model.NewBoolVar(f"emerg_{r.id}_{cid_i}_{cid_j}")
                    model.Add(vars[cid_i]["start"] <= vars[cid_j]["start"]).OnlyEnforceIf(
                        [lit_i, lit_j, ebit]
                    )
                    model.Add(vars[cid_j]["end"] + clean_j <= vars[cid_i]["start"]).OnlyEnforceIf(
                        [lit_i, lit_j, ebit.Not()]
                    )
                if cj.priority == "emergency" and ci.priority == "elective":
                    ebit = model.NewBoolVar(f"emerg_{r.id}_{cid_j}_{cid_i}")
                    model.Add(vars[cid_j]["start"] <= vars[cid_i]["start"]).OnlyEnforceIf(
                        [lit_i, lit_j, ebit]
                    )
                    model.Add(vars[cid_i]["end"] + clean_i <= vars[cid_j]["start"]).OnlyEnforceIf(
                        [lit_i, lit_j, ebit.Not()]
                    )

    # Surgeon no-overlap and daily max hours
    for s in instance.surgeons:
        intervals = []
        hour_terms = []
        for cid, cv in vars.items():
            if s.id in cv["a"]:
                c = case_map[cid]
                iv = model.NewOptionalIntervalVar(
                    cv["start"], c.planning_duration, cv["end"], cv["a"][s.id],
                    f"siv_{cid}_{s.id}"
                )
                intervals.append(iv)
                hour_terms.append(c.planning_duration * cv["a"][s.id])
        if intervals:
            model.AddNoOverlap(intervals)
        if hour_terms:
            model.Add(sum(hour_terms) <= int(s.max_hours * 60))

    # Staff no-overlap
    for stf in instance.staff:
        intervals = []
        for cid, cv in vars.items():
            if stf.id in cv["t"]:
                c = case_map[cid]
                iv = model.NewOptionalIntervalVar(
                    cv["start"], c.planning_duration, cv["end"], cv["t"][stf.id],
                    f"tiv_{cid}_{stf.id}"
                )
                intervals.append(iv)
        if intervals:
            model.AddNoOverlap(intervals)

    # Objective
    terms = []
    if mode != "feasibility":
        SCALE = 100.0

        def iw(x: float) -> int:
            return int(round(float(x) * SCALE))

        for cid, cv in vars.items():
            c = case_map[cid]

            if c.priority == "urgent" and params.get("urgent_wait", 0):
                terms.append(iw(params["urgent_wait"]) * (cv["start"] - c.earliest_start))
            if c.priority == "emergency" and params.get("emergency_wait", 0):
                terms.append(iw(params["emergency_wait"]) * (cv["start"] - c.earliest_start))

            if params.get("utilization", 0):
                terms.append(-iw(params["utilization"]) * c.planning_duration * cv["scheduled"])

            if c.priority == "elective" and params.get("deferral", 0):
                terms.append(iw(params["deferral"]) * (1 - cv["scheduled"]))

            if params.get("overtime", 0):
                ot = model.NewIntVar(0, 1000, f"ot_{cid}")
                model.Add(ot >= cv["end"] - TARGET_END)
                model.Add(ot >= 0)
                terms.append(iw(params["overtime"]) * ot)

            if params.get("deadline", 0) and c.deadline is not None:
                late = model.NewIntVar(0, 1000, f"late_{cid}")
                model.Add(late >= cv["end"] - c.deadline)
                model.Add(late >= 0)
                terms.append(iw(params["deadline"]) * late)

            if params.get("preference", 0) and c.preferred_surgeons:
                pref_expr = 0
                has_pref = False
                for sid in c.preferred_surgeons:
                    if sid in cv["a"]:
                        pref_expr += cv["a"][sid]
                        has_pref = True
                if has_pref:
                    terms.append(iw(params["preference"]) * (cv["scheduled"] - pref_expr))

        if mode == "repair" and params.get("movement", 0):
            for cid, old_start in anchors.items():
                if cid not in vars:
                    continue
                if fixed.get(cid, {}).get("lock_start"):
                    continue
                diff = model.NewIntVar(0, DAY_END, f"move_{cid}")
                old_start_i = safe_int(old_start)
                if old_start_i is None:
                    continue
                model.Add(diff >= vars[cid]["start"] - old_start_i)
                model.Add(diff >= old_start_i - vars[cid]["start"])
                terms.append(iw(params["movement"]) * diff)

    if terms:
        model.Minimize(sum(terms))
    else:
        model.Minimize(0)

    try:
        status = solver.Solve(model)
    except Exception as exc:
        return empty_schedule_df(instance), {
            "mode": mode,
            "status": f"ERROR: {exc}",
            "feasible": False,
            "optimal": False,
            "time": 0.0,
            "objective": None,
            "bound": None,
            "gap_abs": None,
            "gap_percent": None,
        }

    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    info = {
        "mode": mode,
        "status": solver.StatusName(status),
        "feasible": feasible,
        "optimal": status == cp_model.OPTIMAL,
        "time": float(solver.WallTime()),
        "objective": None,
        "bound": None,
        "gap_abs": None,
        "gap_percent": None,
    }

    if feasible:
        try:
            info["objective"] = float(solver.ObjectiveValue())
        except Exception:
            info["objective"] = None

        info["bound"] = solver_best_bound(solver)

        if info["objective"] is not None and info["bound"] is not None:
            info["gap_abs"] = abs(info["bound"] - info["objective"])
            info["gap_percent"] = 100.0 * info["gap_abs"] / max(1.0, abs(info["objective"]))

    if not feasible:
        return empty_schedule_df(instance), info

    rows = []
    for c in instance.cases:
        cv = vars[c.id]
        sched = bool(solver.Value(cv["scheduled"]))
        room = None
        surgeon = None
        staff_assigned = []
        start = None
        end = None

        if sched:
            room = next((rid for rid, lit in cv["p"].items() if solver.Value(lit)), None)
            surgeon = next((sid for sid, lit in cv["a"].items() if solver.Value(lit)), None)
            staff_assigned = sorted([sid for sid, lit in cv["t"].items() if solver.Value(lit)])
            start = int(solver.Value(cv["start"]))
            end = int(solver.Value(cv["end"]))

        rows.append({
            "case_id": c.id,
            "procedure": c.procedure,
            "priority": c.priority,
            "specialty": c.specialty,
            "room": room,
            "surgeon": surgeon,
            "staff": staff_assigned,
            "start": start,
            "end": end,
            "duration": c.planning_duration,
            "earliest_start": c.earliest_start,
            "deadline": c.deadline,
            "scheduled": sched,
        })

    return pd.DataFrame(rows), info


def solve_two_stage(
    instance: Instance,
    params: Dict[str, float],
    time_limit: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    t0 = time.time()
    feas_df, feas_info = build_and_solve(
        instance, params, mode="feasibility", time_limit=max(1.0, time_limit * 0.30)
    )
    elapsed = time.time() - t0
    opt_df, opt_info = build_and_solve(
        instance, params, mode="optimize", time_limit=max(1.0, time_limit - elapsed * 0.30)
    )
    info = {
        "feasibility": feas_info,
        "optimization": opt_info,
        "feasible": bool(opt_info["feasible"] or feas_info["feasible"]),
        "total_time": time.time() - t0,
    }
    df = opt_df if opt_info["feasible"] else feas_df
    return df, info


# ----------------------------------------------------------------------------
# Independent brute-force / hard-constraint checker
# ----------------------------------------------------------------------------
def check_hard_constraints(
    instance: Instance,
    schedule_df: pd.DataFrame,
    required_must_be_scheduled: bool = True,
    availability_exempt: Optional[Set[str]] = None,
) -> List[str]:
    violations: List[str] = []
    availability_exempt = availability_exempt or set()

    if schedule_df is None or schedule_df.empty:
        return ["No schedule provided."]

    records = {r["case_id"]: r for r in schedule_df.to_dict("records")}
    case_map = {c.id: c for c in instance.cases}

    for c in instance.cases:
        if c.id not in records:
            violations.append(f"{c.id}: missing from schedule")
            continue
        row = records[c.id]
        sched = bool(row.get("scheduled", False))
        exempt = c.id in availability_exempt

        if required_must_be_scheduled and c.priority in ("emergency", "urgent") and not sched:
            violations.append(f"{c.id}: required {c.priority} case deferred")
        if not sched:
            continue

        start = safe_int(row.get("start"))
        end = safe_int(row.get("end"))
        if start is None or end is None:
            violations.append(f"{c.id}: missing start/end")
            continue

        if start < c.earliest_start:
            violations.append(f"{c.id}: starts before earliest start")

        room = get_room(instance, row.get("room"))
        if room is None:
            violations.append(f"{c.id}: invalid room")
        else:
            if not c.required_features.issubset(room.features):
                violations.append(f"{c.id}: room lacks required equipment")
            if not exempt:
                if start < room.open_start or end > room_effective_end(room):
                    violations.append(f"{c.id}: outside room availability")
                if room.failed_after is not None and end > room.failed_after:
                    violations.append(f"{c.id}: uses room after equipment failure")

        surgeon = get_surgeon(instance, row.get("surgeon"))
        if surgeon is None:
            violations.append(f"{c.id}: invalid surgeon")
        else:
            if surgeon.specialty != c.specialty:
                violations.append(f"{c.id}: surgeon specialty mismatch")
            if not exempt:
                ok_window = False
                for ws, we in surgeon.windows:
                    if surgeon.unavailable_after is not None:
                        we = min(we, surgeon.unavailable_after)
                    if start >= ws and end <= we:
                        ok_window = True
                        break
                if not ok_window:
                    violations.append(f"{c.id}: surgeon availability violated")

        assigned_staff = as_list(row.get("staff"))
        role_counts = defaultdict(int)
        for sid in assigned_staff:
            stf = get_staff(instance, sid)
            if stf is None:
                violations.append(f"{c.id}: invalid staff {sid}")
                continue
            role_counts[stf.role] += 1
            if c.required_skills and not stf.skills.issuperset(c.required_skills):
                violations.append(f"{c.id}: staff {sid} lacks required skill")
            if not exempt:
                if start < stf.shift_start or end > staff_effective_end(stf):
                    violations.append(f"{c.id}: staff {sid} shift violated")
                if stf.unavailable_after is not None and end > stf.unavailable_after:
                    violations.append(f"{c.id}: staff {sid} unavailable violation")

        for role, req in c.required_staff.items():
            if role_counts.get(role, 0) != req:
                violations.append(f"{c.id}: staff role {role} count mismatch")

    # Room sequencing, cleaning, emergency precedence
    for r in instance.rooms:
        items = []
        for cid, row in records.items():
            if not row.get("scheduled"):
                continue
            if row.get("room") != r.id:
                continue
            c = case_map.get(cid)
            if c is None:
                continue
            s = safe_int(row.get("start"))
            e = safe_int(row.get("end"))
            if s is None or e is None:
                continue
            items.append((s, e, cid, c.procedure))
        items.sort()

        for i in range(len(items) - 1):
            s1, e1, id1, proc1 = items[i]
            s2, e2, id2, proc2 = items[i + 1]
            if e1 > s2:
                violations.append(f"Room {r.id}: overlap between {id1} and {id2}")
            clean1 = instance.cleaning_by_procedure.get(proc1, 30)
            if e1 + clean1 > s2:
                violations.append(f"Room {r.id}: cleaning violation between {id1} and {id2}")

        emergencies = [x for x in items if case_map[x[2]].priority == "emergency"]
        electives = [x for x in items if case_map[x[2]].priority == "elective"]
        for es, ee, eid, _ in emergencies:
            for xs, xe, xid, xproc in electives:
                if xs < es:
                    clean_x = instance.cleaning_by_procedure.get(xproc, 30)
                    if xe + clean_x > es:
                        violations.append(
                            f"Room {r.id}: elective {xid} delays emergency {eid}"
                        )

    # Surgeon overlap and max hours
    for s in instance.surgeons:
        intervals = []
        total = 0
        for cid, row in records.items():
            if row.get("scheduled") and row.get("surgeon") == s.id:
                st = safe_int(row.get("start"))
                en = safe_int(row.get("end"))
                du = safe_int(row.get("duration"))
                if st is None or en is None or du is None:
                    continue
                intervals.append((st, en, cid))
                total += du
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                violations.append(f"Surgeon {s.id}: double booking")
        if total > int(s.max_hours * 60):
            violations.append(f"Surgeon {s.id}: max daily hours exceeded")

    # Staff overlap
    for stf in instance.staff:
        intervals = []
        for cid, row in records.items():
            if not row.get("scheduled"):
                continue
            if stf.id in as_list(row.get("staff")):
                st = safe_int(row.get("start"))
                en = safe_int(row.get("end"))
                if st is None or en is None:
                    continue
                intervals.append((st, en, cid))
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                violations.append(f"Staff {stf.id}: double booking")

    return violations


# ----------------------------------------------------------------------------
# Baselines: greedy first-fit and rule-based scheduler (logical module: eval/)
# ----------------------------------------------------------------------------
def greedy_schedule(instance: Instance, mode: str = "first_fit") -> pd.DataFrame:
    cases = list(instance.cases)
    if mode == "first_fit":
        cases.sort(key=lambda c: (PRIORITY_RANK.get(c.priority, 9), c.earliest_start))
    else:
        cases.sort(key=lambda c: (
            PRIORITY_RANK.get(c.priority, 9),
            c.deadline if c.deadline is not None else 10**9,
            -c.planning_duration,
        ))

    room_busy: Dict[str, List[Tuple[int, int, str]]] = {r.id: [] for r in instance.rooms}
    surgeon_busy: Dict[str, List[Tuple[int, int]]] = {s.id: [] for s in instance.surgeons}
    staff_busy: Dict[str, List[Tuple[int, int]]] = {st.id: [] for st in instance.staff}
    surgeon_hours: Dict[str, int] = {s.id: 0 for s in instance.surgeons}

    rows = []

    def room_ok(rid: str, start: int, end: int, c: Case) -> bool:
        clean_new = instance.cleaning_by_procedure.get(c.procedure, 30)
        for s, e, proc in room_busy[rid]:
            clean_old = instance.cleaning_by_procedure.get(proc, 30)
            if not (end + clean_new <= s or e + clean_old <= start):
                return False
        return True

    def resource_overlap(busy: List[Tuple[int, int]], start: int, end: int) -> bool:
        for s, e in busy:
            if not (end <= s or e <= start):
                return True
        return False

    for c in cases:
        assigned = False
        candidate_rooms = [r for r in instance.rooms if c.required_features.issubset(r.features)]
        if mode == "rule":
            candidate_rooms.sort(key=lambda r: len(r.features - c.required_features))

        for r in candidate_rooms:
            if assigned:
                break
            eff_end = room_effective_end(r)
            latest = min(DAY_END - c.planning_duration, eff_end - c.planning_duration)
            for start in range(c.earliest_start, max(c.earliest_start, latest) + 1, 5):
                end = start + c.planning_duration
                if not room_ok(r.id, start, end, c):
                    continue

                surg_candidates = [
                    s for s in instance.surgeons
                    if s.specialty == c.specialty
                    and surgeon_hours[s.id] + c.planning_duration <= int(s.max_hours * 60)
                    and not resource_overlap(surgeon_busy[s.id], start, end)
                    and any(
                        ws <= start and end <= min(we, s.unavailable_after or DAY_END)
                        for ws, we in s.windows
                    )
                ]
                if mode == "rule":
                    surg_candidates.sort(key=lambda s: (
                        0 if s.id in c.preferred_surgeons else 1,
                        surgeon_hours[s.id],
                    ))
                if not surg_candidates:
                    continue
                surgeon = surg_candidates[0]

                staff_assign = []
                ok_staff = True
                for role, req in c.required_staff.items():
                    cand = [
                        st for st in instance.staff
                        if st.role == role
                        and (not c.required_skills or st.skills.issuperset(c.required_skills))
                        and st.shift_start <= start
                        and end <= staff_effective_end(st)
                        and not resource_overlap(staff_busy[st.id], start, end)
                    ]
                    if mode == "rule":
                        cand.sort(key=lambda st: staff_effective_end(st))
                    if len(cand) < req:
                        ok_staff = False
                        break
                    staff_assign.extend([x.id for x in cand[:req]])
                if not ok_staff:
                    continue

                room_busy[r.id].append((start, end, c.procedure))
                surgeon_busy[surgeon.id].append((start, end))
                surgeon_hours[surgeon.id] += c.planning_duration
                for sid in staff_assign:
                    staff_busy[sid].append((start, end))

                rows.append({
                    "case_id": c.id,
                    "procedure": c.procedure,
                    "priority": c.priority,
                    "specialty": c.specialty,
                    "room": r.id,
                    "surgeon": surgeon.id,
                    "staff": staff_assign,
                    "start": start,
                    "end": end,
                    "duration": c.planning_duration,
                    "earliest_start": c.earliest_start,
                    "deadline": c.deadline,
                    "scheduled": True,
                })
                assigned = True
                break

        if not assigned:
            rows.append({
                "case_id": c.id,
                "procedure": c.procedure,
                "priority": c.priority,
                "specialty": c.specialty,
                "room": None,
                "surgeon": None,
                "staff": [],
                "start": None,
                "end": None,
                "duration": c.planning_duration,
                "earliest_start": c.earliest_start,
                "deadline": c.deadline,
                "scheduled": False,
            })

    order = {c.id: i for i, c in enumerate(instance.cases)}
    rows.sort(key=lambda x: order.get(x["case_id"], 999))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Metrics and evaluation
# ----------------------------------------------------------------------------
def compute_metrics(
    instance: Instance,
    df: pd.DataFrame,
    method: str,
    solver_time: Optional[float] = None,
    gap_percent: Optional[float] = None,
    required_must_be_scheduled: bool = True,
    availability_exempt: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if df is None or df.empty:
        sched = pd.DataFrame()
    else:
        sched = df[df["scheduled"]].dropna(subset=["start", "end"]).copy()

    total_room_minutes = sum(room_effective_end(r) - r.open_start for r in instance.rooms)
    occupied = float(sched["duration"].sum()) if not sched.empty else 0.0
    utilization = occupied / total_room_minutes if total_room_minutes else 0.0

    urgent = sched[sched["priority"] == "urgent"] if not sched.empty else sched
    urgent_wait = 0.0
    if not urgent.empty:
        waits = urgent["start"].astype(float) - urgent["earliest_start"].astype(float)
        urgent_wait = float(waits.mean())

    on_time_rate = 0.0
    overtime_hours = 0.0
    if not sched.empty:
        def on_time_row(row):
            s = safe_int(row.get("start"))
            e = safe_int(row.get("end"))
            dl = row.get("deadline")
            es = safe_int(row.get("earliest_start"))
            if s is None or e is None:
                return False
            if dl is not None and not (isinstance(dl, float) and math.isnan(dl)):
                return e <= int(dl)
            if es is None:
                return False
            return s <= es + 30

        on_time_rate = float(sched.apply(on_time_row, axis=1).mean())
        overtime_hours = float(np.maximum(0, sched["end"].astype(float) - TARGET_END).sum() / 60.0)

    deferred = int((~df["scheduled"]).sum()) if df is not None and not df.empty else len(instance.cases)
    violations = check_hard_constraints(
        instance,
        df,
        required_must_be_scheduled=required_must_be_scheduled,
        availability_exempt=availability_exempt,
    )

    return {
        "method": method,
        "on_time_start_rate": on_time_rate,
        "urgent_wait_min": urgent_wait,
        "room_utilization": utilization,
        "overtime_hours": overtime_hours,
        "deferred_cases": deferred,
        "solver_time_sec": solver_time if solver_time is not None else np.nan,
        "optimality_gap_pct": gap_percent if gap_percent is not None else np.nan,
        "hard_violations": len(violations),
    }


def create_random_event(instance: Instance, schedule_df: pd.DataFrame, now: int, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    if schedule_df is None or schedule_df.empty:
        sched = pd.DataFrame()
    else:
        sched = schedule_df[schedule_df["scheduled"]].copy()

    types = ["emergency", "overrun", "surgeon_unavailable", "equipment_failure"]
    etype = rng.choice(types)

    if etype == "emergency":
        return {"type": "emergency", "time": now, "description": f"Emergency arrival at {fmt_min(now)}"}

    if etype == "overrun" and not sched.empty:
        future = sched[sched["start"] > now]
        pick = future if not future.empty else sched
        row = pick.iloc[rng.randrange(len(pick))]
        extra = rng.choice([15, 20, 30, 45])
        return {
            "type": "overrun",
            "time": now,
            "case_id": row["case_id"],
            "extra": extra,
            "description": f"Overrun +{extra} min on {row['case_id']}",
        }

    if etype == "surgeon_unavailable" and not sched.empty:
        future = sched[sched["start"] > now]
        if not future.empty:
            sid = future.iloc[rng.randrange(len(future))]["surgeon"]
        else:
            sid = instance.surgeons[rng.randrange(len(instance.surgeons))].id
        return {
            "type": "surgeon_unavailable",
            "time": now,
            "surgeon_id": sid,
            "description": f"Surgeon {sid} unavailable from {fmt_min(now)}",
        }

    if not sched.empty:
        future = sched[sched["start"] > now]
        if not future.empty:
            rid = future.iloc[rng.randrange(len(future))]["room"]
        else:
            rid = instance.rooms[rng.randrange(len(instance.rooms))].id
    else:
        rid = instance.rooms[rng.randrange(len(instance.rooms))].id

    return {
        "type": "equipment_failure",
        "time": now,
        "room_id": rid,
        "description": f"Equipment failure in {rid} at {fmt_min(now)}",
    }


# ----------------------------------------------------------------------------
# Dynamic repair (logical modules: events/, repair/)
# ----------------------------------------------------------------------------
def inject_event(
    instance: Instance,
    old_df: pd.DataFrame,
    event: Dict[str, Any],
    params: Dict[str, float],
    threshold: int = 30,
    auto_confirm: bool = False,
    time_limit: float = 8.0,
) -> Dict[str, Any]:
    t0 = time.time()
    inst = copy.deepcopy(instance)
    now = int(event["time"])

    if old_df is None or old_df.empty:
        old_records = {}
    else:
        old_records = {r["case_id"]: r for r in old_df.to_dict("records")}

    case_map = {c.id: c for c in inst.cases}

    started_ids: Set[str] = {
        cid for cid, rec in old_records.items()
        if rec.get("scheduled") and (safe_int(rec.get("start")) or 0) < now
    }

    affected: Set[str] = set()
    reasons: List[str] = []

    if event["type"] == "emergency":
        new_case = make_emergency_case(
            inst,
            now,
            seed=(now * 1000003 + len(inst.cases)) % 1000000007,
        )
        inst.cases.append(new_case)
        case_map[new_case.id] = new_case
        affected.add(new_case.id)
        event = dict(event)
        event["case_id"] = new_case.id
        reasons.append(f"New emergency {new_case.id} arrived")

    elif event["type"] == "overrun":
        cid = event.get("case_id")
        c = case_map.get(cid) if cid is not None else None
        if c is not None:
            extra = int(event.get("extra", 20))
            old_dur = c.planning_duration
            c.planning_duration = int(old_dur + extra)
            affected.add(cid)
            reasons.append(f"Case {cid} duration extended by {extra} min")

    elif event["type"] == "surgeon_unavailable":
        sid = event.get("surgeon_id")
        s = get_surgeon(inst, sid) if sid is not None else None
        if s is not None:
            s.unavailable_after = now
            for cid, rec in old_records.items():
                if (
                    rec.get("scheduled")
                    and (safe_int(rec.get("start")) or 0) >= now
                    and (safe_int(rec.get("end")) or 0) > now
                    and rec.get("surgeon") == sid
                ):
                    affected.add(cid)
            reasons.append(f"Surgeon {sid} unavailable from {fmt_min(now)}")

    elif event["type"] == "equipment_failure":
        rid = event.get("room_id")
        r = get_room(inst, rid) if rid is not None else None
        if r is not None:
            r.failed_after = now
            for cid, rec in old_records.items():
                if (
                    rec.get("scheduled")
                    and (safe_int(rec.get("start")) or 0) >= now
                    and (safe_int(rec.get("end")) or 0) > now
                    and rec.get("room") == rid
                ):
                    affected.add(cid)
            reasons.append(f"Room {rid} equipment failure from {fmt_min(now)}")

    def row_invalidated(rec: Dict[str, Any]) -> bool:
        if not rec.get("scheduled"):
            return False
        if (safe_int(rec.get("start")) or 0) < now:
            return False
        if (safe_int(rec.get("end")) or 0) <= now:
            return False

        if event["type"] == "equipment_failure" and rec.get("room") == event.get("room_id"):
            return True
        if event["type"] == "surgeon_unavailable" and rec.get("surgeon") == event.get("surgeon_id"):
            return True
        return False

    def expand_affected(current: Set[str]) -> Set[str]:
        new = set(current)
        batch = 3

        if event["type"] == "emergency" and event.get("case_id"):
            nc = case_map.get(event["case_id"])
            if nc is not None:
                compat_rooms = {r.id for r in inst.rooms if nc.required_features.issubset(r.features)}
                compat_surgeons = {s.id for s in inst.surgeons if s.specialty == nc.specialty}
                cand = []
                for cid, rec in old_records.items():
                    if not rec.get("scheduled") or (safe_int(rec.get("start")) or 0) < now:
                        continue
                    if rec.get("room") in compat_rooms or rec.get("surgeon") in compat_surgeons:
                        cand.append(cid)
                cand.sort(key=lambda cid: (
                    0 if case_map[cid].priority == "elective" else 1,
                    safe_int(old_records[cid].get("start")) or 0,
                ))
                for cid in cand[:batch * 3]:
                    new.add(cid)

        for cid in list(current):
            rec = old_records.get(cid)
            if not rec or not rec.get("scheduled"):
                continue
            if (safe_int(rec.get("start")) or 0) < now:
                continue

            def add_same(field_name: str, value: Any):
                cand = [
                    xid for xid, xrec in old_records.items()
                    if xrec.get("scheduled")
                    and (safe_int(xrec.get("start")) or 0) >= now
                    and xrec.get(field_name) == value
                    and xid not in new
                ]
                cand.sort(key=lambda xid: safe_int(old_records[xid].get("start")) or 0)
                for xid in cand[:batch]:
                    new.add(xid)

            add_same("room", rec.get("room"))
            add_same("surgeon", rec.get("surgeon"))

            for stf_id in as_list(rec.get("staff")):
                cand = [
                    xid for xid, xrec in old_records.items()
                    if xrec.get("scheduled")
                    and (safe_int(xrec.get("start")) or 0) >= now
                    and stf_id in as_list(xrec.get("staff"))
                    and xid not in new
                ]
                cand.sort(key=lambda xid: safe_int(old_records[xid].get("start")) or 0)
                for xid in cand[:batch]:
                    new.add(xid)

        if new == current:
            future_elective = [
                cid for cid, rec in old_records.items()
                if rec.get("scheduled")
                and (safe_int(rec.get("start")) or 0) >= now
                and case_map.get(cid)
                and case_map[cid].priority == "elective"
            ]
            if future_elective:
                new.update(future_elective)
            else:
                new.update(
                    cid for cid, rec in old_records.items()
                    if rec.get("scheduled") and (safe_int(rec.get("start")) or 0) >= now
                )
        return new

    new_df = None
    info = None

    for attempt in range(5):
        fixed: Dict[str, Dict[str, Any]] = {}
        anchors: Dict[str, int] = {}

        for cid, rec in old_records.items():
            if rec.get("scheduled"):
                st = safe_int(rec.get("start"))
                if st is not None:
                    anchors[cid] = st

            if cid in affected:
                if rec.get("scheduled") and (safe_int(rec.get("start")) or 0) < now:
                    fixed[cid] = {
                        "start": safe_int(rec.get("start")),
                        "room": rec.get("room"),
                        "surgeon": rec.get("surgeon"),
                        "staff": as_list(rec.get("staff")),
                        "lock_start": True,
                        "lock_assignments": True,
                        "scheduled": True,
                    }
                else:
                    # Future affected case is fully free.
                    pass
            else:
                if rec.get("scheduled"):
                    fixed[cid] = {
                        "start": safe_int(rec.get("start")),
                        "room": rec.get("room"),
                        "surgeon": rec.get("surgeon"),
                        "staff": as_list(rec.get("staff")),
                        "lock_start": True,
                        "lock_assignments": True,
                        "scheduled": True,
                    }
                else:
                    fixed[cid] = {"scheduled": False}

        new_df, info = build_and_solve(
            inst,
            params,
            fixed=fixed,
            anchors=anchors,
            mode="repair",
            time_limit=max(1.0, time_limit),
            min_start=now,
            availability_exempt=started_ids,
        )
        if info["feasible"]:
            break

        expanded = expand_affected(affected)
        if expanded == affected:
            break
        affected = expanded

    repair_time = time.time() - t0

    if new_df is None or info is None or not info["feasible"]:
        return {
            "applied": False,
            "needs_confirm": False,
            "new_instance": inst,
            "new_schedule": old_df,
            "old_schedule": old_df,
            "changes": pd.DataFrame(),
            "log": reasons + ["Repair infeasible within local search space"],
            "repair_time": repair_time,
            "info": info or {"feasible": False, "status": "INFEASIBLE"},
            "violations": ["Repair infeasible"],
            "event": event,
            "availability_exempt": started_ids,
        }

    new_records = {r["case_id"]: r for r in new_df.to_dict("records")}
    change_rows = []
    needs_confirm = False

    all_cids = sorted(set(old_records.keys()) | set(new_records.keys()))
    for cid in all_cids:
        old = old_records.get(cid)
        new = new_records.get(cid)

        old_sched = bool(old and old.get("scheduled", False))
        new_sched = bool(new and new.get("scheduled", False))

        old_start = safe_int(old.get("start")) if old_sched else None
        new_start = safe_int(new.get("start")) if new_sched else None
        delta = (new_start - old_start) if (old_start is not None and new_start is not None) else None

        old_dur = safe_int(old.get("duration")) if old else None
        new_dur = safe_int(new.get("duration")) if new else None
        duration_change = old_dur is not None and new_dur is not None and old_dur != new_dur

        moved = delta is not None and abs(delta) > threshold
        status_change = old_sched != new_sched or old is None or new is None
        assignment_change = False
        if old_sched and new_sched:
            assignment_change = (
                old.get("room") != new.get("room")
                or old.get("surgeon") != new.get("surgeon")
                or set(as_list(old.get("staff"))) != set(as_list(new.get("staff")))
            )

        if moved or status_change or assignment_change or duration_change or old is None:
            reason_parts = [event.get("description", "Event")] + reasons
            if old is None:
                reason_parts.append("new case inserted")
            if status_change:
                reason_parts.append("scheduled status changed")
            if moved:
                reason_parts.append(f"start moved {delta:+d} min")
            if assignment_change:
                reason_parts.append("room/surgeon/staff reassigned")
            if duration_change:
                reason_parts.append(f"duration changed {old_dur}->{new_dur}")

            change_rows.append({
                "case_id": cid,
                "old_start": old_start,
                "new_start": new_start,
                "delta_min": delta,
                "old_room": old.get("room") if old else None,
                "new_room": new.get("room") if new else None,
                "old_surgeon": old.get("surgeon") if old else None,
                "new_surgeon": new.get("surgeon") if new else None,
                "old_duration": old_dur,
                "new_duration": new_dur,
                "reason": "; ".join(reason_parts),
            })

            if moved or (old_sched and not new_sched):
                needs_confirm = True

    changes_df = pd.DataFrame(change_rows)
    violations = check_hard_constraints(
        inst,
        new_df,
        required_must_be_scheduled=True,
        availability_exempt=started_ids,
    )

    applied = (not needs_confirm) or auto_confirm
    return {
        "applied": applied,
        "needs_confirm": needs_confirm,
        "new_instance": inst,
        "new_schedule": new_df,
        "old_schedule": old_df,
        "changes": changes_df,
        "log": reasons,
        "repair_time": repair_time,
        "info": info,
        "violations": violations,
        "event": event,
        "availability_exempt": started_ids,
    }


# ----------------------------------------------------------------------------
# Visualization: Gantt and live demo video/animation
# ----------------------------------------------------------------------------
def make_gantt(
    instance: Instance,
    schedule_df: pd.DataFrame,
    old_schedule_df: Optional[pd.DataFrame] = None,
    event_times: Optional[List[int]] = None,
) -> go.Figure:
    fig = go.Figure()
    if schedule_df is None or schedule_df.empty:
        fig.update_layout(title="No scheduled cases")
        return fig

    sched = schedule_df[schedule_df["scheduled"]].dropna(subset=["start", "end", "room"]).copy()
    if sched.empty:
        fig.update_layout(title="No scheduled cases")
        return fig

    room_order = [r.name for r in instance.rooms]
    room_id_to_name = {r.id: r.name for r in instance.rooms}
    sched["room_name"] = sched["room"].map(room_id_to_name)

    if old_schedule_df is not None and not old_schedule_df.empty:
        old = old_schedule_df[old_schedule_df["scheduled"]].dropna(subset=["start", "end", "room"]).copy()
        if not old.empty:
            old["room_name"] = old["room"].map(room_id_to_name)
            fig.add_trace(go.Bar(
                x=old["end"] - old["start"],
                base=old["start"],
                y=old["room_name"],
                orientation="h",
                name="Previous",
                marker_color="rgba(200, 30, 30, 0.18)",
                opacity=0.55,
                hovertemplate="Previous<br>%{customdata}<extra></extra>",
                customdata=old["case_id"],
            ))

    fig.add_trace(go.Bar(
        x=sched["end"] - sched["start"],
        base=sched["start"],
        y=sched["room_name"],
        orientation="h",
        name="Current",
        marker_color=[PRIORITY_COLORS.get(p, "#888") for p in sched["priority"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Room: %{y}<br>"
            "Start: %{customdata[1]}<br>"
            "End: %{customdata[2]}<br>"
            "Priority: %{customdata[3]}<br>"
            "Surgeon: %{customdata[4]}"
            "<extra></extra>"
        ),
        customdata=list(zip(
            sched["case_id"],
            sched["start"].map(fmt_min),
            sched["end"].map(fmt_min),
            sched["priority"],
            sched["surgeon"],
        )),
    ))

    tick_vals = list(range(DAY_START, DAY_END + 1, 60))
    fig.update_layout(
        barmode="overlay",
        title="TheatreSched Gantt (change highlights in red)",
        xaxis=dict(
            title="Time of day",
            range=[DAY_START, DAY_END],
            tickvals=tick_vals,
            ticktext=[fmt_min(t) for t in tick_vals],
        ),
        yaxis=dict(categoryorder="array", categoryarray=room_order[::-1]),
        height=max(320, 70 * len(instance.rooms)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=120, r=30, t=60, b=50),
    )

    for et in (event_times or []):
        et_i = safe_int(et)
        if et_i is None:
            continue
        fig.add_vline(
            x=et_i,
            line_dash="dash",
            line_color="orange",
            annotation_text=fmt_min(et_i),
            annotation_position="top",
        )

    return fig


def make_demo_video(
    instance: Instance,
    schedule_df: pd.DataFrame,
    events: Optional[List[Dict[str, Any]]] = None,
    step: int = 30,
    fps: int = 5,
) -> Optional[Dict[str, Any]]:
    if schedule_df is None or schedule_df.empty:
        return None

    sched = schedule_df[schedule_df["scheduled"]].dropna(subset=["start", "end", "room"]).copy()
    if sched.empty:
        return None

    events = events or []
    width, height = 980, 110 + 58 * len(instance.rooms)
    left, right, top = 110, 30, 70
    plot_w = width - left - right
    span = max(1, DAY_END - DAY_START)
    font = ImageFont.load_default()
    frames = []

    room_y = {r.id: top + i * 58 for i, r in enumerate(instance.rooms)}

    for current in range(DAY_START, DAY_END + 1, step):
        img = Image.new("RGB", (width, height), "white")
        d = ImageDraw.Draw(img)
        d.text((15, 15), "TheatreSched live demo", fill="black", font=font)
        d.text((15, 35), f"Time: {fmt_min(current)}", fill="black", font=font)

        for h in range(DAY_START, DAY_END + 1, 60):
            x = left + int((h - DAY_START) / span * plot_w)
            d.line([(x, top - 15), (x, height - 30)], fill=(230, 230, 230), width=1)
            d.text((x - 12, height - 25), fmt_min(h), fill="black", font=font)

        for r in instance.rooms:
            y = room_y[r.id]
            d.text((10, y + 12), r.name, fill="black", font=font)
            d.line([(left, y + 20), (width - right, y + 20)], fill=(210, 210, 210), width=1)

        for _, row in sched.iterrows():
            s = safe_int(row.get("start"))
            e = safe_int(row.get("end"))
            if s is None or e is None:
                continue

            x0 = left + int((s - DAY_START) / span * plot_w)
            x1 = left + int((e - DAY_START) / span * plot_w)
            y = room_y.get(row.get("room"), top)
            bar_h = 28

            if e <= current:
                fill = (190, 190, 190)
            elif s <= current < e:
                col = PRIORITY_COLORS.get(row.get("priority", "elective"), "#888")
                try:
                    fill = tuple(int(col.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
                except Exception:
                    fill = (120, 120, 120)
            else:
                fill = (235, 235, 245)

            d.rectangle([x0, y + 6, x1, y + 6 + bar_h], outline=(80, 80, 80), fill=fill)
            if x1 - x0 > 34:
                d.text((x0 + 3, y + 12), str(row.get("case_id", "")), fill="black", font=font)

        for ev in events:
            et = safe_int(ev.get("time"))
            if et is None or et < DAY_START or et > DAY_END:
                continue
            x = left + int((et - DAY_START) / span * plot_w)
            d.line([(x, top - 25), (x, height - 35)], fill=(255, 140, 0), width=2)
            d.text((x - 20, top - 42), str(ev.get("type", "event"))[:8], fill=(180, 80, 0), font=font)

        xc = left + int((current - DAY_START) / span * plot_w)
        d.line([(xc, top - 20), (xc, height - 30)], fill=(220, 20, 60), width=2)

        frames.append(np.array(img))

    if not frames:
        return None

    if imageio is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                path = tmp.name
            imageio.mimsave(path, frames, fps=fps)
            with open(path, "rb") as f:
                data = f.read()
            os.unlink(path)
            return {"mime": "video/mp4", "bytes": data, "ext": "mp4"}
        except Exception:
            pass

    try:
        buf = io.BytesIO()
        imgs = [Image.fromarray(fr) for fr in frames]
        imgs[0].save(
            buf,
            format="GIF",
            append_images=imgs[1:],
            save_all=True,
            duration=int(1000 / max(1, fps)),
            loop=0,
        )
        return {"mime": "image/gif", "bytes": buf.getvalue(), "ext": "gif"}
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Tests (logical module: eval/tests)
# ----------------------------------------------------------------------------
def run_property_tests(n_random: int = 5, time_limit: float = 3.0) -> pd.DataFrame:
    results = []
    params = default_params()

    inst = generate_instance(2, 10, seed=123, emergency_rate=0.15)
    df, info = solve_two_stage(inst, params, time_limit=12.0)
    viol = check_hard_constraints(inst, df, required_must_be_scheduled=True)
    results.append({
        "test": "small_2_rooms_10_cases_exact",
        "passed": info["optimization"].get("status") == "OPTIMAL" and len(viol) == 0,
        "details": (
            f"status={info['optimization'].get('status')}, "
            f"violations={len(viol)}, "
            f"gap={fmt_num(info['optimization'].get('gap_percent'), 2, '%')}"
        ),
    })

    if df is not None and not df.empty:
        corrupt = df.copy()
        sched_rows = corrupt[corrupt["scheduled"]].index
        if len(sched_rows) >= 2:
            i0, i1 = sched_rows[0], sched_rows[1]
            corrupt.loc[i1, "start"] = corrupt.loc[i0, "start"]
            corrupt.loc[i1, "end"] = corrupt.loc[i1, "start"] + corrupt.loc[i1, "duration"]
            viol2 = check_hard_constraints(inst, corrupt, required_must_be_scheduled=False)
            results.append({
                "test": "checker_detects_artificial_overlap",
                "passed": len(viol2) > 0,
                "details": f"detected {len(viol2)} violations",
            })

    for k in range(n_random):
        seed = 5000 + k
        rooms = random.choice([2, 3])
        cases = random.randint(5, 9)
        inst_r = generate_instance(rooms, cases, seed=seed, emergency_rate=0.18)
        df_r, info_r = build_and_solve(inst_r, params, mode="optimize", time_limit=time_limit)
        if info_r["feasible"]:
            vr = check_hard_constraints(inst_r, df_r, required_must_be_scheduled=True)
            results.append({
                "test": f"random_cpsat_instance_{k}",
                "passed": len(vr) == 0,
                "details": f"seed={seed}, violations={len(vr)}",
            })
        else:
            results.append({
                "test": f"random_cpsat_instance_{k}",
                "passed": False,
                "details": f"seed={seed}, status={info_r.get('status')}",
            })

    inst_b = generate_instance(2, 8, seed=777, emergency_rate=0.1)
    for method in ["first_fit", "rule"]:
        gdf = greedy_schedule(inst_b, method)
        gv = check_hard_constraints(inst_b, gdf, required_must_be_scheduled=False)
        results.append({
            "test": f"baseline_{method}_no_hard_violations_for_placed_cases",
            "passed": len(gv) == 0,
            "details": f"violations={len(gv)}",
        })

    mu, sigma = lognormal_params_from_mean_sd(100, 25)
    p50 = lognormal_quantile(mu, sigma, 0.5)
    p80 = lognormal_quantile(mu, sigma, 0.8)
    results.append({
        "test": "lognormal_p80_greater_than_p50",
        "passed": p80 > p50,
        "details": f"p50={p50:.1f}, p80={p80:.1f}",
    })

    return pd.DataFrame(results)


def run_evaluation(
    days: int,
    num_rooms: int,
    num_cases: int,
    time_limit: float,
    surge: bool,
    do_events: bool,
    progress_cb=None,
) -> pd.DataFrame:
    rows = []
    params = default_params()
    base_emergency_rate = 0.12 * (1.3 if surge else 1.0)

    for d in range(days):
        seed = 9000 + d
        inst = generate_instance(num_rooms, num_cases, seed=seed, emergency_rate=base_emergency_rate)

        t0 = time.time()
        cpsat_df, cpsat_info = solve_two_stage(inst, params, time_limit=time_limit)
        cpsat_time = time.time() - t0
        rows.append(compute_metrics(
            inst,
            cpsat_df,
            "CP-SAT two-stage",
            solver_time=cpsat_time,
            gap_percent=cpsat_info["optimization"].get("gap_percent"),
            required_must_be_scheduled=True,
        ))

        t0 = time.time()
        greedy_df = greedy_schedule(inst, "first_fit")
        rows.append(compute_metrics(
            inst,
            greedy_df,
            "Greedy first-fit",
            solver_time=time.time() - t0,
            required_must_be_scheduled=False,
        ))

        t0 = time.time()
        rule_df = greedy_schedule(inst, "rule")
        rows.append(compute_metrics(
            inst,
            rule_df,
            "Rule-based",
            solver_time=time.time() - t0,
            required_must_be_scheduled=False,
        ))

        if do_events:
            changes_total = 0
            repair_time_total = 0.0
            violations_total = 0
            cur_inst = copy.deepcopy(inst)
            cur_df = cpsat_df.copy()

            for ev_i in range(2):
                now = DAY_START + 120 + 60 * ev_i
                ev = create_random_event(cur_inst, cur_df, now=now, seed=seed + ev_i)
                res = inject_event(
                    cur_inst,
                    cur_df,
                    ev,
                    params,
                    threshold=30,
                    auto_confirm=True,
                    time_limit=max(2.0, time_limit * 0.4),
                )
                repair_time_total += res["repair_time"]
                if res["applied"]:
                    cur_inst = res["new_instance"]
                    cur_df = res["new_schedule"]
                    changes_total += len(res["changes"])
                violations_total += len(res["violations"])

            deferred = 0
            if cur_df is not None and not cur_df.empty and "scheduled" in cur_df.columns:
                deferred = int((~cur_df["scheduled"]).sum())

            rows.append({
                "method": "CP-SAT + repair (2 events)",
                "on_time_start_rate": np.nan,
                "urgent_wait_min": np.nan,
                "room_utilization": np.nan,
                "overtime_hours": np.nan,
                "deferred_cases": deferred,
                "solver_time_sec": repair_time_total,
                "optimality_gap_pct": np.nan,
                "hard_violations": violations_total,
                "schedule_changes": changes_total,
            })

        if progress_cb:
            progress_cb((d + 1) / max(1, days))

    df = pd.DataFrame(rows)
    if "schedule_changes" not in df.columns:
        df["schedule_changes"] = np.nan
    return df


# ----------------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------------
st.set_page_config(page_title="TheatreSched", layout="wide", initial_sidebar_state="expanded")
st.title("🎭 TheatreSched")
st.caption(
    "Autonomous operating-room scheduling and repair under hard clinical/operational constraints. "
    "Synthetic data only. Not for clinical use."
)


def init_demo():
    params = default_params()
    size_key = st.session_state.get("instance_size", "Small")
    if size_key not in SIZE_MAP:
        size_key = "Small"
    num_rooms, num_cases = SIZE_MAP[size_key]

    inst = generate_instance(num_rooms=num_rooms, num_cases=num_cases, seed=7, emergency_rate=0.15)

    try:
        df, info = solve_two_stage(inst, params, time_limit=8.0)
    except Exception as exc:
        df = greedy_schedule(inst, "first_fit")
        info = {
            "feasibility": {"status": f"ERROR: {exc}", "feasible": False},
            "optimization": {
                "status": "FALLBACK_GREEDY",
                "feasible": bool(df is not None and not df.empty and df["scheduled"].any()),
                "objective": None,
                "bound": None,
                "gap_percent": None,
                "time": 0.0,
            },
            "feasible": bool(df is not None and not df.empty and df["scheduled"].any()),
            "total_time": 0.0,
        }

    if not info.get("feasible", False):
        df_g = greedy_schedule(inst, "first_fit")
        if df_g is not None and not df_g.empty and df_g["scheduled"].any():
            df = df_g
            info = {
                "feasibility": info.get("feasibility", {}),
                "optimization": {
                    "status": "FALLBACK_GREEDY",
                    "feasible": True,
                    "objective": None,
                    "bound": None,
                    "gap_percent": None,
                    "time": 0.0,
                },
                "feasible": True,
                "total_time": info.get("total_time", 0.0),
            }

    st.session_state.params = params
    st.session_state.instance = inst
    st.session_state.schedule = df
    st.session_state.solver_info = info
    st.session_state.change_log = []
    st.session_state.events = []
    st.session_state.pending_repair = None
    st.session_state.avail_exempt = set()
    st.session_state.init_error = None

    try:
        ev = {"type": "emergency", "time": 600, "description": "Demo emergency arrival at 10:00"}
        res = inject_event(
            inst,
            df,
            ev,
            params,
            threshold=30,
            auto_confirm=False,
            time_limit=6.0,
        )
        st.session_state.events.append(ev)

        if res["needs_confirm"]:
            st.session_state.pending_repair = res
        elif res["applied"]:
            st.session_state.instance = res["new_instance"]
            st.session_state.schedule = res["new_schedule"]
            st.session_state.avail_exempt = set(res.get("availability_exempt") or set())
            for row in res["changes"].to_dict("records"):
                row["confirmed"] = "auto"
                st.session_state.change_log.append(row)
    except Exception as exc:
        st.session_state.init_error = f"Demo repair initialization failed: {exc}"

    vid = make_demo_video(
        st.session_state.instance,
        st.session_state.schedule,
        st.session_state.events,
    )
    st.session_state.demo_video = vid


if "instance_size" not in st.session_state:
    st.session_state.instance_size = "Small"

if "theatresched_initialized" not in st.session_state:
    with st.spinner("Loading realistic demo data, solving small instance, generating live demo..."):
        init_demo()
    st.session_state.theatresched_initialized = True


with st.sidebar:
    st.header("⚙️ Controls")

    size_keys = list(SIZE_MAP.keys())
    default_index = size_keys.index(st.session_state.instance_size) if st.session_state.instance_size in SIZE_MAP else 0
    size_choice = st.radio("Instance size", size_keys, index=default_index)

    if st.button("Apply size / regenerate demo", use_container_width=True):
        st.session_state.instance_size = size_choice
        st.session_state.pop("theatresched_initialized", None)
        st.rerun()

    st.subheader("Solver")
    time_limit = st.slider("Static solver time limit (sec)", 2, 30, 8, 1)
    repair_time_limit = st.slider("Repair solver time limit (sec)", 2, 20, 6, 1)

    st.subheader("Objective weights")
    params = st.session_state.params
    params["urgent_wait"] = st.slider("Urgent waiting", 0.0, 20.0, float(params["urgent_wait"]), 0.5)
    params["utilization"] = st.slider("Room utilization", 0.0, 10.0, float(params["utilization"]), 0.5)
    params["overtime"] = st.slider("Overtime", 0.0, 10.0, float(params["overtime"]), 0.5)
    params["preference"] = st.slider("Surgeon preference", 0.0, 50.0, float(params["preference"]), 1.0)
    params["movement"] = st.slider("Repair movement penalty", 0.0, 30.0, float(params["movement"]), 0.5)
    params["deferral"] = st.slider("Elective deferral penalty", 0.0, 2000.0, float(params["deferral"]), 50.0)
    st.session_state.params = params

    st.subheader("Repair governance")
    threshold = st.slider("Human confirmation threshold (min)", 5, 120, 30, 5)
    auto_confirm = st.checkbox("Auto-confirm changes within threshold", value=True)

    st.subheader("Actions")
    if st.button("🧠 Re-solve current instance", use_container_width=True):
        with st.spinner("Solving..."):
            df, info = solve_two_stage(st.session_state.instance, st.session_state.params, time_limit=time_limit)
            st.session_state.schedule = df
            st.session_state.solver_info = info
            st.session_state.avail_exempt = set()
            st.session_state.demo_video = make_demo_video(
                st.session_state.instance,
                st.session_state.schedule,
                st.session_state.events,
            )
        st.success("Re-solved.")

    st.markdown("---")
    st.markdown(
        "**Safety note:** This is a decision-support prototype using synthetic data. "
        "No real patient data is used. Clinical review is mandatory before any real-world use."
    )


tab_overview, tab_schedule, tab_duration, tab_events, tab_eval, tab_tests, tab_report = st.tabs(
    ["Overview", "Schedule", "Duration model", "Events & repair", "Evaluation", "Tests", "Report"]
)

instance = st.session_state.instance
schedule = st.session_state.schedule
solver_info = st.session_state.solver_info
params = st.session_state.params
current_exempt = set(st.session_state.get("avail_exempt") or set())

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    if st.session_state.get("init_error"):
        st.warning(st.session_state["init_error"])

    st.subheader("Live demo")
    metrics = compute_metrics(
        instance,
        schedule,
        "CP-SAT",
        required_must_be_scheduled=True,
        availability_exempt=current_exempt,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("On-time start rate", f"{metrics['on_time_start_rate']:.0%}")
    c2.metric("Urgent wait", f"{metrics['urgent_wait_min']:.0f} min")
    c3.metric("Room utilization", f"{metrics['room_utilization']:.0%}")
    c4.metric("Overtime", f"{metrics['overtime_hours']:.1f} h")
    c5.metric("Deferred cases", f"{metrics['deferred_cases']}")
    c6.metric("Hard violations", f"{metrics['hard_violations']}")

    st.markdown("#### Solver status")
    opt_info = solver_info.get("optimization", {}) if isinstance(solver_info, dict) else {}
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Status", str(opt_info.get("status", "N/A")))
    sc2.metric("Objective", fmt_num(opt_info.get("objective"), 0))
    sc3.metric("Bound", fmt_num(opt_info.get("bound"), 0))

    gap = opt_info.get("gap_percent")
    sc4.metric("Optimality gap", "N/A" if gap is None else f"{float(gap):.2f}%")

    st.markdown("#### Generated demo video")
    vid = st.session_state.get("demo_video")
    if vid:
        if vid["mime"] == "video/mp4":
            st.video(vid["bytes"], format="video/mp4")
        else:
            st.image(vid["bytes"], use_container_width=True)
        st.download_button(
            "Download demo video",
            data=vid["bytes"],
            file_name=f"theatresched_demo.{vid['ext']}",
            mime=vid["mime"],
        )
    else:
        st.info("Demo video unavailable for current schedule.")

    st.markdown("#### Recent change log")
    if st.session_state.change_log:
        st.dataframe(pd.DataFrame(st.session_state.change_log), use_container_width=True)
    else:
        st.caption("No confirmed changes yet. Go to Events & repair to confirm the pending demo repair.")

# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
with tab_schedule:
    st.subheader("Gantt schedule")
    old_df = None
    if st.session_state.pending_repair:
        old_df = st.session_state.pending_repair.get("old_schedule")
    event_times = [safe_int(e.get("time")) for e in st.session_state.events if safe_int(e.get("time")) is not None]
    fig = make_gantt(instance, schedule, old_df, event_times)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Case table")
    display_df = schedule.copy()
    if not display_df.empty:
        display_df["start_fmt"] = display_df["start"].map(fmt_min)
        display_df["end_fmt"] = display_df["end"].map(fmt_min)
        display_df["staff"] = display_df["staff"].apply(lambda x: ", ".join(as_list(x)))
        cols = [
            "case_id", "procedure", "priority", "room", "surgeon", "staff",
            "start_fmt", "end_fmt", "duration", "earliest_start", "deadline", "scheduled",
        ]
        st.dataframe(display_df[[c for c in cols if c in display_df.columns]], use_container_width=True)

    st.subheader("Hard-constraint audit")
    violations = check_hard_constraints(
        instance,
        schedule,
        required_must_be_scheduled=True,
        availability_exempt=current_exempt,
    )
    if violations:
        st.error(f"{len(violations)} violation(s) detected")
        st.write(violations[:50])
    else:
        st.success("No hard-constraint violations detected by independent checker.")

# ---------------------------------------------------------------------------
# Duration model
# ---------------------------------------------------------------------------
with tab_duration:
    st.subheader("Log-normal duration model")
    st.markdown(
        "Historical durations are synthetic. For each procedure, a log-normal distribution is fitted "
        "and the 80th percentile is used as the planning buffer."
    )

    proc_names = list(PROCEDURES.keys())
    selected_proc = st.selectbox("Procedure", proc_names)
    meta = PROCEDURES[selected_proc]
    mu, sigma = lognormal_params_from_mean_sd(meta["mean"], meta["sd"])
    samples = simulate_historical_durations(mu, sigma, n=1000, seed=11)
    p50 = float(np.quantile(samples, 0.5))
    p80 = float(np.quantile(samples, 0.8))
    p95 = float(np.quantile(samples, 0.95))

    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Median", f"{p50:.0f} min")
    dc2.metric("P80 planning value", f"{p80:.0f} min")
    dc3.metric("P95", f"{p95:.0f} min")

    hist = go.Figure()
    hist.add_trace(go.Histogram(x=samples, nbinsx=40, marker_color="#1f77b4", opacity=0.75))
    hist.add_vline(x=p80, line_dash="dash", line_color="red", annotation_text="P80")
    hist.update_layout(
        title=f"Synthetic duration distribution: {selected_proc}",
        xaxis_title="Duration (min)",
        yaxis_title="Count",
    )
    st.plotly_chart(hist, use_container_width=True)

    st.subheader("Procedure catalog")
    catalog_rows = []
    for p, m in PROCEDURES.items():
        mu_p, sigma_p = lognormal_params_from_mean_sd(m["mean"], m["sd"])
        catalog_rows.append({
            "procedure": p,
            "specialty": m["specialty"],
            "mean_min": m["mean"],
            "sd_min": m["sd"],
            "mu": round(mu_p, 3),
            "sigma": round(sigma_p, 3),
            "p80_min": int(math.ceil(lognormal_quantile(mu_p, sigma_p, 0.8))),
            "cleaning_min": m["cleaning"],
            "features": ", ".join(sorted(m["features"])) if m["features"] else "-",
        })
    st.dataframe(pd.DataFrame(catalog_rows), use_container_width=True)

# ---------------------------------------------------------------------------
# Events & repair
# ---------------------------------------------------------------------------
with tab_events:
    st.subheader("Dynamic events and minimal-change repair")

    pending = st.session_state.pending_repair
    if pending:
        st.warning("A repair proposal is awaiting human confirmation.")
        st.markdown("**Event:** " + str(pending["event"].get("description", "Event")))
        st.markdown("**Reasons:** " + "; ".join(pending.get("log", [])))

        pinfo = pending.get("info", {}) or {}
        st.caption(
            f"Repair status: {pinfo.get('status', 'N/A')} | "
            f"Repair time: {fmt_num(pending.get('repair_time'), 2, 's')} | "
            f"Gap: {'N/A' if pinfo.get('gap_percent') is None else fmt_num(pinfo.get('gap_percent'), 2, '%')}"
        )

        if pending.get("changes") is not None and not pending["changes"].empty:
            st.dataframe(pending["changes"], use_container_width=True)
        else:
            st.caption("No case-level changes computed.")

        viol = pending.get("violations", [])
        if viol:
            st.error(f"Post-repair checker found {len(viol)} violation(s).")
            st.write(viol[:20])
        else:
            st.success("Post-repair schedule passes hard-constraint checker.")

        b1, b2 = st.columns(2)
        if b1.button("✅ Confirm repair", type="primary", use_container_width=True):
            st.session_state.instance = pending["new_instance"]
            st.session_state.schedule = pending["new_schedule"]
            st.session_state.avail_exempt = set(pending.get("availability_exempt") or set())
            for row in pending["changes"].to_dict("records"):
                row["confirmed"] = "human"
                st.session_state.change_log.append(row)
            st.session_state.pending_repair = None
            st.session_state.demo_video = make_demo_video(
                st.session_state.instance,
                st.session_state.schedule,
                st.session_state.events,
            )
            st.success("Repair confirmed and applied.")
            st.rerun()

        if b2.button("❌ Reject repair", use_container_width=True):
            for row in pending["changes"].to_dict("records"):
                row["confirmed"] = "rejected"
                st.session_state.change_log.append(row)
            st.session_state.pending_repair = None
            st.info("Repair rejected. Previous schedule retained.")
            st.rerun()

    else:
        st.markdown("#### Inject a new event")
        ec1, ec2 = st.columns(2)
        event_type = ec1.selectbox(
            "Event type",
            ["Emergency arrival", "Case overrun", "Surgeon unavailable", "Equipment failure"],
        )

        event_max = max(DAY_START, DAY_END - 300)
        default_event_time = min(600, event_max)
        now = ec2.slider("Event time", DAY_START, event_max, default_event_time, 15)
        ec2.caption(f"Selected: {fmt_min(now)}")

        event = None
        if event_type == "Emergency arrival":
            event = {"type": "emergency", "time": int(now), "description": f"Emergency arrival at {fmt_min(now)}"}

        elif event_type == "Case overrun":
            if schedule is None or schedule.empty:
                sched_cases = []
            else:
                sched_cases = schedule[schedule["scheduled"]]["case_id"].tolist()
            if not sched_cases:
                st.info("No scheduled cases available for overrun.")
            else:
                cid = st.selectbox("Case", sched_cases)
                extra = st.slider("Extra minutes", 10, 90, 25, 5)
                event = {
                    "type": "overrun",
                    "time": int(now),
                    "case_id": cid,
                    "extra": int(extra),
                    "description": f"Overrun +{extra} min on {cid}",
                }

        elif event_type == "Surgeon unavailable":
            sid = st.selectbox("Surgeon", [s.id for s in instance.surgeons])
            event = {
                "type": "surgeon_unavailable",
                "time": int(now),
                "surgeon_id": sid,
                "description": f"Surgeon {sid} unavailable from {fmt_min(now)}",
            }

        else:
            rid = st.selectbox("Room", [r.id for r in instance.rooms])
            event = {
                "type": "equipment_failure",
                "time": int(now),
                "room_id": rid,
                "description": f"Equipment failure in {rid} at {fmt_min(now)}",
            }

        if event and st.button("⚡ Inject event and compute repair", use_container_width=True):
            with st.spinner("Computing minimal-change repair..."):
                res = inject_event(
                    st.session_state.instance,
                    st.session_state.schedule,
                    event,
                    st.session_state.params,
                    threshold=threshold,
                    auto_confirm=auto_confirm,
                    time_limit=repair_time_limit,
                )
            st.session_state.events.append(event)

            if res["needs_confirm"] and not auto_confirm:
                st.session_state.pending_repair = res
                st.warning("Repair requires human confirmation.")
            elif res["applied"]:
                st.session_state.instance = res["new_instance"]
                st.session_state.schedule = res["new_schedule"]
                st.session_state.avail_exempt = set(res.get("availability_exempt") or set())
                for row in res["changes"].to_dict("records"):
                    row["confirmed"] = "auto" if not res["needs_confirm"] else "auto_within_threshold"
                    st.session_state.change_log.append(row)
                st.session_state.demo_video = make_demo_video(
                    st.session_state.instance,
                    st.session_state.schedule,
                    st.session_state.events,
                )
                st.success(
                    f"Repair applied. Changes: {len(res['changes'])}. "
                    f"Solver time: {res['repair_time']:.2f}s"
                )
            else:
                st.error("Repair could not be applied within the local search space.")
                st.write(res.get("log", []))

    st.markdown("#### Event log")
    if st.session_state.events:
        st.dataframe(pd.DataFrame(st.session_state.events), use_container_width=True)
    else:
        st.caption("No events injected yet.")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    st.subheader("Synthetic evaluation")
    st.markdown(
        "Full specification calls for 200 synthetic days. To keep the demo interactive, run a sample first. "
        "Increase days for offline benchmarking."
    )

    ev1, ev2, ev3, ev4 = st.columns(4)
    days = ev1.slider("Days", 1, 200, 2, 1)
    rooms_eval = ev2.selectbox("Rooms", [2, 4, 8], index=1)
    cases_eval = ev3.selectbox("Cases/day", [10, 25, 45, 80], index=1)
    tl_eval = ev4.slider("Solver time/day (sec)", 2, 30, 6, 1)

    surge = st.checkbox("Stress test: +30% emergency surge", value=False)
    do_events = st.checkbox("Include 2 repair events per day", value=True)

    if st.button("📊 Run evaluation", use_container_width=True):
        progress = st.progress(0.0, text="Preparing...")

        def cb(p):
            progress.progress(min(1.0, p), text=f"Evaluating... {int(p*100)}%")

        with st.spinner("Running evaluation..."):
            eval_df = run_evaluation(days, rooms_eval, cases_eval, tl_eval, surge, do_events, cb)
        progress.empty()

        st.session_state.eval_results = eval_df

    if "eval_results" in st.session_state:
        eval_df = st.session_state.eval_results
        st.dataframe(eval_df, use_container_width=True)

        st.markdown("#### Method comparison")
        static_df = eval_df[eval_df["method"].isin(["CP-SAT two-stage", "Greedy first-fit", "Rule-based"])]
        if not static_df.empty:
            m1, m2 = st.columns(2)
            with m1:
                fig = go.Figure()
                for meth in static_df["method"].unique():
                    sub = static_df[static_df["method"] == meth]
                    fig.add_trace(go.Bar(
                        name=meth,
                        x=["On-time", "Utilization"],
                        y=[sub["on_time_start_rate"].mean(), sub["room_utilization"].mean()],
                    ))
                fig.update_layout(barmode="group", title="Mean on-time and utilization")
                st.plotly_chart(fig, use_container_width=True)
            with m2:
                fig2 = go.Figure()
                for meth in static_df["method"].unique():
                    sub = static_df[static_df["method"] == meth]
                    fig2.add_trace(go.Bar(
                        name=meth,
                        x=["Urgent wait", "Overtime h"],
                        y=[sub["urgent_wait_min"].mean(), sub["overtime_hours"].mean()],
                    ))
                fig2.update_layout(barmode="group", title="Mean urgent wait and overtime")
                st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
with tab_tests:
    st.subheader("Property-based and constraint-model tests")
    n_tests = st.slider("Random instances", 1, 20, 5, 1)
    if st.button("🧪 Run tests", use_container_width=True):
        with st.spinner("Running tests..."):
            test_df = run_property_tests(n_random=n_tests, time_limit=3.0)
        st.session_state.test_results = test_df

    if "test_results" in st.session_state:
        test_df = st.session_state.test_results
        passed = int(test_df["passed"].sum())
        total = len(test_df)
        st.metric("Tests passed", f"{passed}/{total}")
        st.dataframe(test_df, use_container_width=True)
        if passed == total:
            st.success("All tests passed.")
        else:
            st.warning("Some tests failed. See details above.")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
with tab_report:
    st.subheader("Modeling assumptions, clinical review, and privacy limits")
    st.markdown(
        """
### Logical module map
Although the deliverable is a single `app.py` for ease of running, the code is organized into logical modules:

- `data/`: synthetic instance generation, rooms, surgeons, staff, cases.
- `duration/`: log-normal fitting and 80th-percentile planning buffers.
- `model/`: CP-SAT decision variables, hard constraints, interval/no-overlap constraints.
- `solver/`: two-stage feasibility then optimization, optimality gap reporting.
- `events/`: emergency arrival, overrun, surgeon unavailability, equipment failure.
- `repair/`: minimal-change local repair, human confirmation threshold, change logging.
- `eval/`: metrics, baselines, stress testing, property-based tests, independent hard-constraint checker.

### Hard constraints modeled
- No double-booking of rooms, surgeons, anesthesiologists, and nurses.
- Required room equipment/features must be present.
- Surgeon specialty, availability windows, and maximum daily operating hours.
- Staff role coverage, shift availability, and skill requirements.
- Mandatory cleaning gaps between cases in the same room, dependent on previous case type.
- Emergency cases must not be delayed by elective cases in the same room.

### Soft objectives
Weighted and configurable in the sidebar:
- Minimize urgent waiting time.
- Maximize room utilization.
- Minimize overtime after the target end time.
- Respect preferred surgeon assignments.
- Minimize schedule disruption during repair by penalizing movement from prior start times.
- Penalize elective deferral.

### Two-stage solver
1. **Feasibility model**: hard constraints only, used to quickly establish a valid baseline.
2. **Optimization model**: adds weighted soft objectives and reports optimality gap when available.

The solver-bound reporting uses a compatibility helper because different OR-Tools CP-SAT
Python builds expose the best objective bound through slightly different APIs.

### Dynamic repair design
- Cases already started before the event are fixed and exempt from event-based availability
  truncation, while still participating in no-overlap and cleaning constraints.
- Future cases are not allowed to be moved into the past during repair.
- The repair search space is expanded locally only when needed:
  - first event-affected cases,
  - then immediate successors/predecessors in the same room, surgeon, or staff chain,
  - then future electives, and finally future cases if necessary.
- Any change moving a case by more than the configured threshold requires human confirmation.
- Every change is logged with old/new start, resource changes, duration changes, and reason.

### Duration modeling
- Synthetic historical durations are generated per procedure.
- A log-normal distribution is fitted.
- The 80th percentile is used as the planning duration to create operational buffers.
- Overrun events increase the planned duration and trigger repair.

### Evaluation
The app can generate synthetic days with realistic:
- emergency/urgent/elective mixes,
- arrival times,
- procedure duration distributions,
- room equipment constraints,
- staff shifts,
- surgeon specialties and max hours.

Reported metrics:
- on-time start/completion proxy rate,
- average urgent waiting time,
- room utilization,
- overtime hours,
- deferred cases,
- solver time,
- optimality gap for CP-SAT,
- number of schedule changes after events,
- hard-constraint violations after solve and after repair.

Baselines:
- greedy first-fit,
- manually designed rule-based scheduler.

Stress test:
- +30% emergency arrival rate.

### Independent verification
- A brute-force style hard-constraint checker recomputes all constraints from raw data.
- Property-based tests generate random small instances and verify:
  - CP-SAT solutions pass the checker,
  - the checker detects artificial violations,
  - baselines do not violate hard constraints for placed cases,
  - duration quantiles behave monotonically.

### Clinical review required before real use
This prototype is not clinically safe for production use without:
- review by anesthesia, nursing, perioperative, and surgical leaders;
- validation of cleaning time policies by infection prevention;
- review of emergency preemption rules by clinical governance;
- prospective shadow-mode evaluation against real schedules;
- definition of escalation, override, and manual adjustment workflows;
- model monitoring, drift detection, and incident review processes;
- regulatory/quality assessment appropriate to the jurisdiction.

### Privacy limits
- The app uses only synthetic data.
- No protected health information (PHI) or personally identifiable information (PII) is included.
- Before any real deployment, data minimization, de-identification/access controls, audit logging,
  and compliance with HIPAA/GDPR or local equivalent must be established.
- Historical duration models must be trained under appropriate governance and must not expose
  patient-level information through outputs.
        """
    )