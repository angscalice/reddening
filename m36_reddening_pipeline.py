# M36 INTERSTELLAR REDDENING PIPELINE
# Objectives: Measure the amount (A0) and direction of interstellar reddening toward the open
#   cluster M36 (NGC 1960) by subtracting each star's isochrone colours and decomposing the
#   colour residuals with SVD. Compare the direction with the Gaia DR3 extinction law.
# Data supplied by Panja et al. (2021, VizieR J/ApJ/910/80), Gaia DR3 and PARSEC isochrones.
# MA305 PROJECT
# Code by Angelina Scalice.

### IMPORTS
import getpass
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")                   # save figures without opening windows
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from zero_point import zpt

# NOTE: RUNNING THIS SCRIPT will create an output folder of CSV tables, figures and summary.txt.
#   View these for full data tables. The two PARSEC isochrone files must be in data/isochrones,
#   and the first run needs a Gaia archive login (later runs re-use the saved downloads).


### SETTINGS
# Define folders:
root_dir = Path(__file__).resolve().parent
data_dir = root_dir / "data"
iso_dir = data_dir / "isochrones"
out_dir = root_dir / "output"
fig_dir = out_dir / "figures"

# Run control
run_validation = True
reuse_downloads = True          # re-use the VizieR / Gaia tables already saved in output
gaia_poll_s = 5                 #s , how often the Gaia job phase is checked
gaia_timeout_s = 600            #s , give up on one Gaia job after this long

# Panja et al. (2021) member table
vizier_catalog = "J/ApJ/910/80"
expected_members = 200

# Gaia cross-match (one cone around the members, matched locally) and quality cuts
credentials_file = root_dir / ".gaia_credentials.txt"
xmatch_radius_arcsec = 1.0      #arcsec , cross-match radius
cone_pad_arcmin = 1.0           #arcmin , cone radius = farthest member from the centre + pad
ruwe_max = 1.4                  # renormalised unit weight error limit
excess_nsigma = 3.0             # |C*| < N sigma_C* (Riello et al. 2021)
mag_err_floor = 0.02            #mag , replaces zero or missing magnitude errors

# Literature values for M36 (Panja et al. 2021, ApJ, 910, 80)
literature = {
    "distance_pc": 1200.0,      #pc , 1.20 +/- 0.13 kpc
    "distance_err_pc": 130.0,
    "age_myr": 15.0,            #Myr
    "ebv_range": (0.20, 0.25),  #mag , E(B-V)
    "rv": 3.1,
}
a0_start = literature["rv"] * sum(literature["ebv_range"]) / 2.0   #mag , starting A0 (~0.70)

# Bayesian distances
edsd_length_pc = 1350.0                 #pc , exponentially decreasing space density prior
star_rgrid_pc = (10.0, 20000.0, 6000)   #pc , (min, max, n points), logarithmic
cluster_dgrid_pc = (300.0, 5000.0, 20000)   #pc , (min, max, n points), linear; uniform prior

# Iterative reddening solve
a0_tol = 0.001                  #mag , convergence tolerance
max_iter = 50
clip_nsigma = 3.0               # outlier boundary
branch_dcolour = 0.05           #mag , candidates closer than this in BP-RP count as one branch

# PARSEC isochrones
iso_gaia_file = iso_dir / "parsec_gaiaedr3.dat"
iso_ir_file = iso_dir / "parsec_2mass_spitzer.dat"
iso_logage = None               # None: nearest grid age to literature["age_myr"]
iso_max_label = 3               # keep PMS(0), MS(1), SGB(2), RGB(3)

# Analysis
outlier_nsigma = 3.0

# Validation
rng_seed = 305
n_null = 50                     # null-test realisations
inject_a0 = [0.2, 0.5, 1.0, 2.0]    #mag , injected A0 levels
n_inject = 20                   # realisations per injected level
n_bootstrap = 1000
n_pca_noise = 300               # noise-only realisations: reference for the differential-reddening PCA
n_pca_noise_inject = 20         # ...per injection realisation (kept small for speed)
spatial_k = 10                  # spatial coherence: neighbours averaged per star
spatial_k_scales = [5, 10, 20]  # ...and the scales reported
n_permutations = 2000           # shuffled-position realisations for the p-value


### CONSTANTS
# Bands (blue to red) and adjacent-band colours
BANDS = ["BP", "G", "RP", "J", "H", "Ks", "I1", "I2"]
BAND_LABELS = {"BP": "BP", "G": "G", "RP": "RP", "J": "J", "H": "H",
               "Ks": "Ks", "I1": "[3.6]", "I2": "[4.5]"}

# Gaia EDR3 extinction law (Fitzpatrick 2019 curve, main-sequence fit, X = (BP-RP)0).
# Source: ESA Gaia, "Gaia EDR3 extinction law", file Fitz19_EDR3_MainSequence.csv
#   https://www.cosmos.esa.int/web/gaia/edr3-extinction-law
# k = a1 + a2 X + a3 X^2 + a4 X^3 + a5 A0 + a6 A0^2 + a7 A0^3 + a8 A0 X + a9 A0 X^2 + a10 X A0^2
EDR3_LAW_BPRP = {
    "G": [0.995969721536602, -0.159726460302015, 0.0122380738156057, 0.00090726555099859,
          -0.0377160263914123, 0.00151347495244888, -2.52364537395142e-05,
          0.0114522658102451, -0.000936914989014318, -0.000260296774134201],
    "BP": [1.15363197483424, -0.0814012991657388, -0.036013023976704, 0.0192143585568966,
           -0.022397548243016, 0.000840562680547171, -1.31018008013549e-05,
           0.00660124080271006, -0.000882247501989453, -0.000111215755291684],
    "RP": [0.66320787941067, -0.0179847164933981, 0.000493769449961458, -0.00267994405695751,
           -0.00651422146709376, 3.30179903473159e-05, 1.57894227641527e-06,
           -7.9800898337247e-05, 0.000255679812110045, 1.10476584967393e-05],
}
LAW_X_RANGE = (-0.06, 2.5)        #mag , validity of the law in (BP-RP)0
LAW_A0_RANGE = (0.01, 20.0)       #mag , validity of the law in A0

# Infrared extinction: 2MASS as A_lambda/A_V from Wang & Chen (2019, ApJ 877, 116, Table 3);
# IRAC as A_lambda/A_Ks from Indebetouw et al. (2005), scaled by Wang & Chen's A_Ks/A_V
WANG_CHEN_2019 = {"J": 0.243, "H": 0.131, "Ks": 0.078}
INDEBETOUW_2005_AK = {"I1": 0.56, "I2": 0.43}    # check Table 1

# VizieR column -> name used here
PANJA_COLUMNS = {
    "Seq": "seq", "_RAJ2000": "ra", "_DEJ2000": "dec",
    "Jmag": "J", "e_Jmag": "e_J", "Hmag": "H", "e_Hmag": "e_H", "Kmag": "Ks", "e_Kmag": "e_Ks",
    "3.6mag": "I1", "e_3.6mag": "e_I1", "4.5mag": "I2", "e_4.5mag": "e_I2",
    "plx": "plx_dr2", "Dist": "dist_panja",
}

# Gaia archive query pieces. The cone touches gaia_source only; the GSP-Phot parameters and
# the Bailer-Jones distances are fetched afterwards by source_id so the query doesn't time out
GAIA_SOURCE_FIELDS = """
  g.source_id, g.ra AS ra_dr3, g.dec AS dec_dr3,
  g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
  g.phot_g_mean_flux_over_error, g.phot_bp_mean_flux_over_error, g.phot_rp_mean_flux_over_error,
  g.parallax, g.parallax_error, g.ruwe, g.phot_bp_rp_excess_factor,
  g.nu_eff_used_in_astrometry, g.pseudocolour, g.ecl_lat, g.astrometric_params_solved"""
GAIA_AP_TABLE = "gaiadr3.astrophysical_parameters"
GAIA_AP_FIELDS = ["source_id", "azero_gspphot", "azero_gspphot_lower", "azero_gspphot_upper"]
GAIA_DIST_TABLE = "external.gaiaedr3_distance"
GAIA_DIST_FIELDS = ["source_id", "r_med_geo"]
GAIA_IDS_PER_QUERY = 500          # source_ids per query
GAIA_FINAL_PHASES = ("COMPLETED", "ERROR", "ABORTED")

# Gaia photometry and astrometry constants
MAG_PER_FLUX_ERR = 1.0857362                     # 2.5 / ln 10
GAIA_ZP_ERR = {"G": 0.0027553202, "BP": 0.0027901700, "RP": 0.0037793818}  #mag , Riello+2021
GAIA_FLUX_ERR_COLS = {"G": "phot_g_mean_flux_over_error", "BP": "phot_bp_mean_flux_over_error",
                      "RP": "phot_rp_mean_flux_over_error"}
ZP_SOLUTION_TYPES = [31, 95]                     # 5- and 6-parameter astrometric solutions
# Riello et al. (2021) corrected BP/RP flux excess C* = C - f(BP-RP) and its scatter
EXCESS_BLUE = (1.154360, 0.033772, 0.032277)                 # BP-RP < 0.5
EXCESS_MID = (1.162004, 0.011464, 0.049255, -0.005879)       # 0.5 <= BP-RP < 4
EXCESS_RED = (1.057572, 0.140537)                            # BP-RP >= 4
EXCESS_SIGMA = (0.0059898, 8.817481e-12, 7.618399)           # a + b G^c

# Statistics and numerics
MAD_TO_SIGMA = 1.4826
ISO_SLOPE_STEP_MAG = 0.02        #mag , step for the isochrone slope dx0/dM_G
BRANCH_TARGET_FLOOR_MAG = 0.01   #mag , floor on colour errors when choosing a branch
INJECT_ANGLE_TOL_DEG = 5.0       #deg , injection-recovery direction criterion
PCA_ALIGN_MAX_DEG = 20.0         #deg , centred PC1 counts as the dust direction only within this angle
SPATIAL_WINSOR_NSIGMA = 3.0      # cap differential A0 at this many robust sigma

# PARSEC (CMD) column patterns
ISO_BAND_PATTERNS = {
    "G": [r"^gmag$"],
    "BP": [r"^gbpmag$", r"^bpmag$", r"^gbpftmag$"],
    "RP": [r"^grpmag$", r"^rpmag$"],
    "J": [r"^jmag$"],
    "H": [r"^hmag$"],
    "Ks": [r"^ksmag$", r"^kmag$"],
    "I1": [r"irac.*36", r"^36mag$", r"^irac1"],
    "I2": [r"irac.*45", r"^45mag$", r"^irac2"],
}

# Create the output folders
for folder in (data_dir, iso_dir, out_dir, fig_dir):
    folder.mkdir(parents=True, exist_ok=True)

print('MA305 M36 reddening pipeline')


### BANDS, COLOURS AND EXTINCTION LAW
# Helper functions, used by the reddening solve, analysis and validation

def colour_names(bands):
    return [f"{BAND_LABELS[a]}-{BAND_LABELS[b]}" for a, b in zip(bands[:-1], bands[1:])]


def difference_matrix(nb):
    # D (nb-1 x nb) so that colours = D @ mags for adjacent-band colours
    D = np.zeros((nb - 1, nb))
    for j in range(nb - 1):
        D[j, j], D[j, j + 1] = 1.0, -1.0
    return D


def colour_matrix(mags):
    # (n, nb) magnitudes to (n, nb-1) adjacent colours
    m = np.asarray(mags, float)
    return m[:, :-1] - m[:, 1:]


def colour_covariances(errs):
    # (n, nb) band errors to (n, nb-1, nb-1) colour covariances.
    # Adjacent colours share a band, so the covariance is tridiagonal, not diagonal.
    e = np.asarray(errs, float)
    D = difference_matrix(e.shape[1])
    return np.einsum("ij,nj,kj->nik", D, e ** 2, D)


def gaia_k(band, a0, bprp0):
    # k_band = A_band / A0 from the Gaia EDR3 law (inputs clipped to the valid range)
    a = EDR3_LAW_BPRP[band]
    X = np.clip(np.asarray(bprp0, float), *LAW_X_RANGE)
    A = np.clip(np.asarray(a0, float), *LAW_A0_RANGE)
    return (a[0] + a[1] * X + a[2] * X ** 2 + a[3] * X ** 3
            + a[4] * A + a[5] * A ** 2 + a[6] * A ** 3
            + a[7] * A * X + a[8] * A * X ** 2 + a[9] * X * A ** 2)


def band_k(a0, bprp0, bands):
    # (n, nb) array of A_band / A0 for every star (law evaluated at each star's (BP-RP)0)
    x = np.atleast_1d(np.asarray(bprp0, float))
    a = np.broadcast_to(np.asarray(a0, float), x.shape)
    out = np.empty((x.size, len(bands)))
    for j, b in enumerate(bands):
        if b in EDR3_LAW_BPRP:
            out[:, j] = gaia_k(b, a, x)
        elif b in WANG_CHEN_2019:
            out[:, j] = WANG_CHEN_2019[b]
        else:
            out[:, j] = INDEBETOUW_2005_AK[b] * WANG_CHEN_2019["Ks"]
    return out


def theory_vectors(a0, bprp0, bands):
    # e_i: colour excess per unit A0 for each star, (n, nb-1)
    k = band_k(a0, bprp0, bands)
    return k[:, :-1] - k[:, 1:]


def unit(v):
    w = np.asarray(v, float)
    return w / np.linalg.norm(w)


def compare_vectors(empirical, theory):
    # Cosine, angle, L2 / Linf distance and per-component % difference between two
    # directions (both normalised first)
    a, b = unit(empirical), unit(theory)
    cos = float(np.clip(a @ b, -1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * (a - b) / np.abs(b)
    return {"cosine": cos, "angle_deg": float(np.degrees(np.arccos(cos))),
            "L2": float(np.linalg.norm(a - b)), "Linf": float(np.max(np.abs(a - b))),
            "pct_diff": pct}


def robust_z(x):
    # (x - median) / (1.4826 MAD)
    med = np.nanmedian(x)
    mad = MAD_TO_SIGMA * np.nanmedian(np.abs(x - med))
    return (x - med) / mad if mad > 0 else np.zeros_like(x)


def dist_modulus(d_pc):
    return 5.0 * np.log10(d_pc) - 5.0


### ERROR-WEIGHTED CLUSTER MEAN
# Cluster Mean of Per-Star Estimates
# Robust clip: drop |x - median| > clip * 1.4826 * MAD (catastrophic values).
# Add an intrinsic scatter sigma_int to each variance so that the reduced chi2 of the kept
#   stars equals 1
# Error-normalised clip: drop |x - mean| / sqrt(var + sigma_int^2) > clip, refit, repeat.

def weighted_mean_with_scatter(xs, vs, s2):
    w = 1.0 / (vs + s2)
    return np.sum(w * xs) / np.sum(w), w


def chi2_excess(s2, xs, vs):
    # chi^2 - (n-1) of the weighted mean with intrinsic scatter s2 (root = sigma_int^2)
    m, w = weighted_mean_with_scatter(xs, vs, s2)
    return np.sum(w * (xs - m) ** 2) - (xs.size - 1)


def fit_mean_and_scatter(x, v, keep):
    xs, vs = x[keep], v[keep]
    s2 = 0.0
    if xs.size > 2 and chi2_excess(0.0, xs, vs) > 0:
        hi = max(np.var(xs), 1e-12) * 10 + 1e-6
        while chi2_excess(hi, xs, vs) > 0:
            hi *= 10
        s2 = brentq(chi2_excess, 0.0, hi, args=(xs, vs))
    m, w = weighted_mean_with_scatter(xs, vs, s2)
    return m, w, s2


def robust_weighted_mean(values, variances, clip=3.0):
    x = np.asarray(values, float)
    v = np.asarray(variances, float)
    keep = np.isfinite(x) & np.isfinite(v) & (v > 0)

    if clip and keep.sum() > 5:
        for _ in range(10):
            med = np.median(x[keep])
            mad = MAD_TO_SIGMA * np.median(np.abs(x[keep] - med))
            if mad <= 0:
                break
            new = keep & (np.abs(x - med) <= clip * mad)
            if new.sum() == keep.sum():
                break
            keep = new
    if keep.sum() == 0:
        return {"mean": np.nan, "err": np.nan, "sigma_int": np.nan, "keep": keep,
                "n": 0, "err_scatter": np.nan}
    m, w, s2 = fit_mean_and_scatter(x, v, keep)
    if clip:
        for _ in range(20):
            new = keep & (np.abs(x - m) / np.sqrt(v + s2) <= clip)
            if new.sum() == keep.sum() or new.sum() < 3:
                break
            keep = new
            m, w, s2 = fit_mean_and_scatter(x, v, keep)
    xs = x[keep]
    n = xs.size
    return {"mean": float(m), "err": float(1.0 / np.sqrt(np.sum(w))),
            "sigma_int": float(np.sqrt(s2)), "keep": keep, "n": int(n),
            "err_scatter": float(np.std(xs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan}


### DATA PREPARATION
# Load the Panja et al. (2021) member table from VizieR (or saved copy)
# Cross-match the members with Gaia DR3:
#   download every Gaia source in one cone around the members and match locally within 1 arcsec,
#   then fetch the GSP-Phot extinctions and Bailer-Jones distances by source_id.
# Remove stars with poor astrometry (RUWE) or bad BP/RP photometry (flux excess).
# Correct the Gaia parallaxes for the Lindegren et al. (2021) zero-point.

def cone_query(ra0, dec0, radius_deg):
    # Every gaia_source row inside one cone (indexed, fast)
    return f"""
SELECT {GAIA_SOURCE_FIELDS.strip()}
FROM gaiadr3.gaia_source AS g
WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec),
                   CIRCLE('ICRS', {ra0:.6f}, {dec0:.6f}, {radius_deg:.6f}))
""".strip()


def ids_query(table, fields, source_ids):
    # Selected columns of `table` for a list of source_ids
    id_list = ", ".join(str(int(i)) for i in source_ids)
    return f"SELECT {', '.join(fields)}\nFROM {table}\nWHERE source_id IN ({id_list})"


def gaia_login():
    # Log in to the Gaia archive: environment variables, credentials file, or prompt
    user = os.environ.get("GAIA_USER")
    pw = os.environ.get("GAIA_PASSWORD")

    if not (user and pw) and credentials_file.exists():
        lines = [l.strip() for l in credentials_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) >= 2:
            user, pw = lines[0], lines[1]
            print('    credentials read from %s' % credentials_file.name)
    if not (user and pw):
        print('    no credentials found (looked for environment variables and %s)' % credentials_file)
        print('Gaia archive login')
        user = input('  Gaia username: ').strip()
        pw = getpass.getpass('  Gaia password: ')
    Gaia.login(user=user, password=pw)
    print('    logged in to the Gaia archive as %s' % user)


def run_gaia_job(query, label):
    # Submit one query as a background job and log stage until end
    t0 = time.time()
    job = Gaia.launch_job_async(query, background=True, verbose=False)
    jobid = getattr(job, "jobid", "?")
    last_phase = None
    print("    %s: job %s submitted (also listed under 'Jobs' on the Gaia archive website)" % (label, jobid))
    while True:
        try:
            phase = str(job.get_phase(update=True)).upper()
        except Exception as exc:              # keep polling through a network blip
            phase = f"UNKNOWN ({exc.__class__.__name__})"
        if phase != last_phase:
            print('    %s: job phase %s' % (label, phase))
            last_phase = phase
        if phase in GAIA_FINAL_PHASES:
            break
        if time.time() - t0 > gaia_timeout_s:
            raise TimeoutError(f"{label}: Gaia job {jobid} still {phase} after {gaia_timeout_s} s "
                               "(you can delete it under 'Jobs' on the archive website)")
        time.sleep(gaia_poll_s)
    if phase != "COMPLETED":
        raise RuntimeError(f"{label}: Gaia job {jobid} ended in phase {phase}")
    return job.get_results().to_pandas()


def fetch_by_source_id(source_ids, table, fields, label):
    # Columns of `table` for the matched source_ids, in batches of GAIA_IDS_PER_QUERY
    ids = np.unique(np.asarray(source_ids, dtype=np.int64))
    parts = [run_gaia_job(ids_query(table, fields, ids[s:s + GAIA_IDS_PER_QUERY]), label)
             for s in range(0, ids.size, GAIA_IDS_PER_QUERY)]
    if not parts:
        return pd.DataFrame(columns=fields)
    table_out = pd.concat(parts, ignore_index=True)
    table_out.columns = [c.lower() for c in table_out.columns]
    print('    %s: values for %d of %d sources' % (label, len(table_out), ids.size))
    return table_out


def corrected_excess(C, bp_rp, G):
    # Riello et al. (2021) corrected flux excess C* and its 1-sigma scatter
    x = np.asarray(bp_rp, float)
    b, m, r = EXCESS_BLUE, EXCESS_MID, EXCESS_RED
    a, bb, c = EXCESS_SIGMA
    f = np.where(x < 0.5, b[0] + b[1] * x + b[2] * x ** 2,
                 np.where(x < 4.0, m[0] + m[1] * x + m[2] * x ** 2 + m[3] * x ** 3,
                          r[0] + r[1] * x))
    sig = a + bb * np.asarray(G, float) ** c
    return np.asarray(C, float) - f, sig


print('\nData Preparation:')

# Load the member table (VizieR, or the saved copy)
raw_csv = out_dir / "m36_panja_raw.csv"
if reuse_downloads and raw_csv.exists():
    print('Members: re-using %s (set reuse_downloads = False to re-query)' % raw_csv.name)
    raw = pd.read_csv(raw_csv)
else:
    print('Members: querying VizieR %s (row_limit = -1)' % vizier_catalog)
    tables = Vizier(columns=["**", "_RAJ2000", "_DEJ2000"], row_limit=-1).get_catalogs(vizier_catalog)
    if len(tables) == 0:
        raise RuntimeError(f"VizieR returned no tables for {vizier_catalog}")
    for name, t in zip(tables.keys(), tables):
        print('    table %s: %d rows' % (name, len(t)))
    # The member table: 200 rows if present, otherwise the biggest table
    best = next((t for t in tables if len(t) == expected_members), None)
    if best is None:
        best = max(tables, key=len)
    raw = best.to_pandas()
    raw.to_csv(raw_csv, index=False)
    print('    saved raw table unchanged as %s' % raw_csv.name)
if len(raw) != expected_members:
    print('WARNING: expected %d members, got %d' % (expected_members, len(raw)))
else:
    print('Members: confirmed %d rows' % len(raw))

# Rename to the names used here, keep only those columns, add a row id for the cross-match
members = raw.rename(columns=PANJA_COLUMNS)[list(PANJA_COLUMNS.values())]
members = members.assign(uid=np.arange(len(members)))

# Cross-match with Gaia DR3 (or re-use the saved match)
xmatch_csv = out_dir / "m36_gaia_dr3_xmatch.csv"
if reuse_downloads and xmatch_csv.exists():
    print('Cross-match: re-using %s' % xmatch_csv.name)
    gx = pd.read_csv(xmatch_csv)
else:
    print('Cross-match: matching %d positions with Gaia DR3 within %.1f arcsec'
          % (len(members), xmatch_radius_arcsec))
    gaia_login()
    try:
        # One cone around the members: centre = mean position, radius = farthest member + pad
        mem = SkyCoord(members["ra"].values * u.deg, members["dec"].values * u.deg)
        centre = SkyCoord(members["ra"].mean() * u.deg, members["dec"].mean() * u.deg)
        radius_deg = mem.separation(centre).deg.max() + cone_pad_arcmin / 60.0
        query = cone_query(centre.ra.deg, centre.dec.deg, radius_deg)
        (out_dir / "query_gaia_cone.adql").write_text(query, encoding="utf-8")
        print('    cone: centre (%.4f, %.4f) deg, radius %.1f arcmin'
              % (centre.ra.deg, centre.dec.deg, radius_deg * 60))
        cone = run_gaia_job(query, "cone download")
        cone.columns = [c.lower() for c in cone.columns]
        print('    cone holds %d Gaia sources; matching locally within %.1f arcsec'
              % (len(cone), xmatch_radius_arcsec))

        # Every source within the radius is kept, one row per (member, candidate)
        gai = SkyCoord(cone["ra_dr3"].values * u.deg, cone["dec_dr3"].values * u.deg)
        idx_m, idx_g, sep, _ = gai.search_around_sky(mem, xmatch_radius_arcsec * u.arcsec)
        gx = cone.iloc[idx_g].reset_index(drop=True)
        gx.insert(0, "sep_arcsec", sep.to(u.arcsec).value)
        gx.insert(0, "uid", members["uid"].values[idx_m])
        print('    positional match: %d candidate rows for %d members' % (len(gx), gx["uid"].nunique()))

        ap = fetch_by_source_id(gx["source_id"], GAIA_AP_TABLE, GAIA_AP_FIELDS, "GSP-Phot parameters")
        dist_bj = fetch_by_source_id(gx["source_id"], GAIA_DIST_TABLE, GAIA_DIST_FIELDS,
                                     "Bailer-Jones distances")
    finally:
        try:
            Gaia.logout()
        except Exception:
            pass
    gx = gx.merge(ap, on="source_id", how="left").merge(dist_bj, on="source_id", how="left")
    gx.to_csv(xmatch_csv, index=False)
    print('    saved %s (%d candidate matches)' % (xmatch_csv.name, len(gx)))

# Keep the nearest candidate per member; flag members with 2+ candidates
ncand = gx.groupby("uid").size().rename("n_candidates")
best = gx.sort_values("sep_arcsec").drop_duplicates("uid").set_index("uid").join(ncand)
stars = members.set_index("uid").join(best, how="left").reset_index()
stars["n_candidates"] = stars["n_candidates"].fillna(0).astype(int)
unmatched = stars[stars["source_id"].isna()]
multi = stars[stars["n_candidates"] > 1]
unmatched[["seq", "ra", "dec"]].to_csv(out_dir / "xmatch_unmatched.csv", index=False)
multi[["seq", "ra", "dec", "n_candidates", "source_id", "sep_arcsec"]].to_csv(
    out_dir / "xmatch_multiple_candidates.csv", index=False)
print('    matched %d / %d; unmatched %d; with 2+ candidates %d (nearest kept, flagged)'
      % (stars["source_id"].notna().sum(), len(stars), len(unmatched), len(multi)))

# Gaia magnitudes, magnitude errors and the corrected BP/RP flux excess
stars["G"], stars["BP"], stars["RP"] = stars["phot_g_mean_mag"], stars["phot_bp_mean_mag"], stars["phot_rp_mean_mag"]
for b, col in GAIA_FLUX_ERR_COLS.items():
    with np.errstate(divide="ignore", invalid="ignore"):
        stars[f"e_{b}"] = np.sqrt((MAG_PER_FLUX_ERR / stars[col]) ** 2 + GAIA_ZP_ERR[b] ** 2)
stars["Cstar"], stars["sig_Cstar"] = corrected_excess(stars["phot_bp_rp_excess_factor"],
                                                       stars["BP"] - stars["RP"], stars["G"])

# Quality cuts, logged one by one
cut_rows = []
keep = pd.Series(True, index=stars.index)
cuts = [("match", "has a Gaia DR3 match within radius", stars["source_id"].notna()),
        ("ruwe", f"RUWE < {ruwe_max}", stars["ruwe"] < ruwe_max),
        ("flux_excess", f"|C*| < {excess_nsigma:g} sigma (normal BP/RP flux excess)",
         np.abs(stars["Cstar"]) < excess_nsigma * stars["sig_Cstar"])]
for name, desc, passed in cuts:
    new = keep & passed
    cut_rows.append({"name": name, "cut": desc, "n_before": int(keep.sum()),
                     "n_removed": int(keep.sum() - new.sum()), "n_after": int(new.sum()), "note": ""})
    keep = new

# Replace zero or missing magnitude errors with the error floor
n_replaced = 0
for b in BANDS:
    bad = keep & stars[b].notna() & (stars[f"e_{b}"].isna() | (stars[f"e_{b}"] <= 0))
    n_replaced += int(bad.sum())
    stars.loc[bad, f"e_{b}"] = mag_err_floor
cut_rows.append({"name": "error_floor", "cut": f"zero/missing magnitude errors set to {mag_err_floor}",
                 "n_before": int(keep.sum()), "n_removed": 0, "n_after": int(keep.sum()),
                 "note": f"{n_replaced} band-errors replaced"})

stars["in_clean"] = keep
stars["in_sample"] = keep & stars[BANDS].notna().all(axis=1)
cut_rows.append({"name": "sample", "cut": "Gaia + 2MASS + IRAC photometry complete",
                 "n_before": int(keep.sum()), "n_removed": int(keep.sum() - stars["in_sample"].sum()),
                 "n_after": int(stars["in_sample"].sum()), "note": ""})
cut_log = pd.DataFrame(cut_rows)
cut_log.to_csv(out_dir / "data_validation_cuts.csv", index=False)
for r in cut_rows:
    print('    %-55s %4d -> %4d' % (r["cut"], r["n_before"], r["n_after"]))

# Parallax zero-point (Lindegren et al. 2021): plx_corr = plx - Z5
zp = np.full(len(stars), np.nan)
zpt.load_tables()
sel = stars["parallax"].notna() & stars["astrometric_params_solved"].isin(ZP_SOLUTION_TYPES)
zp[sel.values] = zpt.get_zpt(stars.loc[sel, "phot_g_mean_mag"].values,
                             stars.loc[sel, "nu_eff_used_in_astrometry"].values,
                             stars.loc[sel, "pseudocolour"].values,
                             stars.loc[sel, "ecl_lat"].values,
                             stars.loc[sel, "astrometric_params_solved"].values,
                             _warnings=False)
stars["zpt"] = zp
stars["plx_corr"] = stars["parallax"] - stars["zpt"]
print('Parallax zero-point (Lindegren et al. 2021): median Z5 = %.4f mas over %d stars'
      % (np.nanmedian(zp), np.isfinite(zp).sum()))
stars.to_csv(out_dir / "m36_clean.csv", index=False)


### BAYESIAN DISTANCES
# Per star: posterior with an exponentially decreasing space density (EDSD) prior,
#   p(r | plx) ~ r^2 exp(-r/L) N(plx | 1000/r, sigma^2)      [r in pc, plx in mas]
# Cluster: one shared distance d_c fitted to all members (uniform prior),
#   p(d_c | {plx_i}) ~ prod_i N(plx_i | 1000/d_c, sigma_i^2)
# d_c sets the distance modulus. Compare with Panja et al. (2021) and Bailer-Jones et al. (2021).

def cdf_quantiles(x, logp, qs=(0.16, 0.5, 0.84)):
    # Quantiles of 1-D posteriors tabulated on grid x (one row per star)
    lp = np.atleast_2d(logp)
    p = np.exp(lp - lp.max(axis=1, keepdims=True))
    dx = np.diff(x)
    cum = np.concatenate([np.zeros((p.shape[0], 1)),
                          np.cumsum(0.5 * (p[:, 1:] + p[:, :-1]) * dx, axis=1)], axis=1)
    cum /= cum[:, -1:]
    return np.array([[np.interp(q, c, x) for q in qs] for c in cum])


print('\nBayesian Distances:')
use = stars["in_clean"] & stars["plx_corr"].notna() & stars["parallax_error"].notna()
plx = stars.loc[use, "plx_corr"].values[:, None]
plx_err = stars.loc[use, "parallax_error"].values[:, None]
print('Bayesian distances for %d stars (EDSD L = %.0f pc)' % (use.sum(), edsd_length_pc))

# Per-star posteriors on logarithmic grid
r_grid = np.geomspace(star_rgrid_pc[0], star_rgrid_pc[1], int(star_rgrid_pc[2]))
logp = 2 * np.log(r_grid) - r_grid / edsd_length_pc - 0.5 * ((plx - 1000.0 / r_grid) / plx_err) ** 2
q = cdf_quantiles(r_grid, logp)
stars.loc[use, "r_lo16"], stars.loc[use, "r_med"], stars.loc[use, "r_hi84"] = q[:, 0], q[:, 1], q[:, 2]

# Cluster posterior on linear grid
d_grid = np.linspace(cluster_dgrid_pc[0], cluster_dgrid_pc[1], int(cluster_dgrid_pc[2]))
logpost = -0.5 * np.sum(((plx - 1000.0 / d_grid) / plx_err) ** 2, axis=0)
d_lo16, d_c, d_hi84 = cdf_quantiles(d_grid, logpost)[0]
post = np.exp(logpost - logpost.max())
trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 renamed trapz
post /= trapz(post, d_grid)
dm = dist_modulus(d_c)
dm_err = 5.0 / np.log(10) * 0.5 * (d_hi84 - d_lo16) / d_c
print('    hierarchical d_c = %.1f (+%.1f / -%.1f) pc  ->  DM = %.3f +/- %.3f mag'
      % (d_c, d_hi84 - d_c, d_c - d_lo16, dm, dm_err))

# Comparison with other distance estimates
dist = {"d_c_pc": float(d_c), "d_c_lo16": float(d_lo16), "d_c_hi84": float(d_hi84), "DM": dm, "DM_err": dm_err,
        "median_per_star_r_med_pc": float(np.median(q[:, 1])), "n_stars": int(use.sum())}
dp = stars.loc[use, "dist_panja"]
if dp.notna().any():
    scale = 1000.0 if np.nanmedian(dp) < 50 else 1.0     # Panja's Dist is in kpc
    dist["panja_median_dist_pc"] = float(np.nanmedian(dp) * scale)
if stars.loc[use, "r_med_geo"].notna().any():
    dist["bailer_jones_median_r_med_geo_pc"] = float(np.nanmedian(stars.loc[use, "r_med_geo"]))
    dist["median_per_star_minus_bj_pc"] = float(np.nanmedian(stars.loc[use, "r_med"] - stars.loc[use, "r_med_geo"]))
if stars.loc[use, "plx_dr2"].notna().any():
    dist["median_plx_dr3corr_minus_dr2_mas"] = float(np.nanmedian(stars.loc[use, "plx_corr"] - stars.loc[use, "plx_dr2"]))
for key in ("median_per_star_r_med_pc", "panja_median_dist_pc", "bailer_jones_median_r_med_geo_pc",
            "median_per_star_minus_bj_pc", "median_plx_dr3corr_minus_dr2_mas"):
    if key in dist:
        print('    %s = %.3f' % (key, dist[key]))
pd.DataFrame({"d_pc": d_grid, "posterior": post}).iloc[::10].to_csv(
    out_dir / "cluster_distance_posterior.csv", index=False)


### PARSEC ISOCHRONE
# Read the two CMD tables (Gaia EDR3; 2MASS + Spitzer), both made at A_V = 0.
# Merge them row by row on (age, initial mass) and keep the grid age nearest the literature age.
# An isochrone is a dict: "mags" (n_pts, 8) absolute magnitudes in BANDS order,
#   "label" (PARSEC evolutionary label) "logage" and "mh" ([M/H]).

def norm_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_col(df, patterns):
    normed = {norm_name(c): c for c in df.columns}
    for p in patterns:
        for n, c in normed.items():
            if re.search(p, n):
                return c
    return None


def read_cmd_file(path):
    # Read a CMD (PARSEC) isochrone table into a DataFrame using header line
    p = Path(path)
    header = None
    if not p.exists():
        raise FileNotFoundError(f"Isochrone file not found: {p}\nDownload it from the CMD web form.")
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("#"):
                toks = s.lstrip("#").split()
                normed = [norm_name(t) for t in toks]
                if "mini" in normed and any("logage" in t for t in normed):
                    header = toks
            elif s:
                break
    if header is None:
        raise ValueError(f"Could not find the column header line in {p}. "
                         "Is this a CMD isochrone table ('isochrone tables' output)?")
    df = pd.read_csv(p, sep=r"\s+", comment="#", header=None, names=header,
                     engine="python", on_bad_lines="skip")
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def iso_band(iso, b):
    return iso["mags"][:, BANDS.index(b)]


def interpolate_at_MG(iso, MG, bprp_guess, branch_flag_dcolour=0.05,
                      target=None, target_var=None, band_idx=None):
    # Intrinsic magnitudes at each star's absolute G.
    # Where several branches overlap (pre-main sequence vs main
    # sequence) the candidate closest to the star is chosen and the star is flagged:
    # closest by chi^2 in all colours if `target` (n, nc dereddened colours of the bands
    # `band_idx`) and `target_var` are given, otherwise closest in (BP-RP)0 to `bprp_guess`.
    # Returns mags (n, 8) [NaN when M_G is outside the isochrone], n_branches, label.
    mags, label = iso["mags"], iso["label"]
    mg = np.atleast_1d(np.asarray(MG, float))
    guess = np.atleast_1d(np.asarray(bprp_guess, float))
    g = iso_band(iso, "G")
    lo, hi = g[:-1], g[1:]
    ok_seg = np.isfinite(lo) & np.isfinite(hi) & (lo != hi)
    iBP, iRP = BANDS.index("BP"), BANDS.index("RP")
    out = np.full((mg.size, mags.shape[1]), np.nan)
    nbr = np.zeros(mg.size, int)
    lab = np.full(mg.size, -1)

    for i, m in enumerate(mg):
        if not np.isfinite(m):
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (m - lo) / (hi - lo)
        seg = np.where(ok_seg & (t >= 0) & (t <= 1))[0]
        if seg.size == 0:
            continue
        tt = t[seg][:, None]
        cand = mags[seg] + tt * (mags[seg + 1] - mags[seg])
        cbprp = cand[:, iBP] - cand[:, iRP]
        # Distinct branches (adjacent segments meeting at a vertex count once)
        nbr[i] = 1 + int(np.sum(np.diff(np.sort(cbprp)) > branch_flag_dcolour))
        if target is not None and np.all(np.isfinite(target[i])):
            cc = cand[:, band_idx][:, :-1] - cand[:, band_idx][:, 1:]
            k = int(np.argmin(np.sum((cc - target[i]) ** 2 / target_var[i], axis=1)))
        else:
            gi = guess[i] if np.isfinite(guess[i]) else np.nanmedian(cbprp)
            k = int(np.argmin(np.abs(cbprp - gi)))
        out[i] = cand[k]
        lab[i] = label[seg[k]] if tt[k, 0] < 0.5 else label[seg[k] + 1]
    return out, nbr, lab


print('\nPARSEC isochrone:')
iso_gaia = read_cmd_file(iso_gaia_file)
iso_ir = read_cmd_file(iso_ir_file)
age_g, mini_g, label_g, mh_g = (find_col(iso_gaia, [r"^logage"]), find_col(iso_gaia, [r"^mini$"]),
                                find_col(iso_gaia, [r"^label$", r"^stage$"]), find_col(iso_gaia, [r"^mh$", r"^feh$"]))
age_i, mini_i = find_col(iso_ir, [r"^logage"]), find_col(iso_ir, [r"^mini$"])
if None in (age_g, mini_g, age_i, mini_i):
    raise ValueError("Isochrone files lack logAge / Mini columns.")
iso_gaia = iso_gaia.assign(_age=iso_gaia[age_g].round(4), _mini=iso_gaia[mini_g].round(7))
iso_ir = iso_ir.assign(_age=iso_ir[age_i].round(4), _mini=iso_ir[mini_i].round(7))

# Find each band's column (Gaia bands in the first file, 2MASS and IRAC in the second)
iso_cols = {}
for b in BANDS:
    src = iso_gaia if b in ("G", "BP", "RP") else iso_ir
    iso_cols[b] = find_col(src, ISO_BAND_PATTERNS[b])
    if iso_cols[b] is None:
        raise ValueError(f"Isochrone file has no {b} column. Columns: {list(src.columns)}")

# Merge the two tables and keep the grid age nearest the target
iso_merged = iso_gaia.merge(iso_ir[["_age", "_mini"] + [iso_cols[b] for b in ("J", "H", "Ks", "I1", "I2")]],
                            on=["_age", "_mini"], how="inner", suffixes=("", "_ir"))
if len(iso_merged) < 0.9 * min(len(iso_gaia), len(iso_ir)):
    print('WARNING: Only %d of %d/%d isochrone rows matched between the two CMD files; '
          'make sure both were made with the same ages, metallicity and settings.'
          % (len(iso_merged), len(iso_gaia), len(iso_ir)))
ages = np.unique(iso_merged["_age"].values)
age_target = iso_logage if iso_logage is not None else np.log10(literature["age_myr"] * 1e6)
age = ages[np.argmin(np.abs(ages - age_target))]
iso_sel = iso_merged[iso_merged["_age"] == age]
if label_g is not None:
    iso_sel = iso_sel[iso_sel[label_g] <= iso_max_label]
iso_sel = iso_sel.sort_values("_mini")
mh = float(iso_sel[mh_g].iloc[0]) if mh_g is not None else 0.0
iso = {"mags": np.column_stack([iso_sel[iso_cols[b]].values for b in BANDS]),
       "label": iso_sel[label_g].values.astype(int) if label_g is not None else np.ones(len(iso_sel), int),
       "logage": float(age), "mh": mh}
print('PARSEC isochrone: logAge = %.3f (target %.3f; grid has %d ages), [M/H] = %.2f, %d points'
      % (age, age_target, ages.size, mh, len(iso_sel)))


### ITERATIVE REDDENING SOLVE
# Build the colour matrix X from adjacent-band colours (linearly independent); checks rank.
# Compute each star's absolute magnitude M_G = G - 5 log10(d_c) + 5 - k_G A0.
# Read each star's intrinsic colours x0_i from A_V = 0 isochrone at M_G;
#   residuals R = X - X0 are the colour excesses (each row is one star).
# Compute each star's theory vector e_i (colour excess per unit A0) from the Gaia law at its
#   (BP-RP)0 and A0, plus the infrared ratios.
# Least squares per star: A0_i = e_i.r_i / e_i.e_i ; cluster A0 = error-weighted mean.
# Repeat until A0 changes by less than a0_tol.

def build_colour_matrix(df, bands, verbose=True):
    mags = df[bands].to_numpy(float)
    errs = df[[f"e_{b}" for b in bands]].to_numpy(float)
    X = colour_matrix(mags)
    cov = colour_covariances(errs)
    sv = np.linalg.svd(X, compute_uv=False)
    rank = int(np.linalg.matrix_rank(X))
    if verbose:
        print('Colour matrix %d x %d (%s); rank = %d'
              % (X.shape[0], X.shape[1], ", ".join(colour_names(bands)), rank))
        print('    singular values of X: %s' % np.array2string(sv, precision=4))
    return X, cov, mags, errs, {"rank": rank, "singular_values": sv}


def evaluate_solution(a0, bprp0, ctx):
    # Absolute magnitudes, residuals, theory vectors and A0 at a given A0 and (BP-RP)0
    # estimate. `ctx` holds the fixed arrays prepared in solve().
    X, cov, errs = ctx["X"], ctx["cov"], ctx["errs"]
    G, bprp_obs, dm = ctx["G"], ctx["bprp_obs"], ctx["dm"]
    iso, iso_idx, bands, D, iG, n = ctx["iso"], ctx["iso_idx"], ctx["bands"], ctx["D"], ctx["iG"], ctx["n"]
    iBP_iso, iRP_iso = BANDS.index("BP"), BANDS.index("RP")
    h = ISO_SLOPE_STEP_MAG

    # Absolute magnitudes
    kk = band_k(a0, bprp0, ["BP", "G", "RP"])
    MG = G - dm - kk[:, 1] * a0
    # Intrinsic colours - where branches overlap, the one closest to the star in all its
    # dereddened colours (chi^2); recomputed every iteration
    guess = bprp_obs - (kk[:, 0] - kk[:, 2]) * a0
    target = X - a0 * theory_vectors(a0, bprp0, bands)
    tvar = np.diagonal(cov, axis1=1, axis2=2) + BRANCH_TARGET_FLOOR_MAG ** 2
    iso_mags, nbr, lab = interpolate_at_MG(iso, MG, guess, branch_dcolour, target, tvar, iso_idx)
    x0 = colour_matrix(iso_mags[:, iso_idx])
    bprp0_new = iso_mags[:, iBP_iso] - iso_mags[:, iRP_iso]
    bprp0_new = np.where(np.isfinite(bprp0_new), bprp0_new, guess)
    R = X - x0
    # G noise also moves M_G and hence x0 along the isochrone: dr/dm = D - s (x) u_G,
    # with s = dx0/dM_G the local isochrone slope on the same branch
    xp = colour_matrix(interpolate_at_MG(iso, MG + h, bprp0_new)[0][:, iso_idx])
    xm = colour_matrix(interpolate_at_MG(iso, MG - h, bprp0_new)[0][:, iso_idx])
    slope = np.where(np.isfinite(xp) & np.isfinite(xm), (xp - xm) / (2 * h),
                     np.where(np.isfinite(xp), (xp - x0) / h,
                              np.where(np.isfinite(xm), (x0 - xm) / h, 0.0)))
    J = np.broadcast_to(D, (n,) + D.shape).copy()
    J[:, :, iG] -= slope
    cov_r = np.einsum("nij,nj,nkj->nik", J, errs ** 2, J)
    # Theory vectors
    E = theory_vectors(a0, bprp0_new, bands)
    # Least-squares A0 per star and the cluster mean
    ee = np.einsum("ij,ij->i", E, E)
    a0_i = np.einsum("ij,ij->i", E, R) / ee
    var_i = np.einsum("ij,ijk,ik->i", E, cov_r, E) / ee ** 2
    a0_i = np.where(np.isfinite(var_i), a0_i, np.nan)
    wm = robust_weighted_mean(a0_i, var_i, clip_nsigma)
    return dict(MG=MG, x0=x0, R=R, E=E, cov=cov_r, a0_i=a0_i, var_i=var_i, wm=wm,
                bprp0=bprp0_new, nbr=nbr, lab=lab, slope=slope, kG=kk[:, 1])


def solve(df, iso, bands, d_pc, verbose=True):
    # Run the iterative reddening solve on stars in df (which must hold `bands` and
    # `e_<band>` columns).
    a0 = a0_start
    X, cov, mags, errs, rank_info = build_colour_matrix(df, bands, verbose)
    iBP, iG, iRP = bands.index("BP"), bands.index("G"), bands.index("RP")
    dm = dist_modulus(d_pc)
    bprp_obs = mags[:, iBP] - mags[:, iRP]
    ctx = {"X": X, "cov": cov, "errs": errs, "G": mags[:, iG], "bprp_obs": bprp_obs, "dm": dm,
           "iso": iso, "iso_idx": [BANDS.index(b) for b in bands], "bands": bands,
           "D": difference_matrix(len(bands)), "iG": iG, "n": X.shape[0]}
    k0 = band_k(a0, bprp_obs, ["BP", "RP"])
    bprp0 = bprp_obs - (k0[:, 0] - k0[:, 1]) * a0     # first guess of (BP-RP)0
    history = []
    converged = False

    for it in range(1, max_iter + 1):
        ev = evaluate_solution(a0, bprp0, ctx)
        wm = ev["wm"]
        a0_new = wm["mean"]
        history.append({"iteration": it, "A0_in": a0, "A0_out": a0_new, "A0_err": wm["err"],
                        "sigma_int": wm["sigma_int"], "n_used": wm["n"],
                        "n_outside_isochrone": int((~np.isfinite(ev["a0_i"])).sum())})
        if verbose:
            print('    iter %2d: A0 = %.4f +/- %.4f  (sigma_int %.3f, %d stars, %d outside isochrone)'
                  % (it, a0_new, wm["err"], wm["sigma_int"], wm["n"], history[-1]["n_outside_isochrone"]))
        if not np.isfinite(a0_new):
            break
        bprp0 = ev["bprp0"]
        delta = abs(a0_new - a0)
        a0 = a0_new
        if delta < a0_tol:
            converged = True
            break

    ev = evaluate_solution(a0, bprp0, ctx)     # final products at the converged A0
    wm = ev["wm"]
    valid = np.isfinite(ev["a0_i"])
    table = pd.DataFrame({
        "MG": ev["MG"], "bprp_obs": bprp_obs, "bprp0": ev["bprp0"], "n_branches": ev["nbr"],
        "iso_label": ev["lab"], "branch_flag": ev["nbr"] > 1, "A0_i": ev["a0_i"],
        "A0_i_err_phot": np.sqrt(ev["var_i"]), "in_isochrone": valid, "used_in_mean": wm["keep"],
    }, index=df.index)
    if verbose:
        print('    %s after %d iterations: A0 = %.4f +/- %.4f mag '
              '(scatter-based err %.4f; sigma_int %.3f; %d stars used, %d branch-flagged)'
              % ("converged" if converged else "NOT converged", len(history), wm["mean"],
                 wm["err"], wm["err_scatter"], wm["sigma_int"], wm["n"], int((ev["nbr"] > 1).sum())))
    return {
        "bands": bands, "colours": colour_names(bands), "d_pc": d_pc, "DM": dm,
        "A0": wm["mean"], "A0_err": wm["err"], "A0_err_scatter": wm["err_scatter"],
        "sigma_int": wm["sigma_int"], "n_used": wm["n"], "converged": converged,
        "iterations": len(history), "history": pd.DataFrame(history), "table": table,
        "X": X, "X0": ev["x0"], "R": ev["R"], "E": ev["E"], "cov": ev["cov"], "valid": valid,
        "keep": wm["keep"], "rank": rank_info, "iso": iso, "mags": mags, "errs": errs,
        "iso_slope": ev["slope"], "kG": ev["kG"],
    }


print('\nIterative Reddening Solve:')
sample = stars[stars["in_sample"]]
if len(sample) < 5:
    raise RuntimeError(f"Only {len(sample)} stars have complete photometry; "
                       "check the cuts in data_validation_cuts.csv")
print('%d stars, starting A0 = %.3f' % (len(sample), a0_start))
res = solve(sample, iso, BANDS, d_c)
res["history"].to_csv(out_dir / "A0_convergence.csv", index=False)
sample[["seq"]].join(res["table"]).to_csv(out_dir / "per_star_A0.csv")
np.savetxt(out_dir / "residual_matrix.csv", res["R"], delimiter=",",
           header=",".join(res["colours"]), comments="")


### ANALYSIS
# SVD of the residual matrix R WITHOUT mean-centring (the dust is common to all members, so
#   subtracting the mean colour would remove it); v1 = empirical reddening direction, s1/s2.
# Compare v1 with the extinction law: cosine / angle, L2, Linf, % difference; star by star.
# Rank colours by how much they are reddened; check what is left after subtracting the
#   fitted reddening; flag outliers
# Differential reddening: mean-centred PCA of R - A0 e_i (how stars differ from one another).
# Spatial coherence: does the differential A0 follow position on the sky (patchy dust) or is it
#   random from star to star (binaries, age spread)? Compare.

def svd_direction(R, reference):
    # SVD of R (rows = stars); v1 is sign-matched to `reference`
    _, s, Vt = np.linalg.svd(R, full_matrices=False)
    v1 = Vt[0] if Vt[0] @ reference >= 0 else -Vt[0]
    return {"v1": v1, "s": s,
            "s1_over_s2": float(s[0] / s[1]) if s.size > 1 and s[1] > 0 else np.inf}


def differential_directions(E, kG, slope):
    # Direction in which EXTRA dust on one star moves its residual. The extra extinction
    # reddens it along e_i, but also makes it look fainter, so the isochrone lookup reads its
    # intrinsic colours lower down the isochrone:  e_eff,i = e_i - k_G,i * (dx0/dM_G)_i.
    return E - kG[:, None] * slope


def centred_pca(D, E_eff, cov, rng, n_noise):
    # Differential reddening: conventional (mean-centred) PCA of D = R - A0 e_i, the residuals
    # after each star's share of the common dust has been removed. (Plain mean-centring of R
    # is not enough: the Gaia law depends on intrinsic colour). What remains is how stars differ from one another.
    #
    # Find what PC1 is, and how much differential reddening there is

    Rc = D - D.mean(axis=0)
    reference = E_eff.mean(axis=0)
    L = np.linalg.cholesky(cov)
    ee = np.einsum("ij,ij->i", E_eff, E_eff)
    noise_var = float(np.mean(np.trace(cov, axis1=1, axis2=2)))      # per-star variance from errors
    dA0_noise_var = np.einsum("ij,ijk,ik->i", E_eff, cov, E_eff) / ee ** 2  # per-star noise on delta A0
    cos_noise, frac_noise = [], []

    _, s, Vt = np.linalg.svd(Rc, full_matrices=False)
    w1 = Vt[0] if Vt[0] @ reference >= 0 else -Vt[0]
    var_frac = s ** 2 / np.sum(s ** 2)
    cmp = compare_vectors(w1, reference)
    for _ in range(n_noise):
        noise = np.einsum("nij,nj->ni", L, rng.normal(size=(cov.shape[0], cov.shape[1])))
        noise = noise - noise.mean(axis=0)
        _, s_n, Vt_n = np.linalg.svd(noise, full_matrices=False)
        cos_noise.append(abs(float(unit(Vt_n[0]) @ unit(reference))))
        frac_noise.append(float(s_n[0] ** 2 / np.sum(s_n ** 2)))
    # Direction test only: the photometric noise is very anisotropic (IR errors >> Gaia),
    # so noise alone already gives PC1 a large variance share; that share is reported, not tested
    aligned = bool(cmp["angle_deg"] < PCA_ALIGN_MAX_DEG
                   and abs(cmp["cosine"]) > np.percentile(cos_noise, 95))
    # Per-star differential A0 along each star's own e_eff, and its intrinsic spread
    dA0 = np.einsum("ij,ij->i", E_eff, Rc) / ee
    excess = float(np.var(dA0) - np.mean(dA0_noise_var))
    return {"w1": w1, "var_frac": var_frac, "cosine": cmp["cosine"], "angle_deg": cmp["angle_deg"],
            "aligned": aligned,
            "abs_cos_noise_95pct": float(np.percentile(cos_noise, 95)),
            "var_frac1_noise_95pct": float(np.percentile(frac_noise, 95)),
            "scatter_over_noise": float(np.sum(s ** 2) / Rc.shape[0] / noise_var),
            "dA0_spread_over_noise": float(np.var(dA0) / np.mean(dA0_noise_var)),
            "sigma_differential_A0": float(np.sqrt(max(excess, 0.0))),
            "differential_A0": dA0}


def coherent_fraction(r, k):
    # Fraction f of the variance that is spatially coherent, from the correlation r
    # between a star's value and the mean of its k neighbours, for a smooth field plus
    # independent scatter:  r = f / sqrt(f + (1 - f)/k).
    a = 1.0 - 1.0 / k
    rr = max(float(r), 0.0) ** 2
    return float(min((rr * a + np.sqrt(rr ** 2 * a ** 2 + 4.0 * rr / k)) / 2.0, 1.0))


def neighbour_corr(v, idx):
    # Pearson correlation between each value and the mean of its neighbours
    return float(np.corrcoef(v, v[idx].mean(axis=1))[0, 1])


print('\nAnalysis:')
in_fit = res["keep"] & res["valid"]             # stars on the isochrone and kept in the mean
R, E = res["R"][in_fit], res["E"][in_fit]
colours, A0, tab = res["colours"], res["A0"], res["table"]
ebar = E.mean(axis=0)

# Reddening direction
svd = svd_direction(R, ebar)
v1 = svd["v1"]
print('Direction: singular values %s ; s1/s2 = %.2f' % (np.array2string(svd["s"], precision=4), svd["s1_over_s2"]))
print('    v1 = %s' % ", ".join(f"{c}: {x:+.3f}" for c, x in zip(colours, v1)))

# Comparison with the extinction law
cmp_mean = compare_vectors(v1, ebar)
with np.errstate(invalid="ignore", divide="ignore"):
    ang_v1_ei = np.degrees(np.arccos(np.clip(E @ unit(v1) / np.linalg.norm(E, axis=1), -1, 1)))
    ang_ri_ei = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", R, E) / (np.linalg.norm(R, axis=1) * np.linalg.norm(E, axis=1)), -1, 1)))
per_star = pd.DataFrame({"bprp0": tab.loc[in_fit, "bprp0"].values, "angle_v1_to_e_i_deg": ang_v1_ei,
                         "angle_r_i_to_e_i_deg": ang_ri_ei}, index=tab.index[in_fit])
print('Comparison with the law: v1 vs mean theory: cos = %.5f, angle = %.2f deg, L2 = %.4f, Linf = %.4f'
      % (cmp_mean["cosine"], cmp_mean["angle_deg"], cmp_mean["L2"], cmp_mean["Linf"]))
print('    per-component %% difference: %s'
      % ", ".join(f"{c}: {p:+.1f}%" for c, p in zip(colours, cmp_mean["pct_diff"])))
print('    star-by-star angle(v1, e_i): median %.2f deg (range %.2f-%.2f)'
      % (np.median(ang_v1_ei), ang_v1_ei.min(), ang_v1_ei.max()))

# Which colours suffer most
order = np.argsort(-np.abs(v1))
ranking = pd.DataFrame({"colour": [colours[i] for i in order], "v1_loading": v1[order],
                        "theory_unit_loading": unit(ebar)[order]})
# Empirical excess per unit A0 and band-level A_lambda/A0 (anchored on the reddest band)
eps_emp = R.mean(axis=0) / A0 if abs(A0) > 1e-6 else np.full(R.shape[1], np.nan)
k_theory = band_k(A0, np.median(tab.loc[in_fit, "bprp0"]), BANDS)[0]
k_emp = np.empty(len(BANDS))
k_emp[-1] = k_theory[-1]
for j in range(len(BANDS) - 2, -1, -1):
    k_emp[j] = k_emp[j + 1] + eps_emp[j]
band_table = pd.DataFrame({"band": BANDS, "A_lambda_over_A0_theory": k_theory,
                           "A_lambda_over_A0_empirical": k_emp,
                           "pct_diff": 100 * (k_emp - k_theory) / k_theory})
# What is left after subtracting the fitted reddening
s_after = svd_direction(R - A0 * E, ebar)
# Outliers: per-star A0 and residual norm (robust z)
full_after = res["R"] - A0 * res["E"]
resid_norm = np.linalg.norm(full_after, axis=1)
z_a0 = robust_z(tab["A0_i"].values)
z_res = robust_z(resid_norm)
outlier = res["valid"] & ((np.abs(z_a0) > outlier_nsigma) | (z_res > outlier_nsigma))
flags = pd.DataFrame({"resid_norm_after": resid_norm, "z_A0_i": z_a0, "z_resid": z_res,
                      "outlier": outlier}, index=tab.index)
if "Ks-[3.6]" in colours:                       # infrared-excess hint (disks)
    flags["ir_excess_candidate"] = res["valid"] & (
        robust_z(full_after[:, colours.index("Ks-[3.6]")]) > outlier_nsigma)
print('Colours: most affected colour: %s (v1 loading %+.3f)'
      % (ranking.iloc[0]["colour"], ranking.iloc[0]["v1_loading"]))
print('    after subtracting reddening: singular values %s (s1/s2 = %.2f)'
      % (np.array2string(s_after["s"], precision=4), s_after["s1_over_s2"]))
print('    outliers flagged: %d' % int(outlier.sum()))

# Differential reddening - conventional (mean-centred) PCA of the residuals
E_eff = differential_directions(E, res["kG"][in_fit], res["iso_slope"][in_fit])
cpca = centred_pca(R - A0 * E, E_eff, res["cov"][in_fit],
                   np.random.default_rng(rng_seed + 11), n_pca_noise)
diff_table = pd.DataFrame({"bprp0": tab.loc[in_fit, "bprp0"].values,
                           "differential_A0": cpca["differential_A0"],
                           "A0_i_minus_A0": tab.loc[in_fit, "A0_i"].values - A0}, index=tab.index[in_fit])
print('Differential reddening: centred PCA of residuals: star-to-star scatter is %.1fx what the '
      'photometric errors alone give' % cpca["scatter_over_noise"])
print('    PC1 holds %.1f%% of that variance (noise alone: up to %.1f%%); angle to the '
      'differential-reddening direction %.1f deg (cos %.3f; noise alone |cos| up to %.3f)'
      % (100 * cpca["var_frac"][0], 100 * cpca["var_frac1_noise_95pct"], cpca["angle_deg"],
         cpca["cosine"], cpca["abs_cos_noise_95pct"]))
print('    w1 = %s' % ", ".join(f"{c}: {x:+.3f}" for c, x in zip(colours, cpca["w1"])))
if cpca["aligned"]:
    print('    -> PC1 is the dust direction: the star-to-star variation is mainly differential reddening')
else:
    print('    -> PC1 is NOT the dust direction: the star-to-star variation is dominated by something else '
          '(e.g. isochrone mismatch, binaries, IR excess)')
print('    per-star differential A0: spread %.1fx photometric noise -> rms differential A0 = %.3f mag'
      % (cpca["dA0_spread_over_noise"], cpca["sigma_differential_A0"]))

# Spatial coherence: for each star, correlate its differential A0 with the mean of its k nearest
# neighbours on the sky; the p-value comes from shuffling the values among the positions
rng_spatial = np.random.default_rng(rng_seed + 13)
pos = sample.loc[diff_table.index, ["ra", "dec"]]
ra0, dec0 = float(np.median(pos["ra"])), float(np.median(pos["dec"]))
xy = np.column_stack([(pos["ra"].values - ra0) * np.cos(np.radians(dec0)) * 60.0,
                      (pos["dec"].values - dec0) * 60.0])                 #arcmin
tree = cKDTree(xy)
dA0 = diff_table["differential_A0"].values
med = float(np.median(dA0))
sig = MAD_TO_SIGMA * float(np.median(np.abs(dA0 - med)))
dA0_capped = np.clip(dA0, med - SPATIAL_WINSOR_NSIGMA * sig, med + SPATIAL_WINSOR_NSIGMA * sig)
total_rms = float(np.std(dA0_capped))
spatial_scales = []
idx_main = None
for k in spatial_k_scales:
    nb_dist, nb_idx = tree.query(xy, k=k + 1)
    nb_dist, nb_idx = nb_dist[:, 1:], nb_idx[:, 1:]                       # drop the star itself
    r_obs = neighbour_corr(dA0_capped, nb_idx)
    r_perm = np.array([neighbour_corr(rng_spatial.permutation(dA0_capped), nb_idx)
                       for _ in range(n_permutations)])
    r_adj = r_obs - float(np.mean(r_perm))                                # remove the small no-signal bias
    f = coherent_fraction(r_adj, k)
    f_lim = coherent_fraction(float(np.percentile(r_perm, 95)) - float(np.mean(r_perm)), k)
    spatial_scales.append({"k": int(k), "scale_arcmin": float(np.median(nb_dist[:, -1])),
                           "r": r_obs, "r_null_mean": float(np.mean(r_perm)),
                           "r_null_95pct": float(np.percentile(r_perm, 95)),
                           "p_value": float((1 + np.sum(r_perm >= r_obs)) / (1 + n_permutations)),
                           "coherent_fraction": f, "coherent_rms": float(np.sqrt(f) * total_rms),
                           # smallest coherent signal the test could detect (95%): upper limit if p >= 0.01
                           "coherent_rms_limit95": float(np.sqrt(max(f, f_lim)) * total_rms)})
    if k == spatial_k:
        idx_main = nb_idx
if idx_main is None:
    idx_main = tree.query(xy, k=spatial_k + 1)[1][:, 1:]
spatial_main = next((sc for sc in spatial_scales if sc["k"] == spatial_k), spatial_scales[0])
diff_table["neighbour_mean_differential_A0"] = dA0_capped[idx_main].mean(axis=1)
diff_table["x_arcmin"], diff_table["y_arcmin"] = xy[:, 0], xy[:, 1]

print('Spatial coherence: differential A0 vs position (%d shuffles of the positions)' % n_permutations)
for sm in spatial_scales:
    print('    k = %2d neighbours (~%.1f arcmin): r = %+.3f (shuffled: %+.3f, 95%% up to %+.3f), '
          'p = %.4f -> coherent share %.0f%%, coherent rms %.3f mag'
          % (sm["k"], sm["scale_arcmin"], sm["r"], sm["r_null_mean"], sm["r_null_95pct"], sm["p_value"],
             100 * sm["coherent_fraction"], sm["coherent_rms"]))
sm = spatial_main
if sm["p_value"] < 0.01:
    print('    -> spatially coherent at p = %.4f: about %.0f%% of the differential A0 (rms ~%.2f mag) '
          'varies smoothly across the field, as patchy dust does; the rest is star-to-star '
          '(binaries, age spread, noise)' % (sm["p_value"], 100 * sm["coherent_fraction"], sm["coherent_rms"]))
else:
    print('    -> not spatially coherent: the differential A0 does not follow position on the sky, '
          'so it is mainly star-to-star (binaries, age spread, noise), not patchy dust; '
          'coherent differential A0 < %.2f mag rms (95%%) on ~%.1f arcmin scales'
          % (sm["coherent_rms_limit95"], sm["scale_arcmin"]))

# Analysis tables
ranking.to_csv(out_dir / "colour_ranking.csv", index=False)
band_table.to_csv(out_dir / "band_extinction.csv", index=False)
per_star.to_csv(out_dir / "per_star_angles.csv")
flags.to_csv(out_dir / "outlier_flags.csv")
diff_table.to_csv(out_dir / "differential_A0.csv")


### VALIDATION
# Null test: dust-free synthetic stars -> A0 ~ 0, no dominant reddening direction
# Original method: 3-colour mean-centred SVD on the same dust-free stars (should fail)
# Injection-recovery: inject A0 = 0.2, 0.5, 1.0, 2.0 and recover A0 and direction.
# Bootstrap: resample the stars ~1000 times for 68 / 95% intervals.
# External comparison: per-star A0 vs Gaia GSP-Phot azero_gspphot.
# Synthetic stars sit at the real stars' absolute magnitudes and colours, with their errors.

def make_synthetic(iso, MG_template, bprp_template, err_template, d_pc, a0_true, bands, rng):
    # Intrinsic magnitudes from the isochrone (branch chosen by the template colour, so the
    # pre-main sequence is kept), reddened with the Gaia law + IR ratios at a0_true, plus
    # noise drawn with errors resampled from observed error rows
    iso_mags = interpolate_at_MG(iso, MG_template, bprp_template)[0]
    ok = np.all(np.isfinite(iso_mags), axis=1)
    iso_mags = iso_mags[ok]
    absm = iso_mags[:, [BANDS.index(b) for b in bands]]
    bprp0 = iso_mags[:, BANDS.index("BP")] - iso_mags[:, BANDS.index("RP")]
    app = absm + dist_modulus(d_pc) + band_k(max(a0_true, 0.0), bprp0, bands) * a0_true
    errs = np.asarray(err_template, float)[rng.integers(len(err_template), size=ok.sum())]
    df = pd.DataFrame(app + rng.normal(0.0, errs), columns=bands)
    for j, b in enumerate(bands):
        df[f"e_{b}"] = errs[:, j]
    df["true_bprp0"] = bprp0
    return df


val = {}
if run_validation:
    print('\nValidation:')
    rng = np.random.default_rng(rng_seed)
    # Template: M_G, (BP-RP)0 and error rows of the stars that fell on the isochrone
    MG_t = tab.loc[res["valid"], "MG"].values
    bprp_t = tab.loc[res["valid"], "bprp0"].values
    errs_t = res["errs"][res["valid"]]

    # Null test
    a0s, errs_a0, cos_null, cos_noise = [], [], [], []
    for _ in range(n_null):
        syn = make_synthetic(iso, MG_t, bprp_t, errs_t, d_c, 0.0, BANDS, rng)
        r = solve(syn, iso, BANDS, d_c, verbose=False)
        a0s.append(r["A0"])
        errs_a0.append(r["A0_err"])
        r_use = r["valid"] & r["keep"]
        r_ebar = r["E"][r_use].mean(axis=0)
        cos_null.append(abs(compare_vectors(svd_direction(r["R"][r_use], r_ebar)["v1"], r_ebar)["cosine"]))
        # Reference: pure noise with the same colour covariances (what noise alone produces)
        L = np.linalg.cholesky(r["cov"][r_use])
        noise = np.einsum("nij,nj->ni", L, rng.normal(size=L.shape[:2]))
        cos_noise.append(abs(compare_vectors(svd_direction(noise, r_ebar)["v1"], r_ebar)["cosine"]))
    a0s, errs_a0 = np.array(a0s), np.array(errs_a0)
    null = {"A0_mean": float(np.mean(a0s)), "A0_std": float(np.std(a0s)),
            "A0_mean_err": float(np.mean(errs_a0)),
            "abs_cos_v1_theory_median": float(np.median(cos_null)),
            "abs_cos_noise_only_95pct": float(np.percentile(cos_noise, 95)),
            # >1 means the formal weighted-mean error is too small (e.g. stars at the PMS
            # turn-on hook); quote the bootstrap interval for the real data in that case
            "error_calibration_factor": float(np.std(a0s) / np.mean(errs_a0))}
    null["pass"] = bool(abs(null["A0_mean"]) < 3 * null["A0_std"] / np.sqrt(len(a0s)) + 0.005
                        and null["abs_cos_v1_theory_median"] <= null["abs_cos_noise_only_95pct"])
    val["null_test"] = null
    print('Null test: A0 = %.4f, spread %.4f over %d runs (mean quoted error %.4f, '
          'calibration factor %.2f); median |cos(v1, theory)| = %.3f vs noise-only 95th pct %.3f -> %s'
          % (null["A0_mean"], null["A0_std"], len(a0s), null["A0_mean_err"], null["error_calibration_factor"],
             null["abs_cos_v1_theory_median"], null["abs_cos_noise_only_95pct"],
             "PASS" if null["pass"] else "CHECK"))

    # Original method on dust-free stars: X = [BP-RP, BP-G, G-RP], mean-centred, SVD
    bands3 = ["BP", "G", "RP"]
    syn = make_synthetic(iso, MG_t, bprp_t, errs_t[:, [BANDS.index(b) for b in bands3]], d_c, 0.0, bands3, rng)
    BP3, G3, RP3 = syn["BP"].values, syn["G"].values, syn["RP"].values
    X3 = np.column_stack([BP3 - RP3, BP3 - G3, G3 - RP3])
    _, s3, Vt3 = np.linalg.svd(X3 - X3.mean(axis=0), full_matrices=False)
    k3 = band_k(a0_start, np.median(syn["true_bprp0"]), bands3)[0]
    e3 = np.array([k3[0] - k3[2], k3[0] - k3[1], k3[1] - k3[2]])
    c3 = compare_vectors(Vt3[0] if Vt3[0] @ e3 >= 0 else -Vt3[0], e3)
    val["original_method"] = {"singular_values": s3, "rank": int(np.linalg.matrix_rank(X3)),
                              "cosine": c3["cosine"], "angle_deg": c3["angle_deg"]}
    print('Original-method check (no dust!): cos(v1, theory) = %.4f, angle %.2f deg; '
          'singular values %s (rank %d) -> high similarity without dust is why the method changed'
          % (c3["cosine"], c3["angle_deg"], np.array2string(s3, precision=3), val["original_method"]["rank"]))

    # Injection-recovery
    inj_rows = []
    for a0_inj in inject_a0:
        for _ in range(n_inject):
            syn = make_synthetic(iso, MG_t, bprp_t, errs_t, d_c, a0_inj, BANDS, rng)
            r = solve(syn, iso, BANDS, d_c, verbose=False)
            r_use = r["valid"] & r["keep"]
            e_inj = theory_vectors(a0_inj, syn["true_bprp0"].values, BANDS).mean(axis=0)
            r_sv = svd_direction(r["R"][r_use], e_inj)
            inj_rows.append({"A0_injected": a0_inj, "A0_recovered": r["A0"], "A0_err": r["A0_err"],
                             # uniform A0, no differential dust -> differential-reddening floor at this A0
                             "differential_A0_floor": centred_pca(
                                 r["R"][r_use] - r["A0"] * r["E"][r_use],
                                 differential_directions(r["E"][r_use], r["kG"][r_use], r["iso_slope"][r_use]),
                                 r["cov"][r_use], rng, n_pca_noise_inject)["sigma_differential_A0"],
                             "angle_to_injected_deg": compare_vectors(r_sv["v1"], e_inj)["angle_deg"],
                             "converged": r["converged"]})
    inj_runs = pd.DataFrame(inj_rows)
    inj = inj_runs.groupby("A0_injected").agg(A0_recovered_mean=("A0_recovered", "mean"),
                                              A0_recovered_std=("A0_recovered", "std"),
                                              A0_err_mean=("A0_err", "mean"),
                                              differential_floor_median=("differential_A0_floor", "median"),
                                              differential_floor_95pct=("differential_A0_floor",
                                                                        lambda x: np.percentile(x, 95)),
                                              angle_median_deg=("angle_to_injected_deg", "median")).reset_index()
    inj["bias"] = inj["A0_recovered_mean"] - inj["A0_injected"]
    inj["pass"] = (np.abs(inj["bias"]) < 2 * np.maximum(inj["A0_recovered_std"], inj["A0_err_mean"])) \
        & (inj["angle_median_deg"] < INJECT_ANGLE_TOL_DEG)
    val["injection_recovery"] = inj
    for _, row in inj.iterrows():
        print('Injection A0 = %.2f: recovered %.4f +/- %.4f (bias %+.4f), angle %.2f deg -> %s;  '
              'differential-A0 floor %.3f (95%%: %.3f)'
              % (row["A0_injected"], row["A0_recovered_mean"], row["A0_recovered_std"], row["bias"],
                 row["angle_median_deg"], "PASS" if row["pass"] else "CHECK",
                 row["differential_floor_median"], row["differential_floor_95pct"]))
    inj_runs.to_csv(out_dir / "validation_injection.csv", index=False)

    # Bootstrap: resample stars on the isochrone; redo cluster mean and direction
    valid_idx = np.where(res["valid"])[0]
    a0_i = tab["A0_i"].values
    var_i = tab["A0_i_err_phot"].values ** 2
    boot_rows = []
    for _ in range(n_bootstrap):
        idx = rng.choice(valid_idx, size=valid_idx.size, replace=True)
        wm = robust_weighted_mean(a0_i[idx], var_i[idx], clip_nsigma)
        R_b, E_b = res["R"][idx][wm["keep"]], res["E"][idx][wm["keep"]]
        ebar_b = E_b.mean(axis=0)
        sv_b = svd_direction(R_b, ebar_b)
        boot_rows.append({"A0": wm["mean"], "angle_deg": compare_vectors(sv_b["v1"], ebar_b)["angle_deg"],
                          "s1_over_s2": sv_b["s1_over_s2"]})
    boot = pd.DataFrame(boot_rows)
    val["bootstrap"] = {}
    for c in boot.columns:
        q = np.percentile(boot[c], [2.5, 16, 50, 84, 97.5])
        val["bootstrap"][c] = {"median": q[2], "68": (q[1], q[3]), "95": (q[0], q[4])}
        print('Bootstrap %-11s median %.4f  68%% [%.4f, %.4f]  95%% [%.4f, %.4f]'
              % (c, q[2], q[1], q[3], q[0], q[4]))
    boot.to_csv(out_dir / "validation_bootstrap.csv", index=False)

    # External comparison: per-star A0 vs Gaia GSP-Phot azero_gspphot
    t = tab.join(sample[["azero_gspphot", "azero_gspphot_lower", "azero_gspphot_upper"]])
    ok = t["in_isochrone"] & t["azero_gspphot"].notna()
    if ok.sum() == 0:
        print('External comparison: no stars with GSP-Phot azero')
        val["external"] = {"n": 0}
    else:
        d_ext = t.loc[ok, "A0_i"] - t.loc[ok, "azero_gspphot"]
        ext = {"n": int(ok.sum()), "median_A0_i_minus_gspphot": float(np.median(d_ext)),
               "robust_std": float(MAD_TO_SIGMA * np.median(np.abs(d_ext - np.median(d_ext)))),
               "gspphot_median_azero": float(np.median(t.loc[ok, "azero_gspphot"])),
               "pearson_r": float(np.corrcoef(t.loc[ok, "A0_i"], t.loc[ok, "azero_gspphot"])[0, 1])}
        comb = np.sqrt(res["A0_err"] ** 2 + (ext["robust_std"] / np.sqrt(ext["n"])) ** 2)
        ext["cluster_offset_sigma"] = float((res["A0"] - ext["gspphot_median_azero"]) / comb)
        val["external"] = ext
        print('External: pipeline A0 = %.3f vs GSP-Phot median azero = %.3f (%d stars, offset %.1f sigma); '
              'per-star median diff %.3f, robust std %.3f, r = %.2f'
              % (res["A0"], ext["gspphot_median_azero"], ext["n"], ext["cluster_offset_sigma"],
                 ext["median_A0_i_minus_gspphot"], ext["robust_std"], ext["pearson_r"]))


### FIGURES
# Gaia and 2MASS CMDs with the isochrone (A0 = 0 and fitted) and reddening vectors,
# the singular-value spectrum, the A0 convergence history, the angle to each star's theory
# vector, colour excesses vs the law, differential reddening on the sky, injection-recovery
# and an interactive 3D plot.

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "legend.fontsize": 10})

BAND_TEX = {"BP": r"$G_\mathrm{BP}$", "G": r"$G$", "RP": r"$G_\mathrm{RP}$", "J": r"$J$", "H": r"$H$",
            "Ks": r"$K_\mathrm{s}$", "I1": "[3.6]", "I2": "[4.5]"}
BAND_HTML = {"BP": "G<sub>BP</sub>", "G": "G", "RP": "G<sub>RP</sub>", "J": "J", "H": "H",
             "Ks": "K<sub>s</sub>", "I1": "[3.6]", "I2": "[4.5]"}
COLOUR_TEX = [f"{BAND_TEX[a]} $-$ {BAND_TEX[b]}" for a, b in zip(BANDS[:-1], BANDS[1:])]
COLOUR_HTML = [f"{BAND_HTML[a]} − {BAND_HTML[b]}" for a, b in zip(BANDS[:-1], BANDS[1:])]
BPRP0_TEX = r"$(G_\mathrm{BP} - G_\mathrm{RP})_0$"
BPRP0_HTML = "(G<sub>BP</sub> − G<sub>RP</sub>)<sub>0</sub>"
V1_TEX = r"$\mathbf{v}_1$"
V1_HTML = "<b>v</b><sub>1</sub>"


def save_figure(fig, name):
    fig.savefig(fig_dir / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def add_reddening_arrow(ax, dx, dy, label):
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = sorted(ax.get_ylim())
    x0 = x_lo + 0.08 * (x_hi - x_lo)
    y0 = y_hi - 0.10 * (y_hi - y_lo) - dy
    ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2, mutation_scale=15))
    ax.text(x0, y0 - 0.03 * (y_hi - y_lo), label, color="tab:green", fontsize=10, va="bottom")


def cmd_limits(ax, x, y, pad_x=0.15, pad_y=0.6):
    ax.set_xlim(np.nanmin(x) - pad_x, np.nanmax(x) + pad_x)
    ax.set_ylim(np.nanmax(y) + pad_y, np.nanmin(y) - pad_y)


def add_3d_arrow(fig, v, scale, name, color, width, dash="solid", showlegend=True, visible=True):
    w = np.asarray(v, float)
    w = w / np.linalg.norm(w) * scale
    fig.add_trace(go.Scatter3d(x=[0, w[0]], y=[0, w[1]], z=[0, w[2]], mode="lines",
                               line=dict(color=color, width=width, dash=dash), name=name,
                               showlegend=showlegend, visible=visible))


print('\nFigures and summary:')
mags, dm_fit = res["mags"], res["DM"]
is_outlier = flags["outlier"].values
is_flagged = tab["branch_flag"].values
iso_bprp0 = iso_band(iso, "BP") - iso_band(iso, "RP")
kiso = band_k(A0, iso_bprp0, ["BP", "G", "RP", "J", "Ks"])
bprp = mags[:, BANDS.index("BP")] - mags[:, BANDS.index("RP")]
G_mag = mags[:, BANDS.index("G")]
J_mag, Ks_mag = mags[:, BANDS.index("J")], mags[:, BANDS.index("Ks")]
JK = J_mag - Ks_mag

# Colour-magnitude diagrams
fig, axes = plt.subplots(1, 2, figsize=(13, 7))
for ax, x, y, iso_x0, iso_y0, kx, ky in (
        (axes[0], bprp, G_mag, iso_bprp0, iso_band(iso, "G"), kiso[:, 0] - kiso[:, 2], kiso[:, 1]),
        (axes[1], JK, Ks_mag, iso_band(iso, "J") - iso_band(iso, "Ks"), iso_band(iso, "Ks"),
         kiso[:, 3] - kiso[:, 4], kiso[:, 4])):
    ax.plot(iso_x0, iso_y0 + dm_fit, "--", c="0.6", lw=1, zorder=1, label="Isochrone, $A_0 = 0$")
    ax.plot(iso_x0 + kx * A0, iso_y0 + dm_fit + ky * A0, c="tab:blue", lw=1.5, zorder=1,
            label=f"Isochrone, $A_0 = {A0:.2f}$")
    ax.scatter(x, y, s=8, c="0.2", zorder=2, label="Members")
    ax.scatter(x[is_flagged], y[is_flagged], s=45, facecolors="none", edgecolors="tab:orange", lw=1.2,
               zorder=3, label="PMS/MS branch flag")
    ax.scatter(x[is_outlier], y[is_outlier], s=50, marker="x", c="tab:red", lw=1.5, zorder=4, label="Outlier")
    cmd_limits(ax, x, y)
kv = band_k(1.0, np.nanpercentile(bprp, 10), ["BP", "G", "RP"])[0]
add_reddening_arrow(axes[0], kv[0] - kv[2], kv[1], "$A_0 = 1$ mag")
kj, kk = band_k(1.0, 0.5, ["J", "Ks"])[0]
add_reddening_arrow(axes[1], (kj - kk) * 2, kk * 2, "$A_0 = 2$ mag")
axes[0].set_xlabel(f"{BAND_TEX['BP']} $-$ {BAND_TEX['RP']} [mag]"); axes[0].set_ylabel("$G$ [mag]"); axes[0].set_title("M36 Gaia CMD")
axes[1].set_xlabel(r"$J - K_\mathrm{s}$ [mag]"); axes[1].set_ylabel(r"$K_\mathrm{s}$ [mag]"); axes[1].set_title("M36 2MASS CMD")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False)
fig.tight_layout(w_pad=4)
save_figure(fig, "cmd.png")

# Differential reddening and spatial coherence: per star on the sky, smoothed over its nearest
# neighbours (patchy dust shows up as coherent patches), and the centred-PCA variance spectrum
d_sky = diff_table.join(sample[["ra", "dec"]])
lim = np.nanpercentile(np.abs(d_sky["differential_A0"]), 95)
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), layout="constrained")
fig.get_layout_engine().set(w_pad=0.2, h_pad=0.15, wspace=0.08)
panels = [("differential_A0", lim, "Differential $A_0$ [mag]",
           f"Differential $A_0$ per star\nRMS {cpca['sigma_differential_A0']:.3f} mag"),
          ("neighbour_mean_differential_A0", lim / 2, f"Mean of {spatial_main['k']} nearest neighbours [mag]",
           f"Smoothed over {spatial_main['k']} nearest neighbours\n"
           f"$r$ = {spatial_main['r']:+.2f}, $p$ = {spatial_main['p_value']:.3g}")]
for ax, (col, vlim, cbar, title) in zip(axes[:2], panels):
    sc = ax.scatter(d_sky["ra"], d_sky["dec"], c=d_sky[col], cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                    s=22, edgecolors="0.4", linewidths=0.3)
    fig.colorbar(sc, ax=ax, label=cbar, shrink=0.85)
    ax.set_title(title)
    ax.invert_xaxis()
    ax.set_xlabel("RA [°]"); ax.set_ylabel("Dec [°]")
    ax.set_aspect(1.0 / np.cos(np.radians(np.median(d_sky["dec"]))), adjustable="datalim")
ax = axes[2]
pcs = np.arange(1, cpca["var_frac"].size + 1)
ax.bar(pcs, 100 * cpca["var_frac"], color="0.55", width=0.7)
ax.axhline(100 * cpca["var_frac1_noise_95pct"], color="tab:red", ls="--", lw=1.2,
           label="PC1 share from noise alone (95th percentile)")
ax.set_xticks(pcs)
ax.set_ylim(0, 1.15 * max(100 * cpca["var_frac"][0], 100 * cpca["var_frac1_noise_95pct"]))
ax.set_xlabel("Principal component (mean-centred residuals)")
ax.set_ylabel("Share of star-to-star variance [%]")
ax.set_title(f"Centred PCA spectrum\nPC1 is {cpca['angle_deg']:.0f}° from the dust direction")
ax.legend(loc="upper right", frameon=False)
save_figure(fig, "differential_reddening.png")

# Interactive 3D plot of the residuals on the three colours with the largest |v1| loadings
ax_idx = list(np.sort(np.argsort(-np.abs(v1))[:3]))
R3, E3 = R[:, ax_idx], E[:, ax_idx]
scale = 1.1 * np.percentile(np.linalg.norm(R3, axis=1), 95)
fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(x=R3[:, 0], y=R3[:, 1], z=R3[:, 2], mode="markers", name="Residual rows <b>r</b><sub>i</sub>",
                             marker=dict(size=4, opacity=0.85, color=tab["bprp0"].values[in_fit],
                                         colorscale="RdYlBu_r",
                                         colorbar=dict(title=BPRP0_HTML, len=0.75, thickness=15, x=1.0, xanchor="left", y=0.45))))
add_3d_arrow(fig3d, v1[ax_idx], scale, f"{V1_HTML} (empirical direction)", "black", 7)
add_3d_arrow(fig3d, ebar[ax_idx], scale, "Mean theory vector (Gaia law)", "green", 7, dash="dash")
seg = []
for i in np.linspace(0, len(E3) - 1, min(10, len(E3))).astype(int):
    seg += [np.zeros(3), unit(E3[i]) * scale * 0.8, np.full(3, np.nan)]
seg = np.array(seg)
fig3d.add_trace(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines", connectgaps=False,
                             line=dict(color="rgba(0,128,0,0.4)", width=2), name="Per-star theory vectors <b>e</b><sub>i</sub>",
                             visible="legendonly"))
side = np.cross(unit(v1[ax_idx]), [0.0, 0.0, 1.0])
side = unit(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0.0, 0.0])
side = side if side[0] + side[1] >= 0 else -side
eye = 2.1 * side + np.array([0.0, 0.0, 0.6])
fig3d.update_layout(
    title=dict(text=f"M36 colour-excess residuals (angle between {V1_HTML} and the law: {cmp_mean['angle_deg']:.1f}°)", x=0.5),
    scene=dict(xaxis_title=f"{COLOUR_HTML[ax_idx[0]]} [mag]", yaxis_title=f"{COLOUR_HTML[ax_idx[1]]} [mag]",
               zaxis_title=f"{COLOUR_HTML[ax_idx[2]]} [mag]", aspectmode="cube",
               camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))),
    legend=dict(x=0.0, xanchor="left", y=1.0, yanchor="top", bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(l=0, r=80, t=60, b=130), height=840,
    annotations=[dict(
        text=(f"Each point is one star's colour excess (observed minus isochrone colours), coloured by {BPRP0_HTML},<br>"
              f"shown on the three colours with the largest {V1_HTML} loadings ({', '.join(COLOUR_HTML[i] for i in ax_idx)}).<br>"
              f"Black: empirical reddening direction {V1_HTML} from the SVD. Dashed green: mean Gaia EDR3 + IR law.<br>"
              f"Faint green (click in legend): per-star law directions, which vary with intrinsic colour.<br>"
              f"On these three axes {V1_HTML} and the law differ by {compare_vectors(v1[ax_idx], ebar[ax_idx])['angle_deg']:.1f}°; "
              f"the {cmp_mean['angle_deg']:.1f}° in the title uses all {len(colours)} colours."),
        xref="paper", yref="paper", x=0.0, y=-0.02, xanchor="left", yanchor="top",
        align="left", showarrow=False, font=dict(size=12))])
fig3d.write_html(fig_dir / "residuals_3d.html", include_plotlyjs="cdn")

# Singular-value spectrum
fig, ax = plt.subplots(figsize=(7, 4.5))
comps = np.arange(1, svd["s"].size + 1)
ax.semilogy(comps, svd["s"], "o-", label=r"Residuals $\mathbf{R}$")
ax.semilogy(comps, s_after["s"], "s--", alpha=0.7, label="After subtracting the fitted reddening")
ax.set_xticks(comps)
ax.set_xlabel("Component"); ax.set_ylabel("Singular value"); ax.legend(frameon=False)
ax.set_title(f"Singular-value spectrum ($s_1/s_2$ = {svd['s1_over_s2']:.1f})")
save_figure(fig, "singular_values.png")

# A0 convergence history
fig, ax = plt.subplots(figsize=(7, 4.5))
hist = res["history"]
ax.errorbar(hist["iteration"], hist["A0_out"], yerr=hist["A0_err"], marker="o", capsize=3)
ax.axhline(A0, color="0.5", ls=":", lw=1, label=f"Converged $A_0$ = {A0:.3f} mag")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel("Iteration"); ax.set_ylabel("$A_0$ [mag]"); ax.legend(frameon=False)
ax.set_title("$A_0$ convergence history")
save_figure(fig, "A0_convergence.png")

# Angle between the empirical direction and each star's theory vector
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(per_star["bprp0"], per_star["angle_v1_to_e_i_deg"], s=14, c="0.2")
ax.set_xlabel(f"{BPRP0_TEX} [mag]"); ax.set_ylabel(f"Angle between {V1_TEX} and " + r"$\mathbf{e}_i$ [°]")
ax.set_title("Empirical direction vs. each star's theory vector")
save_figure(fig, "angle_vs_bprp0.png")

# Colour excesses per unit A0: law vs least squares vs PCA (v1), bootstrap errors
rng_curve = np.random.default_rng(rng_seed)
eps_law = ebar
eps_lsq = R.mean(axis=0) / A0
eps_pca = (ebar @ v1) * v1
fit_idx = np.where(in_fit)[0]
a0_fit = tab["A0_i"].values
var_fit = tab["A0_i_err_phot"].values ** 2
eps_lsq_boot, eps_pca_boot = [], []
for _ in range(n_bootstrap):
    idx = rng_curve.choice(fit_idx, size=fit_idx.size, replace=True)
    wm = robust_weighted_mean(a0_fit[idx], var_fit[idx], clip_nsigma)
    R_b, E_b = res["R"][idx][wm["keep"]], res["E"][idx][wm["keep"]]
    ebar_b = E_b.mean(axis=0)
    v1_b = svd_direction(R_b, ebar_b)["v1"]
    eps_lsq_boot.append(R_b.mean(axis=0) / wm["mean"])
    eps_pca_boot.append((ebar_b @ v1_b) * v1_b)
eps_lsq_err = np.std(eps_lsq_boot, axis=0)
eps_pca_err = np.std(eps_pca_boot, axis=0)

excess = pd.DataFrame({"colour": colours, "E_over_A0_law": eps_law,
                       "E_over_A0_lsq": eps_lsq, "lsq_err": eps_lsq_err,
                       "E_over_A0_pca": eps_pca, "pca_err": eps_pca_err,
                       "lsq_minus_law": eps_lsq - eps_law, "pca_minus_law": eps_pca - eps_law,
                       "lsq_over_law": eps_lsq / eps_law, "pca_over_law": eps_pca / eps_law})
excess.to_csv(out_dir / "colour_excess.csv", index=False)

x = np.arange(len(colours))
dx = 0.12
fig, (ax, axd) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
ax.plot(x, eps_law, "-", marker="_", ms=14, color="0.3", lw=1.5, label="Gaia EDR3 + IR law")
ax.errorbar(x - dx, eps_lsq, yerr=eps_lsq_err, fmt="o", color="tab:blue", capsize=3,
            label=f"Least squares ($A_0$ = {A0:.3f} mag)")
ax.errorbar(x + dx, eps_pca, yerr=eps_pca_err, fmt="s", mfc="none", color="tab:orange", capsize=3,
            label=r"PCA direction $\mathbf{v}_1$")
ax.set_ylabel(r"Colour excess per unit extinction, $E(\mathrm{colour})\,/\,A_0$")
ax.legend(frameon=False, loc="upper right")
ax.set_title(f"M36 colour excesses (angle between {V1_TEX} and the law: {cmp_mean['angle_deg']:.1f}°)")
axd.axhline(0.0, color="0.3", lw=1)
axd.errorbar(x - dx, eps_lsq - eps_law, yerr=eps_lsq_err, fmt="o", color="tab:blue", capsize=3)
axd.errorbar(x + dx, eps_pca - eps_law, yerr=eps_pca_err, fmt="s", mfc="none", color="tab:orange", capsize=3)
axd.set_xticks(x)
axd.set_xticklabels(COLOUR_TEX)
axd.set_xlabel("Adjacent-band colour (blue to red)")
axd.set_ylabel(r"Measured $-$ law")
fig.tight_layout(h_pad=2)
save_figure(fig, "colour_excess.png")

# Injection-recovery
if val:
    lim_inj = [0, max(inj["A0_injected"]) * 1.1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(lim_inj, lim_inj, "k--", lw=1, label="Perfect recovery")
    axes[0].errorbar(inj["A0_injected"], inj["A0_recovered_mean"], yerr=inj["A0_recovered_std"],
                     fmt="o", capsize=4, label="Recovered (mean ± std)")
    axes[0].set_xlabel("Injected $A_0$ [mag]"); axes[0].set_ylabel("Recovered $A_0$ [mag]")
    axes[0].set_title("Amount"); axes[0].legend(frameon=False)
    axes[1].plot(inj["A0_injected"], inj["angle_median_deg"], "o-")
    axes[1].axhline(INJECT_ANGLE_TOL_DEG, color="tab:red", ls="--", lw=1, label="Pass limit")
    axes[1].set_xlabel("Injected $A_0$ [mag]"); axes[1].set_ylabel("Median angle to injected direction [°]")
    axes[1].set_title("Direction"); axes[1].legend(frameon=False)
    fig.suptitle("Injection–recovery test")
    fig.tight_layout(w_pad=4)
    save_figure(fig, "injection_recovery.png")

stars.to_csv(out_dir / "m36_all_stars_final.csv", index=False)


### SUMMARY
# Plain-text summary of every headline number, printed and saved as output/summary.txt.

def fmt(x, spec=".3f", missing="-"):
    # Format number, or `missing` for None / NaN
    try:
        return missing if x is None or not np.isfinite(float(x)) else format(float(x), spec)
    except (TypeError, ValueError):
        return missing


def add_heading(lines, title):
    lines.extend(["", title, "-" * len(title)])


def add_line(lines, label, text, unit=""):
    lines.append(f"  {label:<34}{text}" + (f" {unit}" if unit else ""))


def boot68(key, spec=".3f"):
    # Bootstrap 68% interval of `key`, or "-" if the validation was not run
    b = val.get("bootstrap", {}).get(key)
    return f"{fmt(b['68'][0], spec)} - {fmt(b['68'][1], spec)}" if b else "-"


n_after = cut_log.set_index("name")["n_after"]
# Differential-A0 floor: 95th percentile with no differential dust, at the injection level nearest A0
floor = (float(inj.iloc[int(np.argmin(np.abs(inj["A0_injected"].values - A0)))]["differential_floor_95pct"])
         if val else None)
summary = ["REDDENING PIPELINE - M36 (NGC 1960) SUMMARY"]

add_heading(summary, "Data")
add_line(summary, "members (Panja et al. 2021)", expected_members)
add_line(summary, "with a Gaia DR3 match", n_after.get("match", "-"))
add_line(summary, "after RUWE and flux-excess cuts", n_after.get("flux_excess", "-"))
add_line(summary, "with complete photometry", n_after.get("sample", "-"))

add_heading(summary, "Distance")
add_line(summary, "cluster distance d_c", f"{fmt(dist['d_c_pc'], '.1f')} (+{fmt(dist['d_c_hi84'] - dist['d_c_pc'], '.1f')} "
         f"/ -{fmt(dist['d_c_pc'] - dist['d_c_lo16'], '.1f')}) pc  (statistical only)")
add_line(summary, "distance modulus", f"{fmt(dist['DM'])} +/- {fmt(dist['DM_err'])}", "mag")
add_line(summary, "per-star median distance", fmt(dist['median_per_star_r_med_pc'], '.0f'), "pc")
add_line(summary, "Panja et al. 2021 median", fmt(dist.get('panja_median_dist_pc'), '.0f'), "pc")
add_line(summary, "Bailer-Jones et al. 2021 median", fmt(dist.get('bailer_jones_median_r_med_geo_pc'), '.0f'), "pc")
add_line(summary, "per-star minus Bailer-Jones", fmt(dist.get('median_per_star_minus_bj_pc'), '+.0f'), "pc (median)")
add_line(summary, "DR3 (corrected) - DR2 parallax", fmt(dist.get('median_plx_dr3corr_minus_dr2_mas'), '+.4f'), "mas (median)")
add_line(summary, "literature (Panja et al. 2021)",
         f"{fmt(literature['distance_pc'], '.0f')} +/- {fmt(literature['distance_err_pc'], '.0f')}", "pc")

add_heading(summary, "Model")
add_line(summary, "PARSEC isochrone",
         f"log(age) = {iso['logage']:.2f} ({10 ** iso['logage'] / 1e6:.1f} Myr), [M/H] = {iso['mh']:.2f}")
add_line(summary, "extinction law", "Gaia EDR3 (Fitzpatrick 2019); NIR: Wang & Chen 2019; IRAC: Indebetouw et al. 2005")

add_heading(summary, "Reddening amount")
add_line(summary, "A0 (least squares)", f"{fmt(res['A0'])} +/- {fmt(res['A0_err'])}", "mag")
add_line(summary, "  bootstrap 68%", boot68("A0"), "mag")
add_line(summary, "E(B-V) = A0/3.1", fmt(res["A0"] / 3.1), "mag")
add_line(summary, "stars used / iterations", f"{res['n_used']} / {res['iterations']}")
add_line(summary, "intrinsic scatter sigma_int", fmt(res["sigma_int"]), "mag")
add_line(summary, "literature E(B-V)", f"{literature['ebv_range'][0]:.2f} - {literature['ebv_range'][1]:.2f} "
         f"(A_V ~ {literature['rv'] * literature['ebv_range'][0]:.2f} - {literature['rv'] * literature['ebv_range'][1]:.2f})")

add_heading(summary, "Reddening direction")
add_line(summary, "s1/s2 (one dominant direction?)", fmt(svd["s1_over_s2"], ".2f"))
add_line(summary, "angle v1 to Gaia/IR law", fmt(cmp_mean["angle_deg"], ".1f"), "deg")
add_line(summary, "  bootstrap 68%", boot68("angle_deg", ".1f"), "deg")
add_line(summary, "cosine similarity", fmt(cmp_mean["cosine"], ".4f"))
add_line(summary, "most affected colour", ranking.iloc[0]["colour"])
for c, p in zip(colours, cmp_mean["pct_diff"]):
    add_line(summary, f"  % difference {c}", fmt(p, "+.1f"), "%")

add_heading(summary, "Star-to-star variation (differential reddening)")
add_line(summary, "centred PC1 share of variance", fmt(100 * cpca["var_frac"][0], ".0f"), "%")
add_line(summary, "centred PC1 angle to dust dir.", fmt(cpca["angle_deg"], ".1f"), "deg")
add_line(summary, "rms along dust direction", fmt(cpca["sigma_differential_A0"]), "mag")
add_line(summary, "  floor with no differential dust", fmt(floor), "mag")
add_line(summary, f"spatial coherence r (k = {spatial_k})", fmt(spatial_main["r"], "+.3f"))
add_line(summary, f"  p-value ({n_permutations} shuffles)", fmt(spatial_main["p_value"], ".4f"))
add_line(summary, "coherent (patchy-dust) rms", fmt(spatial_main["coherent_rms"]) if spatial_main["p_value"] < 0.01
         else "< " + fmt(spatial_main["coherent_rms_limit95"]), "mag")

add_heading(summary, "Validation")
if not val:
    summary.append("  not run (run_validation = False)")
else:
    add_line(summary, "null test (no dust)", ("PASS" if null["pass"] else "CHECK") + f", A0 = {fmt(null['A0_mean'], '+.4f')}")
    add_line(summary, "  quoted-error calibration factor", fmt(null["error_calibration_factor"], ".2f"))
    add_line(summary, "injection-recovery levels passed", f"{int(inj['pass'].sum())} / {len(inj)}")
    add_line(summary, "  largest |bias|", fmt(inj["bias"].abs().max(), ".4f"), "mag")
    add_line(summary, "original 3-colour method, no dust", f"cos = {fmt(val['original_method']['cosine'], '.4f')}")
    add_line(summary, "Gaia GSP-Phot median A0", fmt(val["external"].get("gspphot_median_azero")), "mag")
    add_line(summary, "  pipeline A0 minus GSP-Phot", fmt(val["external"].get("cluster_offset_sigma"), "+.1f"), "sigma")

add_heading(summary, "Notes")
summary.append("  * d_c error is statistical; Gaia parallax systematics (~0.01 mas) add ~1% (~14 pc).")
summary.append("  * A0 depends on the distance modulus (closer -> lower A0) and on the isochrone age.")
summary.append("  * Quote the bootstrap intervals; the null test shows the quoted errors run small.")
summary.append("  * The % differences for IRAC colours are large partly because the expected values are tiny.")

(out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
print('')
for line in summary:
    print(line)

#### END