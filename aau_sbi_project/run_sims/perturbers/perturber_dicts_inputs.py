"Generating the subhalo pertuber dictionaries"

import numpy as np
from tqdm import tqdm
import subprocess
from datetime import datetime, timezone
from scipy.stats import uniform, loguniform
import json

import os
import pickle

import sys
sys.path.append("/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/")
#import Nbody_streams.nbody_streams as nbody
import sims_funcs_refined as sfr

import agama 
agama.setUnits(mass=1, length=1, velocity=1) #Msol, kpc, km/s. Time is in kpc/(kms/s)

#######################
##Prior configuration##
#######################

PRIORS = {
    "mass": {
        "distrib": "loguniform",
        "args": {"a":0.1, "b":50}, 
        "scale": 1e7,
        "units": "Msun",},
        #sampled in units of 1e7 Msun, and then rescaled

    "scale_radius": {
        "distrib": "loguniform", 
        "args": {"a": 0.01, "b": 2.5}, 
        "units": "kpc",},

    "v_perp":{
        "distrib":"uniform",
        "args": {"loc": 0, "scale": 200}, 
        "units": "km/s",},
    #v_perp only samples magnitude of velocity, directionality gets added in later

    "v_para": {
        "distrib": "uniform",
        "args": {"loc":-200, "scale":400},
        "units": "km/s", }, 
    
    "angle_pos": {
        "distrib": "uniform", 
        "args": {"loc": 0, "scale": 180}, 
        "units": "deg",}, 

    "angle_delta": {
        "distrib": "uniform", 
        "args": {"loc": -90, "scale": 180}, 
        "units": "deg",},
    #add this factor onto angle pos to get angle vel, avoiding angles that ever allow the subhalo to move toward the stream

    "impact_param": {
        "distrib": "uniform", 
        "args": {"loc": 0.5, "scale": 4.5}, 
        "units": "scale_radii",}, #ends up being loguniform distrib in kpc

    "impact_time": {
        "distrib": "uniform", 
        "args": {"loc":0.05, "scale": 0.45}, 
        "units": "Gyr ago",}, 

    "phi1_impact_today":{
        "distrib": "uniform", 
        "args": {"loc": -15, "scale": 7}, 
        "units": "deg",},}

DELTA_PHI1 = 0.5 #deg, fixed, needed for Arpit's ps implementation

DIST_REGISTRY = {
    "uniform":uniform, 
    "loguniform":loguniform, }

def build_distrib(prior):
    """Building a scipy compatible distribution from the above PRIORS entry."""
    return DIST_REGISTRY[prior["distrib"]](**prior["args"])


##########################
##Sampling configuration##
##########################

VARY = ["mass", "scale_radius"]  # <- parameters actually drawn from PRIORS

FIXED_VALUES = {
    "v_perp": 50.0,
    "v_para": 50.0,
    "angle_pos": 25.0,
    "angle_delta": 45.0,
    "impact_param": 0.5, #kpc        
    "impact_time": 0.25,
    "phi1_impact_today": -10.0,
}

# Sanity check at import time, so a typo fails loudly and immediately
_all_keys = set(VARY) | set(FIXED_VALUES.keys())
_missing = set(PRIORS.keys()) - _all_keys
_extra = _all_keys - set(PRIORS.keys())
assert not _missing, f"These PRIORS keys are neither varied nor fixed: {_missing}"
assert not _extra, f"VARY/FIXED_VALUES reference unknown keys: {_extra}"
assert not (set(VARY) & set(FIXED_VALUES.keys())), \
    f"These keys are in both VARY and FIXED_VALUES: {set(VARY) & set(FIXED_VALUES.keys())}"


#####################
##Run configuration##
#####################

RUN_TAG = "pert_813_mocks" 
RUN_CONFIG = {
    "seed": 123,
    "N_target": 100,
    "N_oversample": 200,
    #"delta_V_cut_kms": 0.3, # keep only perturbers with ΔV > this value
    #"time_window_cut_gyr": 3.0, #try for only the most impulsive impacts, added 8/2
    "batch_size": 100,
    "base_path": "/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/",
    "output_dir": os.path.join("/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/perturbers",
                               RUN_TAG,),
    "unperturbed_stream_file": "stream_unperturbed_622.pkl",
    "potential_files": {
        "mw": "McMillan17_nora.ini",
        "lmc": "LMC_nora.ini",
        "acc_mw": "accMW",
        "traj_lmc": "trajLMC",
    },
    "G_units": 4.302e-6,  # kpc (km/s)^2 / Msol
}
 
G_units = RUN_CONFIG["G_units"]


#############################################
##Load the unperturbed stream and potential##
#############################################

path = os.path.join(RUN_CONFIG["base_path"], RUN_CONFIG["unperturbed_stream_file"])
with open(path, "rb") as f:
    stream_data = pickle.load(f)

stream_unperturb = stream_data["stream"]
stream_unperturb['phi1'] = stream_data['phi1']
stream_unperturb['phi2'] = stream_data['phi2']
meta = stream_data["metadata"]

## potential file paths
pf = RUN_CONFIG["potential_files"]
potMW_path   = os.path.join(RUN_CONFIG["base_path"], pf["mw"])
potLMC_path  = os.path.join(RUN_CONFIG["base_path"], pf["lmc"])
accMW_path   = os.path.join(RUN_CONFIG["base_path"], pf["acc_mw"])
trajLMC_path = os.path.join(RUN_CONFIG["base_path"], pf["traj_lmc"])

## potential models to load
potMW   = agama.Potential(file=potMW_path)
accMW   = np.loadtxt(accMW_path)
trajLMC = np.loadtxt(trajLMC_path)
potacc  = agama.Potential(type='UniformAcceleration', file=accMW)
potLMC  = agama.Potential(file=potLMC_path)
potLMCm = agama.Potential(potential=potLMC, center=trajLMC)
potTotal= agama.Potential(potMW, potLMCm, potacc)

##########
##Sample##
##########

def sample_priors(rng, n):
    """Draw n samples for every PRIORS entry: sampled if in VARY, else held fixed."""
    samples = {}
    for name, prior in PRIORS.items():
        if name in VARY:
            distrib = build_distrib(prior)
            vals = distrib.rvs(n, random_state=rng)
            if "scale" in prior:
                vals = vals * prior["scale"]
            samples[name] = vals
        else:
            samples[name] = np.full(n, FIXED_VALUES[name])
    return samples


####################
##Metadata helpers##
####################
 
def get_git_commit(path="."):
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None
 
 
def build_run_metadata(pass_rate, n_filtered, n_target_actual):
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.abspath(__file__),
        "git_commit": get_git_commit(os.path.dirname(os.path.abspath(__file__))),
        "hostname": os.uname().nodename,
        "priors": PRIORS,
        "vary": VARY,
        "fixed_values": FIXED_VALUES,
        "delta_phi1_fixed": DELTA_PHI1,
        "run_config": RUN_CONFIG,
        "results": {
            "pass_rate": pass_rate,
            "n_filtered": n_filtered,
            "n_target_actual": n_target_actual,
        },
    }
 
 
def save_run_metadata(metadata, output_dir):
    path = os.path.join(output_dir, "run_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved run metadata -> {path}")

########
##Main##
########


def main():
    cfg = RUN_CONFIG
    rng = np.random.default_rng(cfg["seed"])

    s = sample_priors(rng, cfg["N_oversample"])

    mass = s["mass"]
    scale_radius = s["scale_radius"]
    v_perp = s["v_perp"]
    v_para = s["v_para"]
    angle_pos = s["angle_pos"]
    angle_delta = s["angle_delta"]
    angle_vel = angle_pos + angle_delta
    if "impact_param" in VARY:
    # sampled in scale_radii, convert to kpc
        impact_param = s["impact_param"] * scale_radius
    else:
        # held fixed at an absolute kpc value — do NOT multiply by scale_radius
        impact_param = s["impact_param"]  # already kpc, shape (n,) from np.full
    impact_time = -s["impact_time"]
    phi1_impact_today = s["phi1_impact_today"]
 
    time_window = sfr.compute_time_window(v_perp, v_para, impact_param)
    

    # ΔV detectability cut, removing unperturbed streams 
    #v_rel_mag = np.sqrt(v_perp**2 + v_para**2)
    #delta_V = 2 * G_units * mass / (impact_param * v_rel_mag)
    
    #delta_v_mask = delta_V > cfg["delta_V_cut_kms"]
    #time_window_mask = time_window < cfg["time_window_cut_gyr"] #added 8/2
    #mask = delta_v_mask #& time_window_mask #added 8/2
    
    #print(f"  (ΔV pass rate: {delta_v_mask.sum() / cfg['N_oversample']:.3f}, "
      #f"time_window pass rate: {time_window_mask.sum() / cfg['N_oversample']:.3f})")


    #Mask all the values
    # mass = mass[mask]
    # scale_radius = scale_radius[mask]
    # v_perp = v_perp[mask]
    # v_para = v_para[mask]
    # angle_pos = angle_pos[mask]
    # angle_delta = angle_delta[mask]
    # angle_vel = angle_vel[mask]
    # impact_param = impact_param[mask]
    # impact_time = impact_time[mask]
    # phi1_impact_today = phi1_impact_today[mask]
    # time_window = time_window[mask]


    # pass_rate = mask.sum() / cfg["N_oversample"]
    # print(f"Pass rate: {pass_rate:.3f} — need N_oversample > {int(cfg['N_target'] / pass_rate)}")


    # n_filtered = len(mass)
    # assert n_filtered >= cfg["N_target"], (
    #     f"Only {n_filtered} passed — increase N_oversample to "
    #     f"~{int(cfg['N_target'] / pass_rate * 1.1):,}"
    # )

    # Trim to exactly N_target
    idx = slice(0, cfg["N_target"])
    mass = mass[idx]
    scale_radius = scale_radius[idx]
    v_perp = v_perp[idx]
    v_para = v_para[idx]
    angle_pos = angle_pos[idx]
    angle_delta = angle_delta[idx]
    angle_vel = angle_vel[idx]
    impact_param = impact_param[idx]
    impact_time = impact_time[idx]
    phi1_impact_today = phi1_impact_today[idx]
    time_window = time_window[idx]
 
    print(f"Proceeding with exactly {cfg['N_target']} perturbers")



    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    metadata = build_run_metadata(pass_rate = 100, n_filtered=0, n_target_actual=cfg["N_target"])
    save_run_metadata(metadata, output_dir)
 
    batch = []
    file_counter = 0
 
    for i in tqdm(range(cfg["N_target"])):
        inputs = {
            "mass": mass[i],
            "scale_radius": scale_radius[i],
            "phi1_impact_today": phi1_impact_today[i],
            "impact_time": impact_time[i],
            "impact_param": impact_param[i],
            "v_para": v_para[i],
            "v_perp": v_perp[i],
            "angle_pos": angle_pos[i],
            "angle_delta": angle_delta[i],
            "angle_vel": angle_vel[i],
            "delta_phi1": DELTA_PHI1,
            "time_window": time_window[i],
        }
 
        pert = sfr.create_perturber_dict(
            stream_unperturb["part_xv"],
            stream_unperturb["phi1"],
            potTotal,
            inputs["mass"],
            inputs["scale_radius"],
            inputs["phi1_impact_today"],
            inputs["impact_time"],
            inputs["impact_param"],
            inputs["v_para"],
            inputs["v_perp"],
            inputs["angle_pos"],
            inputs["angle_vel"],
            delta_phi1=inputs["delta_phi1"],
            time_window=inputs["time_window"],
        )
 
        batch.append({"inputs": inputs, "perturber": pert})
 
        if len(batch) == cfg["batch_size"]:
            filename = os.path.join(output_dir, f"perturbers_batch_{file_counter:04d}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {filename}")
            batch.clear()
            file_counter += 1
 
    if batch:
        filename = os.path.join(output_dir, f"perturbers_batch_{file_counter:04d}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved final batch {filename}")
 
 
if __name__ == "__main__":
    main()

# ###########################
# ##Apply ΔV detectability cut##
# ###########################

# G_units = 4.302e-6  # kpc (km/s)^2 / Msol
# v_rel_mag = np.sqrt(v_rel_perp**2 + v_rel_para**2)
# delta_V = 2 * G_units * mass_perturber / (impact_parameter * v_rel_mag)

# mask = delta_V > 1.0

# # Apply mask to all sampled arrays
# mass_perturber         = mass_perturber[mask]
# scaleradius_perturber  = scaleradius_perturber[mask]
# v_rel_perp             = v_rel_perp[mask]
# v_rel_para             = v_rel_para[mask]
# angle_pos_at_impact    = angle_pos_at_impact[mask]
# delta_angle            = delta_angle[mask]
# angle_vel_at_impact    = angle_vel_at_impact[mask]
# impact_parameter       = impact_parameter[mask]
# time_impact            = time_impact[mask]
# phi1_impact_today      = phi1_impact_today[mask]
# time_window            = time_window[mask]

# #N_filtered = len(mass_perturber)
# #print(f"Samples passing ΔV > 3 km/s cut: {N_filtered} / {N} ({N_filtered/N:.1%})")

# #mask = delta_V > 3.0
# pass_rate = mask.sum() / N_oversample
# print(f"Pass rate: {pass_rate:.3f} — need N_oversample > {int(N_target / pass_rate)}")

# # Fail loudly if oversampling wasn't enough
# N_filtered = len(mass_perturber)  
# assert N_filtered >= N_target, (
#     f"Only {N_filtered} passed — increase N_oversample to ~{int(N_target / pass_rate * 1.1):,}"
# )

# # Trim to exactly N_target
# mass_perturber        = mass_perturber[:N_target]
# scaleradius_perturber = scaleradius_perturber[:N_target]
# v_rel_perp            = v_rel_perp[:N_target]
# v_rel_para            = v_rel_para[:N_target]
# angle_pos_at_impact   = angle_pos_at_impact[:N_target]
# delta_angle           = delta_angle[:N_target]
# angle_vel_at_impact   = angle_vel_at_impact[:N_target]
# impact_parameter      = impact_parameter[:N_target]
# time_impact           = time_impact[:N_target]
# phi1_impact_today     = phi1_impact_today[:N_target]
# time_window           = time_window[:N_target]

# print(f"Proceeding with exactly {N_target} perturbers")

# ###############################
# ##Save perturber dictionaries##
# ###############################

# batch_size = 10_000
# output_dir = "/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/perturbers"
# os.makedirs(output_dir, exist_ok=True)

# batch = []
# file_counter = 0

# for i in tqdm(range(N_target)):
    
#     # Collect inputs for this perturber
#     inputs = {
#         "mass": mass_perturber[i],
#         "scale_radius": scaleradius_perturber[i],
#         "phi1_impact_today": phi1_impact_today[i],
#         "time_impact": time_impact[i],
#         "impact_parameter": impact_parameter[i],
#         "v_rel_para": v_rel_para[i],
#         "v_rel_perp": v_rel_perp[i],
#         "angle_pos_at_impact": angle_pos_at_impact[i],
#         "angle_vel_at_impact": angle_vel_at_impact[i],
#         "delta_phi1": delta_phi1,
#         "time_window": time_window[i],
#     }

#     # Generate perturber
#     pert = sfr.create_perturber_dict(
#         stream_unperturb['part_xv'],
#         stream_unperturb['phi1'],
#         potTotal,
#         inputs["mass"],
#         inputs["scale_radius"],
#         inputs["phi1_impact_today"],
#         inputs["time_impact"],
#         inputs["impact_parameter"],
#         inputs["v_rel_para"],
#         inputs["v_rel_perp"],
#         inputs["angle_pos_at_impact"],
#         inputs["angle_vel_at_impact"],
#         delta_phi1=inputs["delta_phi1"],
#         time_window=inputs["time_window"],
#     )

#     # Store BOTH inputs and output
#     batch.append({
#         "inputs": inputs,
#         "perturber": pert
#     })

#     # Save batch
#     if len(batch) == batch_size:
#         filename = os.path.join(
#             output_dir, f"perturbers_batch_{file_counter:04d}.pkl"
#         )
    
#         with open(filename, "wb") as f:
#             pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
    
#         print(f"Saved {filename}")
    
#         batch.clear()
#         file_counter += 1

# # Save remaining
# if len(batch) > 0:
#     filename = os.path.join(
#         output_dir, f"perturbers_batch_{file_counter:04d}.pkl"
#     )
    
#     with open(filename, "wb") as f:
#         pickle.dump(batch, f, protocol=pickle.HIGHEST_PROTOCOL)
    
#     print(f"Saved final batch {filename}")
