"""
APEX AI - Circuit Taxonomy, Downforce Clusters & Micro-Sector Profiling (Betterment 2)

Categorizes Formula 1 world championship circuits across 4 physics and engineering clusters:
1. Low-Downforce / High-Speed Power Circuits (Monza, Spa, Las Vegas, Baku)
2. High-Downforce / Thermal Degradation Circuits (Bahrain, Barcelona, Silverstone, Suzuka)
3. High-Downforce / Mechanical Traction Circuits (Monaco, Hungaroring, Singapore, Zandvoort)
4. Balanced Medium-Downforce Street/Hybrid Circuits (Miami, Melbourne, Montreal, Austin, Interlagos)

Includes micro-sector corner profiles (Apex Speed V_min, Brake Energy, Throttle Duty Cycle).
"""

from typing import Dict, Any

# Canonical circuit clusters with physical engineering characteristics
CIRCUIT_TAXONOMY = {
    # -------------------------------------------------------------
    # Cluster 1: Low-Downforce / High-Speed Power Tracks
    # High top speed, low aerodynamic drag, long straights, light aero trim
    # -------------------------------------------------------------
    "Italian Grand Prix": {
        "cluster_id": 1,
        "cluster_name": "Low-Downforce Power",
        "downforce_level": "Ultra-Low",
        "tire_stress_index": 3.2,
        "full_throttle_pct": 78.5,
        "overtaking_difficulty": "Low",
        "v_min_benchmark_kmh": 72.0,   # Prima Variante chicane apex
        "brake_energy_level": "High",
    },
    "Belgian Grand Prix": {
        "cluster_id": 1,
        "cluster_name": "Low-Downforce Power",
        "downforce_level": "Low-Medium",
        "tire_stress_index": 4.5,
        "full_throttle_pct": 71.2,
        "overtaking_difficulty": "Low",
        "v_min_benchmark_kmh": 165.0,  # Pouhon turn-in
        "brake_energy_level": "Medium",
    },
    "Las Vegas Grand Prix": {
        "cluster_id": 1,
        "cluster_name": "Low-Downforce Power",
        "downforce_level": "Ultra-Low",
        "tire_stress_index": 2.8,
        "full_throttle_pct": 76.0,
        "overtaking_difficulty": "Low",
        "v_min_benchmark_kmh": 85.0,
        "brake_energy_level": "High",
    },
    "Azerbaijan Grand Prix": {
        "cluster_id": 1,
        "cluster_name": "Low-Downforce Power",
        "downforce_level": "Low",
        "tire_stress_index": 3.4,
        "full_throttle_pct": 72.0,
        "overtaking_difficulty": "Low-Medium",
        "v_min_benchmark_kmh": 82.0,   # Turn 1 entrance
        "brake_energy_level": "High",
    },
    "Saudi Arabian Grand Prix": {
        "cluster_id": 1,
        "cluster_name": "Low-Downforce Power",
        "downforce_level": "Low-Medium",
        "tire_stress_index": 3.5,
        "full_throttle_pct": 79.0,
        "overtaking_difficulty": "Low-Medium",
        "v_min_benchmark_kmh": 115.0,  # Turn 13 banked hairpin
        "brake_energy_level": "High",
    },

    # -------------------------------------------------------------
    # Cluster 2: High-Downforce / Severe Thermal Degradation
    # High lateral g-forces, high tire surface blister risk, high aero load
    # -------------------------------------------------------------
    "Bahrain Grand Prix": {
        "cluster_id": 2,
        "cluster_name": "High-Downforce Thermal Degradation",
        "downforce_level": "High",
        "tire_stress_index": 4.8,
        "full_throttle_pct": 64.0,
        "overtaking_difficulty": "Medium",
        "v_min_benchmark_kmh": 68.0,   # Turn 1 hairpin
        "brake_energy_level": "High",
    },
    "Spanish Grand Prix": {
        "cluster_id": 2,
        "cluster_name": "High-Downforce Thermal Degradation",
        "downforce_level": "High",
        "tire_stress_index": 4.9,
        "full_throttle_pct": 65.5,
        "overtaking_difficulty": "High",
        "v_min_benchmark_kmh": 138.0,  # Turn 3 long right-hand carousel
        "brake_energy_level": "Medium",
    },
    "British Grand Prix": {
        "cluster_id": 2,
        "cluster_name": "High-Downforce Thermal Degradation",
        "downforce_level": "High",
        "tire_stress_index": 4.9,
        "full_throttle_pct": 70.0,
        "overtaking_difficulty": "Medium",
        "v_min_benchmark_kmh": 245.0,  # Copse corner high-speed entry
        "brake_energy_level": "Low",
    },
    "Japanese Grand Prix": {
        "cluster_id": 2,
        "cluster_name": "High-Downforce Thermal Degradation",
        "downforce_level": "High",
        "tire_stress_index": 5.0,
        "full_throttle_pct": 68.0,
        "overtaking_difficulty": "High",
        "v_min_benchmark_kmh": 182.0,  # Esses sequence turn 4
        "brake_energy_level": "Medium",
    },
    "Qatar Grand Prix": {
        "cluster_id": 2,
        "cluster_name": "High-Downforce Thermal Degradation",
        "downforce_level": "High",
        "tire_stress_index": 5.0,
        "full_throttle_pct": 70.5,
        "overtaking_difficulty": "Medium",
        "v_min_benchmark_kmh": 195.0,  # Triple-apex high lateral load
        "brake_energy_level": "Low-Medium",
    },

    # -------------------------------------------------------------
    # Cluster 3: High-Downforce / Mechanical Traction & Low Grip
    # Maximum aerodynamic wing angle, slow corners, traction-limited exit
    # -------------------------------------------------------------
    "Monaco Grand Prix": {
        "cluster_id": 3,
        "cluster_name": "Maximum-Downforce Mechanical Grip",
        "downforce_level": "Maximum",
        "tire_stress_index": 2.1,
        "full_throttle_pct": 45.0,
        "overtaking_difficulty": "Extreme",
        "v_min_benchmark_kmh": 46.0,   # Grand Hotel Hairpin (slowest corner in F1)
        "brake_energy_level": "Low",
    },
    "Hungarian Grand Prix": {
        "cluster_id": 3,
        "cluster_name": "Maximum-Downforce Mechanical Grip",
        "downforce_level": "Maximum",
        "tire_stress_index": 3.8,
        "full_throttle_pct": 58.0,
        "overtaking_difficulty": "High",
        "v_min_benchmark_kmh": 92.0,   # Turn 2 downhill off-camber
        "brake_energy_level": "High",
    },
    "Singapore Grand Prix": {
        "cluster_id": 3,
        "cluster_name": "Maximum-Downforce Mechanical Grip",
        "downforce_level": "Maximum",
        "tire_stress_index": 3.9,
        "full_throttle_pct": 52.0,
        "overtaking_difficulty": "High",
        "v_min_benchmark_kmh": 74.0,   # Marina Bay chicane
        "brake_energy_level": "High",
    },
    "Dutch Grand Prix": {
        "cluster_id": 3,
        "cluster_name": "Maximum-Downforce Mechanical Grip",
        "downforce_level": "High-Maximum",
        "tire_stress_index": 4.1,
        "full_throttle_pct": 62.0,
        "overtaking_difficulty": "High",
        "v_min_benchmark_kmh": 110.0,  # Hugenholtz banked corner
        "brake_energy_level": "Medium",
    },

    # -------------------------------------------------------------
    # Cluster 4: Balanced Hybrid / Stop-and-Go Street Circuits
    # Mixed high-speed sections, technical chicanes, medium drag compromise
    # -------------------------------------------------------------
    "Miami Grand Prix": {
        "cluster_id": 4,
        "cluster_name": "Hybrid Street / Stop-and-Go",
        "downforce_level": "Medium-High",
        "tire_stress_index": 3.7,
        "full_throttle_pct": 60.5,
        "overtaking_difficulty": "Medium",
        "v_min_benchmark_kmh": 78.0,
        "brake_energy_level": "High",
    },
    "Australian Grand Prix": {
        "cluster_id": 4,
        "cluster_name": "Hybrid Street / Stop-and-Go",
        "downforce_level": "Medium",
        "tire_stress_index": 3.6,
        "full_throttle_pct": 68.0,
        "overtaking_difficulty": "Medium",
        "v_min_benchmark_kmh": 115.0,  # Turn 9-10 chicane
        "brake_energy_level": "Medium",
    },
    "Canadian Grand Prix": {
        "cluster_id": 4,
        "cluster_name": "Hybrid Street / Stop-and-Go",
        "downforce_level": "Medium",
        "tire_stress_index": 3.3,
        "full_throttle_pct": 62.0,
        "overtaking_difficulty": "Low-Medium",
        "v_min_benchmark_kmh": 62.0,   # Hairpin
        "brake_energy_level": "Extreme", # Wall of champions final chicane
    },
    "Austrian Grand Prix": {
        "cluster_id": 4,
        "cluster_name": "Hybrid Street / Stop-and-Go",
        "downforce_level": "Medium",
        "tire_stress_index": 3.5,
        "full_throttle_pct": 69.0,
        "overtaking_difficulty": "Low-Medium",
        "v_min_benchmark_kmh": 76.0,   # Turn 3 uphill hairpin
        "brake_energy_level": "High",
    },
    "United States Grand Prix": {
        "cluster_id": 4,
        "cluster_name": "Hybrid Street / Stop-and-Go",
        "downforce_level": "High",
        "tire_stress_index": 4.4,
        "full_throttle_pct": 63.5,
        "overtaking_difficulty": "Low-Medium",
        "v_min_benchmark_kmh": 82.0,   # Turn 1 blind crest
        "brake_energy_level": "High",
    },
    "Mexico City Grand Prix": {
        "cluster_id": 4,
        "cluster_name": "Hybrid Street / Stop-and-Go",
        "downforce_level": "Maximum Aero (Low Air Density)",
        "tire_stress_index": 3.1,
        "full_throttle_pct": 55.0,
        "overtaking_difficulty": "Medium",
        "v_min_benchmark_kmh": 70.0,   # Stadium section entry
        "brake_energy_level": "Extreme", # Thin air makes brake cooling critical
    },
    "São Paulo Grand Prix": {
        "cluster_id": 4,
        "cluster_name": "Hybrid Street / Stop-and-Go",
        "downforce_level": "Medium-High",
        "tire_stress_index": 4.2,
        "full_throttle_pct": 67.0,
        "overtaking_difficulty": "Low",
        "v_min_benchmark_kmh": 105.0,  # Senna S downhill
        "brake_energy_level": "High",
    },
    "Emilia Romagna Grand Prix": {
        "cluster_id": 4,
        "cluster_name": "Hybrid Street / Stop-and-Go",
        "downforce_level": "High",
        "tire_stress_index": 4.0,
        "full_throttle_pct": 66.0,
        "overtaking_difficulty": "High",
        "v_min_benchmark_kmh": 118.0,  # Variante Alta
        "brake_energy_level": "Medium-High",
    },
    "Chinese Grand Prix": {
        "cluster_id": 4,
        "cluster_name": "Hybrid Street / Stop-and-Go",
        "downforce_level": "Medium-High",
        "tire_stress_index": 4.3,
        "full_throttle_pct": 60.0,
        "overtaking_difficulty": "Low-Medium",
        "v_min_benchmark_kmh": 68.0,   # Turn 1-2 snail sequence
        "brake_energy_level": "High",
    },
    "Abu Dhabi Grand Prix": {
        "cluster_id": 4,
        "cluster_name": "Hybrid Street / Stop-and-Go",
        "downforce_level": "Medium-High",
        "tire_stress_index": 3.9,
        "full_throttle_pct": 63.0,
        "overtaking_difficulty": "Medium",
        "v_min_benchmark_kmh": 85.0,
        "brake_energy_level": "High",
    }
}


OVERTAKING_DIFFICULTY_MAP = {
    "Low": 1.0,
    "Low-Medium": 1.5,
    "Medium": 2.0,
    "Medium-High": 2.5,
    "High": 3.0,
    "Extreme": 3.5,
}


CIRCUIT_ALIASES = {
    "monza": "Italian Grand Prix",
    "italy": "Italian Grand Prix",
    "spa": "Belgian Grand Prix",
    "belgium": "Belgian Grand Prix",
    "spa-francorchamps": "Belgian Grand Prix",
    "silverstone": "British Grand Prix",
    "britain": "British Grand Prix",
    "great britain": "British Grand Prix",
    "monaco": "Monaco Grand Prix",
    "monte carlo": "Monaco Grand Prix",
    "suzuka": "Japanese Grand Prix",
    "japan": "Japanese Grand Prix",
    "baku": "Azerbaijan Grand Prix",
    "azerbaijan": "Azerbaijan Grand Prix",
    "jeddah": "Saudi Arabian Grand Prix",
    "corniche": "Saudi Arabian Grand Prix",
    "saudi": "Saudi Arabian Grand Prix",
    "sakhir": "Bahrain Grand Prix",
    "bahrain": "Bahrain Grand Prix",
    "catalunya": "Spanish Grand Prix",
    "barcelona": "Spanish Grand Prix",
    "spain": "Spanish Grand Prix",
    "hungaroring": "Hungarian Grand Prix",
    "budapest": "Hungarian Grand Prix",
    "hungary": "Hungarian Grand Prix",
    "zandvoort": "Dutch Grand Prix",
    "netherlands": "Dutch Grand Prix",
    "singapore": "Singapore Grand Prix",
    "marina bay": "Singapore Grand Prix",
    "las vegas": "Las Vegas Grand Prix",
    "vegas": "Las Vegas Grand Prix",
    "cota": "United States Grand Prix",
    "austin": "United States Grand Prix",
    "usa": "United States Grand Prix",
    "mexico": "Mexico City Grand Prix",
    "mexico city": "Mexico City Grand Prix",
    "hermanos rodriguez": "Mexico City Grand Prix",
    "interlagos": "São Paulo Grand Prix",
    "sao paulo": "São Paulo Grand Prix",
    "brazil": "São Paulo Grand Prix",
    "imola": "Emilia Romagna Grand Prix",
    "albert park": "Australian Grand Prix",
    "melbourne": "Australian Grand Prix",
    "australia": "Australian Grand Prix",
    "gilles villeneuve": "Canadian Grand Prix",
    "montreal": "Canadian Grand Prix",
    "canada": "Canadian Grand Prix",
    "shanghai": "Chinese Grand Prix",
    "china": "Chinese Grand Prix",
    "red bull ring": "Austrian Grand Prix",
    "spielberg": "Austrian Grand Prix",
    "austria": "Austrian Grand Prix",
    "losail": "Qatar Grand Prix",
    "lusail": "Qatar Grand Prix",
    "qatar": "Qatar Grand Prix",
    "yas marina": "Abu Dhabi Grand Prix",
    "abu dhabi": "Abu Dhabi Grand Prix",
    "miami": "Miami Grand Prix",
}


def get_circuit_features(event_name: str) -> Dict[str, Any]:
    """Return circuit physics, downforce regime, and micro-sector metrics."""
    cleaned = event_name.strip().lower()
    canonical_name = None

    # 1. Alias lookup
    for alias, target in CIRCUIT_ALIASES.items():
        if alias in cleaned:
            canonical_name = target
            break

    # 2. Key matching
    res = None
    if canonical_name and canonical_name in CIRCUIT_TAXONOMY:
        res = dict(CIRCUIT_TAXONOMY[canonical_name])
    else:
        for name, data in CIRCUIT_TAXONOMY.items():
            if name.lower() in cleaned or cleaned in name.lower():
                res = dict(data)
                break

    if res is None:
        # Default fallback: cluster 4 (balanced hybrid)
        res = {
            "cluster_id": 4,
            "cluster_name": "Balanced Hybrid",
            "downforce_level": "Medium",
            "tire_stress_index": 3.8,
            "full_throttle_pct": 64.0,
            "overtaking_difficulty": "Medium",
            "v_min_benchmark_kmh": 85.0,
            "brake_energy_level": "Medium",
        }

    res["overtaking_difficulty_idx"] = OVERTAKING_DIFFICULTY_MAP.get(
        res.get("overtaking_difficulty", "Medium"), 2.0
    )
    return res


if __name__ == "__main__":
    print(f"Total profiled circuits: {len(CIRCUIT_TAXONOMY)}")
    test = get_circuit_features("Monaco Grand Prix")
    print("Monaco Profile:", test)
    test2 = get_circuit_features("Italian Grand Prix")
    print("Monza Profile:", test2)
