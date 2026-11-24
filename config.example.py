API_KEY = "PASTE_YOUR_TOMTOM_API_KEY_HERE"

MANILA_ROADS = [
    ("Espana Blvd", 14.6091, 120.9926, 4, 1800),
    ("Quezon Blvd Quiapo", 14.5988, 120.9836, 4, 1800),
    ("Recto Ave", 14.6038, 120.9830, 3, 1600),
    ("Taft Ave Ermita", 14.5764, 120.9881, 3, 1700),
    ("Roxas Blvd Manila", 14.5736, 120.9822, 4, 1900),
    ("Rizal Ave Sta Cruz", 14.6145, 120.9828, 3, 1600),
    ("Quezon Ave", 14.6368, 121.0253, 4, 1900),
    ("EDSA Ortigas", 14.5815, 121.0537, 5, 2000),
    ("EDSA Guadalupe", 14.5673, 121.0456, 5, 2000),
    ("C5 Bagong Ilog", 14.5885, 121.0789, 4, 1900),
]


def estimate_vehicle_volume(current_speed, freeflow_speed, lanes, cap_per_lane):
    speed_ratio = min(current_speed / freeflow_speed, 1.0)
    congestion_density = max(1.0 - (speed_ratio * 0.85), 0.15)
    return int(lanes * cap_per_lane * congestion_density)