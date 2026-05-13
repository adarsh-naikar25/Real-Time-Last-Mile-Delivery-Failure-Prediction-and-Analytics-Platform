# ============================================================
# MAP VISUALIZATION
# FILE: src/map_visualization.py
# ============================================================

import folium

from streamlit_folium import st_folium

# ============================================================
# CREATE DELIVERY MAP
# ============================================================

def create_delivery_map(

    store_latitude,
    store_longitude,

    drop_latitude,
    drop_longitude,

    prediction_status

):

    # ========================================================
    # CENTER LOCATION
    # ========================================================

    center_lat = (

        store_latitude + drop_latitude

    ) / 2

    center_long = (

        store_longitude + drop_longitude

    ) / 2

    # ========================================================
    # CREATE MAP
    # ========================================================

    delivery_map = folium.Map(

        location=[center_lat, center_long],

        zoom_start=12

    )

    # ========================================================
    # STORE MARKER
    # ========================================================

    folium.Marker(

        location=[

            store_latitude,
            store_longitude

        ],

        popup="🏪 Store Location",

        tooltip="Store",

        icon=folium.Icon(

            color="green",

            icon="shopping-cart",

            prefix="fa"

        )

    ).add_to(delivery_map)

    # ========================================================
    # CUSTOMER MARKER
    # ========================================================

    marker_color = (

        "red"

        if prediction_status == "Failure"

        else "blue"

    )

    folium.Marker(

        location=[

            drop_latitude,
            drop_longitude

        ],

        popup="📍 Customer Location",

        tooltip="Customer",

        icon=folium.Icon(

            color=marker_color,

            icon="home",

            prefix="fa"

        )

    ).add_to(delivery_map)

    # ========================================================
    # DELIVERY ROUTE
    # ========================================================

    route_color = (

        "red"

        if prediction_status == "Failure"

        else "blue"

    )

    folium.PolyLine(

        locations=[

            [

                store_latitude,
                store_longitude

            ],

            [

                drop_latitude,
                drop_longitude

            ]

        ],

        color=route_color,

        weight=5,

        opacity=0.8

    ).add_to(delivery_map)

    # ========================================================
    # RETURN MAP
    # ========================================================

    return delivery_map

# ============================================================
# DISPLAY MAP
# ============================================================

def display_map(delivery_map):

    st_folium(

        delivery_map,

        width=1200,

        height=500

    )