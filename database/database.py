# ============================================================
# MYSQL DATABASE CONNECTION
# ============================================================

import mysql.connector

# ============================================================
# CONNECT TO MYSQL
# ============================================================

db = mysql.connector.connect(

    host="localhost",

    user="root",

    password="Adarsh@1978",

    database="delivery_prediction_db"

)

# ============================================================
# CURSOR
# ============================================================

cursor = db.cursor()

print("✅ MySQL Database Connected Successfully")

# ============================================================
# INSERT PREDICTION FUNCTION
# ============================================================

def insert_prediction(

    weather,
    traffic,
    distance,
    prediction,
    probability

):

    query = """

    INSERT INTO prediction_history (

        weather,
        traffic,
        distance,
        prediction,
        probability

    )

    VALUES (%s, %s, %s, %s, %s)

    """

    values = (

        weather,
        traffic,
        distance,
        prediction,
        probability

    )

    cursor.execute(query, values)

    db.commit()

    print("✅ Prediction Saved Successfully")

# ============================================================
# FETCH HISTORY FUNCTION
# ============================================================

def fetch_history():

    query = """

    SELECT * FROM prediction_history

    ORDER BY created_at DESC

    """

    cursor.execute(query)

    rows = cursor.fetchall()

    return rows