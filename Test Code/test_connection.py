import psycopg2

conn = psycopg2.connect(
    host="ap-south-1.d67e1e29-cc8d-4b15-8cf9-4ea1e5bd8b9f.aws.yugabyte.cloud",
    port=5433,
    dbname="my_database",  # Connect to default DB first
    user="admin",
    password="xRZQdDx9_04APJdJc1CftnyqX9VZyX",
    sslmode="verify-full",
    sslrootcert="./root.crt"
)
conn.autocommit=True
cursor = conn.cursor()
try:
    cursor.execute("select * from news limit 1;")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    # conn.commit()
    cursor.close()
    conn.close()
    print("fetched data successfully!")
except Exception as e:
    print("Faced an error:",e)
