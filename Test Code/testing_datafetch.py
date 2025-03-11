from psycopg2.extras import execute_batch# 🟢 Store News Function (Now Using YugabyteDB)

DB_URL = "postgresql://admin:xRZQdDx9_04APJdJc1CftnyqX9VZyX@ap-south-1.d67e1e29-cc8d-4b15-8cf9-4ea1e5bd8b9f.aws.yugabyte.cloud:5433/my_database?ssl=true&sslmode=verify-full&sslrootcert=./root.crt"