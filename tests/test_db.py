from src.config import settings

print(settings.database_url)

from src.vectordb.postgres_database import PostgresDatabase

db = PostgresDatabase(settings.database_url)
db.initialize_database()
print("Connected!")