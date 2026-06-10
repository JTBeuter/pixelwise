from sqlalchemy import create_engine
from app.models import Base
import os
from dotenv import load_dotenv

load_dotenv()

# Lade die URL und ersetze den Platzhalter mit dem echten Passwort
db_url = os.getenv("DATABASE_URL").replace("${DB_PASSWORD}", os.getenv("DB_PASSWORD"))
engine = create_engine(db_url)

Base.metadata.create_all(engine)
print("Tabelle erfolgreich erstellt!")
