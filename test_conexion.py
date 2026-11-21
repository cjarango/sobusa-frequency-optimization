from sqlalchemy import create_engine, text

engine = create_engine("postgresql+psycopg://postgres:postgres123@localhost:5432/frecuencias")

with engine.connect() as con:
    version = con.execute(text("SELECT version()")).scalar()
    tablas = con.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")).fetchall()
    rutas = con.execute(text("SELECT * FROM rutas")).fetchall()

print("Version:", version)
print("Tablas:", [t[0] for t in tablas])
print("Rutas:", rutas)