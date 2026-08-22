import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from abridgeai.core.config import get_settings

url = get_settings().database_url
for old in ("postgresql+psycopg://", "postgresql://"):
    if url.startswith(old):
        url = url.replace(old, "postgresql+psycopg_async://", 1)
        break

async def main() -> None:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        reg = (await conn.execute(
            text("SELECT to_regclass('public.course_syllabus_imports')"))).scalar()
        print("table exists:", reg)
        cols = (await conn.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name='course_syllabus_imports' ORDER BY ordinal_position"))).all()
        for c in cols:
            print("   ", c[0], c[1])
        ver = (await conn.execute(text("SELECT version_num FROM alembic_version"))).scalar()
        print("alembic_version:", ver)
    await engine.dispose()

asyncio.run(main())
