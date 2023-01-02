

# from http import HTTPStatus
from pathlib import Path

import alembic.command
import alembic.config
# import pydantic
from api_adl.config import settings
from fastapi import FastAPI
from loguru import logger
# from sqlmodel import Session
from starlette.middleware.cors import CORSMiddleware

# from typing import Optional


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f'{settings.API_V1_STR}/openapi.json'
)

# Allow all CORS ORIGINS endpoints
if backend_cors_origins := settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in backend_cors_origins],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

# If we want to update alembic on start of the app
if settings.UPDATE_ALEMBIC:
    @app.on_event('startup')
    def alembic_upgrade():
        logger.info('Attempting to upgrade alembic on startup')

        try:
            alembic_ini_path = Path(__file__).parent / 'alembic.ini'
            alembic_cfg = alembic.config.Config(str(alembic_ini_path))
            alembic_cfg.set_main_option('sqlalchemy.url', settings.DATABASE_URI)
            alembic.command.upgrade(alembic_cfg, 'head')
            logger.info('Successfully upgraded alembic on startup')

        except Exception as e:

            logger.exception('Alembic upgrade failed on startup')

# And now we include the api routers with the prefix of v1
# app.include_router(api_router, prefix=settings.API_V1_STR)


if __name__ == '__main__':
    import argparse

    import uvicorn

    # Define some arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--host',
        help='The host to run the server',
        default='4000'
    )
    parser.add_argument(
        '--port',
        help='The port to run the server',
        default='127.0.0.1'
    )

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
