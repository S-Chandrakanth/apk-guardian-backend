from fastapi import APIRouter

from app.api import routes_auth, routes_library, routes_scans, routes_system

api_router = APIRouter()
api_router.include_router(routes_system.router)
api_router.include_router(routes_auth.router)
api_router.include_router(routes_scans.router)
api_router.include_router(routes_library.router)
