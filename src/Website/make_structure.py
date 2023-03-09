import os

def create_directory_structure(project_name : str) -> None:
    project_directory = os.path.join(os.getcwd(), project_name)

    # create project directory
    os.mkdir(project_directory)

    # create app directory
    app_directory = os.path.join(project_directory, 'app')
    os.mkdir(app_directory)

    # create api directory
    api_directory = os.path.join(app_directory, 'api')
    os.mkdir(api_directory)

    # create routes directory
    routes_directory = os.path.join(api_directory, 'routes')
    os.mkdir(routes_directory)

    # create utils directory
    utils_directory = os.path.join(api_directory, 'utils')
    os.mkdir(utils_directory)

    # create config directory
    config_directory = os.path.join(app_directory, 'config')
    os.mkdir(config_directory)

    # create tests directory
    tests_directory = os.path.join(app_directory, 'tests')
    os.mkdir(tests_directory)

    # create test_routes directory
    test_routes_directory = os.path.join(tests_directory, 'test_routes')
    os.mkdir(test_routes_directory)

    # create __init__.py files
    open(os.path.join(app_directory, '__init__.py'), 'w').close()
    open(os.path.join(api_directory, '__init__.py'), 'w').close()
    open(os.path.join(api_directory, 'models.py'), 'w').close()
    open(os.path.join(routes_directory, '__init__.py'), 'w').close()
    open(os.path.join(utils_directory, '__init__.py'), 'w').close()
    open(os.path.join(config_directory, '__init__.py'), 'w').close()
    open(os.path.join(tests_directory, '__init__.py'), 'w').close()
    open(os.path.join(test_routes_directory, '__init__.py'), 'w').close()

    # create main.py file
    with open(os.path.join(app_directory, 'main.py'), 'w') as f:
        f.write("from fastapi import FastAPI\n\n")
        f.write("app = FastAPI()\n\n")
        f.write("if __name__ == '__main__':\n")
        f.write("    import uvicorn\n")
        f.write("    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)\n")

    # create database.py file
    with open(os.path.join(config_directory, 'database.py'), 'w') as f:
        f.write("from sqlalchemy import create_engine\n")
        f.write("from sqlalchemy.orm import declarative_base\n")
        f.write("from sqlalchemy.orm import sessionmaker\n\n")
        f.write("SQLALCHEMY_DATABASE_URL = 'sqlite:///./test.db'\n")
        f.write("engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={'check_same_thread': False})\n")
        f.write("SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)\n\n")
        f.write("Base = declarative_base()\n")

    # create logging.py file
    with open(os.path.join(config_directory, 'logging.py'), 'w') as f:
        f.write("import logging\n\n")
        f.write("logging.basicConfig(\n")
        f.write("    level=logging.DEBUG,\n")
        f.write("    format='%(asctime)s %(levelname)s %(message)s',\n")
        f.write("    handlers=[\n")
        f.write("        logging.StreamHandler()\n")
        f.write("    ]\n")
        f.write(")\n")

    # create users.py file
    with open(os.path.join(routes_directory, 'users.py'), 'w') as f:
        f.write("from fastapi import APIRouter\n\n")
        f.write("router = APIRouter()\n\n")
        f.write("@router.get('/users/')\n")
        f.write("async def read_users():\n")
        f.write(" return [{'username': 'alice'}, {'username': 'bob'}]\n")

    print(f"Created project structure for {project_name}.")
    
if __name__ == '__main__':
    print("Creating project structure...")
    create_directory_structure('fastapi_website')
    print("Done.")