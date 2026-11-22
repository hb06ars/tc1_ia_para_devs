import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Inicia a interface web para classificação de CSV.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("src.cancer_de_mama.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
