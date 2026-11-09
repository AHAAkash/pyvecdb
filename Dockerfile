FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .
VOLUME /data
EXPOSE 8000
# Listening on 0.0.0.0 inside the container: always set PYVECDB_API_KEY, and publish the
# port to localhost only unless you mean to expose it:  docker run -p 127.0.0.1:8000:8000
CMD ["pyvecdb", "serve", "--host", "0.0.0.0", "--data", "/data/pyvecdb.npz"]
