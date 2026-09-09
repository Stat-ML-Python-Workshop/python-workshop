FROM python:3.12-slim
COPY requirements-ci.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SANDBOX_STUDENT=1
WORKDIR /tmp

