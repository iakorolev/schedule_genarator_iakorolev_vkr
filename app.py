"""Точка входа для локального запуска Flask-приложения из корня проекта."""

from webapp.app import app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
