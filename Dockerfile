FROM python:3.11-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HOME=/app/data/home
WORKDIR /app
COPY requirements-docker.txt /app/
RUN python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements-docker.txt \
    && python -m pip check \
    && python -c "import imageio_ffmpeg,shutil,os; shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(),'/usr/local/bin/ffmpeg'); os.chmod('/usr/local/bin/ffmpeg',0o755)"
RUN groupadd --gid 10001 voiceid && useradd --uid 10001 --gid voiceid --no-create-home voiceid \
    && mkdir -p /app/data /app/models && chown -R voiceid:voiceid /app
COPY --chown=voiceid:voiceid voice_id_mvp /app/voice_id_mvp
COPY --chown=voiceid:voiceid scripts /app/scripts
USER voiceid
CMD ["python", "-m", "uvicorn", "voice_id_mvp.services.workspace_app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
COPY --chown=voiceid:voiceid tests /app/tests
COPY --chown=voiceid:voiceid pytest.ini /app/
CMD ["python", "-m", "pytest", "-q"]
