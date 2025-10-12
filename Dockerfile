FROM python:3.13.5-slim-bookworm as base

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
ENV PATH=/home/ftuser/.local/bin:$PATH
ENV FT_APP_ENV="docker"

# Prepare environment
RUN mkdir /freqtrade \
  && apt-get update \
  && apt-get -y install sudo libatlas3-base curl sqlite3 libgomp1 \
  && apt-get clean \
  && useradd -u 1000 -G sudo -U -m -s /bin/bash ftuser \
  && chown ftuser:ftuser /freqtrade \
  # Allow sudoers
  && echo "ftuser ALL=(ALL) NOPASSWD: /bin/chown" >> /etc/sudoers

WORKDIR /freqtrade

# Copy scripts and set permissions before switching to ftuser
COPY generate-config.py /freqtrade/generate-config.py
COPY entrypoint.sh /freqtrade/entrypoint.sh

RUN chmod +x /freqtrade/entrypoint.sh && \
    mkdir -p /freqtrade/user_data && \
    chown -R ftuser:ftuser /freqtrade/user_data

# Install dependencies
FROM base as python-deps
RUN  apt-get update \
  && apt-get -y install build-essential libssl-dev git libffi-dev libgfortran5 pkg-config cmake gcc \
  && apt-get clean \
  && pip install --upgrade pip wheel

# Install TA-lib
COPY build_helpers/* /tmp/
RUN cd /tmp && /tmp/install_ta-lib.sh && rm -r /tmp/*ta-lib*
ENV LD_LIBRARY_PATH /usr/local/lib

# Install dependencies
COPY --chown=ftuser:ftuser requirements.txt requirements-hyperopt.txt /freqtrade/
USER ftuser
RUN  pip install --user --no-cache-dir "numpy<3.0" \
  && pip install --user --no-cache-dir -r requirements-hyperopt.txt \
  && pip install --user --no-cache-dir ta arrow finta scikit-optimize psycopg2-binary

# Copy dependencies to runtime-image
FROM base as runtime-image
COPY --from=python-deps /usr/local/lib /usr/local/lib
ENV LD_LIBRARY_PATH /usr/local/lib

COPY --from=python-deps --chown=ftuser:ftuser /home/ftuser/.local /home/ftuser/.local

# Copy scripts from base stage
COPY --from=base /freqtrade/generate-config.py /freqtrade/generate-config.py
COPY --from=base /freqtrade/entrypoint.sh /freqtrade/entrypoint.sh

USER ftuser
# Install and execute
COPY --chown=ftuser:ftuser . /freqtrade/

RUN pip install -e . --user --no-cache-dir --no-build-isolation \
  && freqtrade install-ui

# Fix permissions for scripts
RUN chmod +x /freqtrade/entrypoint.sh && \
    mkdir -p /freqtrade/user_data

ENTRYPOINT ["/freqtrade/entrypoint.sh"]
# Default to webserver mode for Render
CMD ["trade"]
