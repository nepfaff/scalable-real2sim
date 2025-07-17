FROM robotlocomotion/drake:1.35.0

ENV WORKSPACE /root/workspace

# build essentials
RUN apt update && \
    apt install -y curl build-essential git && \
    apt install -y libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev \
    liblzma-dev && \
    apt install -y software-properties-common libtbb-dev && \
    apt install -y libeigen3-dev python3.10-venv ffmpeg \
    clangd-12 wget lsb-release cmake libboost-all-dev gdb python3-dbg && \
    update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-12 100

RUN curl https://pyenv.run | bash
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

RUN pyenv global system && \
    pyenv install 3.10.12 && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc



WORKDIR $WORKSPACE
