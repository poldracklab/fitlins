FROM continuumio/miniconda3:4.3.27p0

# Installing precomputed python packages
RUN conda install -y mkl=2018.0.1 mkl-service git=2.9.3;  sync &&\
    conda install -y numpy=1.14.0 \
                     scipy=1.0.0 \
                     scikit-learn=0.19.1 \
                     matplotlib=2.1.2 \
                     seaborn=0.8.1 \
                     pandas=0.22.0 \
                     libxml2=2.9.7 \
                     libxslt=1.1.29\
                     traits=4.6.0; sync &&  \
    chmod +x /opt/conda/bin/*; sync && \
    conda clean --all -y; sync && \
    conda clean -tipsy && sync

# Precaching fonts
RUN python -c "from matplotlib import font_manager"

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /root/

# Installing dev requirements (packages that are not in pypi)
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# Installing fitlins
COPY . /root/src/fitlins
RUN cd /root/src/fitlins && \
    pip install .[all] && \
    rm -rf ~/.cache/pip

RUN ldconfig

WORKDIR /root/src/fitlins

ENTRYPOINT ["/opt/conda/bin/fitlins"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
