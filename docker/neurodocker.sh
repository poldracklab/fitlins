neurodocker generate docker \
  --pkg-manager apt \
  --base neurodebian:stretch \
  --arg DEBIAN_FRONTEND=noninteractive \
  --label maintainer="Shashank Bansal" \
  --env MKL_NUM_THREADS="1" OMP_NUM_THREADS="1" \
  --install apt-utils unzip curl graphviz \
          wget git-annex-standalone git vim \
          tig git-annex-remote-rclone \
  --user neuro \
  --workdir "/home/neuro" \
  --miniconda version=latest create_env=neuro \
  conda_install="python=3.9 numpy pytest
                  traits pandas matplotlib
		  scikit-learn scikit-image
		  seaborn nbformat nb_conda" \
  --run 'conda install -y --channel leej3 --name neuro "afni-minimal" && sync && conda clean -y --all && sync' \
  --copy . "/src/fitlins" \
  --user  root \
  --run 'mkdir /work && chown -R neuro /src /work && chmod a+w /work' \
  --user neuro \
  --arg VERSION \
  --run 'echo "$VERSION" > /src/fitlins/VERSION' \
  --run 'sed -i -e '"'"'s/crashfile_format = pklz/crashfile_format = txt/'"'"' /src/fitlins/fitlins/data/nipype.cfg' \
  --run-bash 'source activate neuro \
      && pip install --no-cache-dir -r /src/fitlins/requirements.txt \
      && rm -rf ~/.cache/pip/* \
      && sync \
      && sed -i "$isource activate neuro" $ND_ENTRYPOINT' \
  --run-bash 'source activate neuro \
      && pip install --no-cache-dir  \
      "/src/fitlins[all]" \
      && rm -rf ~/.cache/pip/* \
      && sync \
      && sed -i "$isource activate neuro" $ND_ENTRYPOINT' \
  --workdir "/work" \
  --entrypoint "/neurodocker/startup.sh fitlins" \
  --arg BUILD_DATE \
  --arg VCS_REF \
  --label org.label-schema.build-date='$BUILD_DATE' \
      org.label-schema.name="FitLins" \
      org.label-schema.description="FitLins - Fit Linear Models to BIDS datasets" \
      org.label-schema.url="http://github.com/poldracklab/fitlins" \
      org.label-schema.vcs-ref='$VCS_REF' \
      org.label-schema.vcs-url="https://github.com/poldracklab/fitlins" \
      org.label-schema.version='$VERSION' \
      org.label-schema.schema-version="1.0"  > Dockerfile
