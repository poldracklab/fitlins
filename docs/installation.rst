.. include:: links.rst

.. _installation:

------------
Installation
------------

There are three ways to use FitLins: in a `Docker Container`_, a `Singularity Container`_, or in a
`Manually Prepared Environment (Python 3.6+)`_.

For the sake of consistency, using containers is highly recommended.
While some command-line options are discussed in this page, it is not intended to be exhausitve.
For a full set of options, see :ref:`Usage`.

Data Organization
=================

To make the examples in this document concrete, let's suppose that we have the following
structure::

  /data/
    raw/
      dsX/
      dsY/
      ...
    prep/
      dsX/
        fmriprep/
      dsY/
      ...
    analyzed/
      dsX/
      dsY/
  /scratch

Here, we have an original BIDS dataset ``/data/raw/dsX``, a `BIDS Derivatives`_ dataset (such as
would be produced by fMRIPrep_) at ``/data/prep/dsX/fmriprep``, and a target directory for storing
FitLins analyses at ``/data/analyzed/dsX``.

.. note::

  There are many ways to organize related datasets while conforming to the BIDS standard.
  This is a simple one that keeps each stage of processing in a separate directory;
  you may prefer to keep all stages for a single dataset in a given directory, or even to nest
  related data structures.
  Each of these approaches is valid, and only minor changes should be needed to the examples below
  to accommodate your preferences.

Additionally, we have a ``/scratch`` directory for storing the intermediate results of a Nipype_
workflow.
This can be useful for debugging or resuming interrupted runs.

Docker Container
================

In order to run FitLins in a Docker container, Docker must be `installed
<https://docs.docker.com/engine/installation/>`_.

Getting the Docker image
------------------------

To download a specific version of FitLins, use::

  docker pull poldracklab/fitlins:<VERSION>

``:<VERSION>`` is a *tag*, in Docker terminology, and all tags may be found on `FitLins' DockerHub
page <https://hub.docker.com/r/poldracklab/fitlins/tags/>`_.
``:latest`` refers to the most recent release, and ``:master`` refers to the most recent changes
in the GitHub repository, but *only as of the last time you ran* ``docker pull``.
We highly recommend using a tag for a specific version, to reduce opportunities for confusion.

Running the Docker image
------------------------

Docker commands take the form::

  docker run <DOCKER_OPTIONS> \
    poldracklab/fitlins:<VERSION> \
    <FITLINS_OPTIONS>

The most important thing for running FitLins in Docker is to mount directories inside the Docker
container, so that the FitLins program running inside the container is able to read files from
and write files to those directories.
As noted above, there are at least three relevant directories:
at least one data directory, containing original and preprocessed datasets;
an output directory, to store FitLins results;
and a working directory, to store the intermediate results of Nipype_ workflows.

These must be mounted using the ``-v`` Docker option.
For example::

  -v /data/raw/dsX:/bids:ro
  -v /data/prep/dsX/fmriprep:/prep:ro
  -v /data/analyzed/dsX:/out
  -v /scratch:/scratch

Note that on the left of each colon (``:``) is the true path to your data.
On the right is where those files will be available inside the container;
this is an arbitrary choice, but we're using short paths for brevity.
The ``:ro`` directive for the input datasets indicates they should be made *read-only* to the
container, which is a good precaution against bugs in FitLins from damaging your inputs.

So a basic command would look like::

  docker run --rm -it \
      -v /data/raw/dsX:/bids:ro \
      -v /data/prep/dsX/fmriprep:/prep:ro \
      -v /data/analyzed/dsX:/out \
      -v /scratch:/scratch \
    poldracklab/fitlins:0.4.0 \
      /bids /out dataset -d /prep -w /scratch

Singularity Container
=====================

For security reasons, many :abbr:`HPC (High Performance Computing)`/:abbr:`HTC (High Throughput
Computing)` environments do not allow Docker containers, but increasingly many are now allowing
`Singularity <https://github.com/singularityware/singularity>`_ containers.

Getting a Singularity image
---------------------------

We hope in the near future to host official Singularity images on `Singularity Hub`_, but it is
currently necessary for users to generate their own images from the Docker images we provide.

For Singularity version 2.5 or higher, ``singularity build`` is sufficient::

  singularity build /my_images/fitlins-<VERSION>.simg \
    docker://poldracklab/fitlins:<VERSION>

Please see `Getting the Docker image`_ for a discussion on versions and tags.

To target older versions of Singularity, you can use `docker2singularity
<https://github.com/singularityware/docker2singularity>`_, which is itself must be run in
Docker::

    docker run --privileged -t --rm \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v /my_images:/output \
      singularityware/docker2singularity \
      poldracklab/fitlins:<VERSION>

For Windows users::

    docker run --privileged -t --rm \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v D:\host\path\where\to\output\singularity\image:/output \
      singularityware/docker2singularity \
      poldracklab/fitlins:<VERSION>

This image may now be transfered to your cluster.

Running a Singularity image
---------------------------

Singularity installations often permit using filesystem paths without translation::

    singularity run --cleanenv /my_images/fitlins-0.4.0.simg \
      /data/raw/dsX /data/analyzed/dsX dataset \
      -d /data/prep/dsX/fmriprep \
      -w /scratch

.. note::

   Singularity by default `exposes all environment variables from the host inside 
   the container <https://github.com/singularityware/singularity/issues/445>`_.
   Because of this your host libraries (such as nipype) could be accidentally used 
   instead of the ones inside the container - if they are included in ``PYTHONPATH``.
   To avoid such situation we recommend using the ``--cleanenv`` singularity flag 
   in production use, as in the above example.

In some cases, your directories may not be available inside the container, in which
case the ``-B`` flag works very similarly to the ``-v`` flag in Docker::

  singularity run --cleanenv \
      -B /data/raw/dsX:/data/raw/dsX \
      -B /data/prep/dsX/fmriprep:/data/prep/dsX/fmriprep \
      -B /data/analyzed/dsX:/data/analyzed/dsX \
      -B /scratch:/scratch \
    /my_images/fitlins-0.4.0.simg \
      /data/raw/dsX /data/analyzed/dsX dataset \
      -d /data/prep/dsX/fmriprep \
      -w /scratch


Manually Prepared Environment (Python 3.6+)
===========================================

Because FitLins sometimes depends on unreleased versions of upstream libraries
(in particular, Nistats_, PyBIDS_ and Nipype_), it is inadvisable to install
directly into your base Python environment.

If you have Anaconda_/Miniconda_ installed, you can create a new environment with::

  conda create -n fitlins python=3.6
  conda activate fitlins

In most Python installations, you can create an environment with
virtualenv_::

  pip install --upgrade virtualenv
  virtualenv --python=python3.6 fitlins.venv
  source fitlins.venv/bin/activate

Once inside the environment::

  pip install fitlins

You can now run FitLins::

    fitlins /data/raw/dsX /data/analyzed/dsX dataset \
      -d /data/prep/dsX/fmriprep \
      -w /scratch

