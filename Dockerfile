FROM julia:1.11.4

RUN apt-get update \
    && apt-get install -y openssh-server apt-transport-https ca-certificates git\
    && rm -rf /var.lib/apt/lists/*

# Copy package in docker
COPY . /home/WWTP-DataAssimilation-ModelComparison
RUN echo "export PATH=${PATH}:/usr/local/julia/bin" >> /etc/profile

# Install all the packages
WORKDIR /home/WWTP-DataAssimilation-ModelComparison
RUN julia -e 'using Pkg; Pkg.add("DrWatson"); Pkg.activate("."); Pkg.instantiate()'
ENV JULIA_PROJECT=/home/WWTP-DataAssimilation-ModelComparison/