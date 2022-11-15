# Reproduction study: Argument Similarity with AMR

This project reproduces the paper "Explainable Unsupervised Argument Similarity Rating with Abstract Meaning Representation and Conclusion Generation"(Opitz et al., ArgMining 2021) [(link)](project_docs/OpitzEtAl21.pdf). 

To reproduce the results of the study, two repositories originally written by the author are required: [a more general repository](https://github.com/flipz357/amr-metric-suite) that contains AMR metrics and the [paper specific repository](https://github.com/Heidelberg-NLP/amr-argument-sim) that contains the AMR metric for argument similarity. As we focus on the results obtained in the paper, we provide both repositories cloned in [```repro_repos/```](repro_repos/).

It will also include an extension. Currently a work in progress as we gather ideas.

This project is a so called "Project Module", part of the [Cogniive Systems](https://www.uni-potsdam.de/en/studium/what-to-study/master/masters-courses-from-a-to-z/cognitive-systems) Master program at the [University of Potsdam](https://www.uni-potsdam.de/en/university-of-potsdam). Contributors: [Tamara Atanasoska](https://github.com/TamaraAtanasoska), [Emanuele De Rossi](https://github.com/EmanueleDeRossi1) and [Galina Ryazanskaya](https://github.com/flying-bear).

## Setup and usage

### Environment recreation 

In the folder ```setup/``` you can find the respective environment replication and package requirements files. There are two options:

1. You can run ```pip install -r setup/requirements.txt``` to install the necessary packages in your existing environment.

2. If you are using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to manage your virtual environments, you can replicate and activate the full exact environment with the following commands:

   ```
   conda env create --name <name> --file setup/conda.yml
   conda activate <name>
   ```

